"""用真实官方工具和 SQLite 终态验证适配判分，只替换判分模型的输入输出。"""

import json
from copy import deepcopy

import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后运行官方判分测试")

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage  # noqa: E402
from tau2.data_model.simulation import TerminationReason  # noqa: E402
from tau2.data_model.tasks import Action, EvaluationCriteria, InitialState, Task  # noqa: E402
from tau2.domains.retail.data_model import RetailDB  # noqa: E402
from tau2.domains.retail.environment import get_environment  # noqa: E402
from tau2.evaluator import evaluator_nl_assertions  # noqa: E402
from tau2.evaluator.evaluator_env import EnvironmentEvaluator  # noqa: E402

from ttwill.backend import RetailBackend  # noqa: E402
from ttwill.bench import evaluation  # noqa: E402
from ttwill.bench.evaluation import EvaluationIncompleteError, evaluate_task  # noqa: E402


@pytest.fixture
def cancel_task() -> Task:
    """构造具有真实官方取消参考动作的零售任务。

    Returns:
        目标为取消 #W0000001 订单、按数据库终态与可选自然语言断言判分的任务。
    """
    return Task(
        id="cancel-small-order",
        user_scenario={"instructions": "取消 #W0000001 订单。"},
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="cancel",
                    name="cancel_pending_order",
                    arguments={"order_id": "#W0000001", "reason": "no longer needed"},
                )
            ],
            reward_basis=["DB", "NL_ASSERTION"],
        ),
    )


def cancel_order(backend: RetailBackend) -> dict:
    """通过真实准备和提交取得订单取消后的完整业务快照。

    Args:
        backend: 当前测试已初始化的独立业务后端。

    Returns:
        已包含退款历史的取消终态。
    """
    args = {"order_id": "#W0000001", "reason": "no longer needed"}
    prepared = backend.prepare("u1", "cancel_pending_order", args, "evaluation-cancel")
    backend.commit("u1", prepared["operation_id"], args)
    return backend.export_snapshot()


def test_real_backend_snapshot_passes_without_visible_tool_calls(
    backend: RetailBackend, seed_snapshot: dict, cancel_task: Task
) -> None:
    """纯文本对话仍按真实业务终态通过，不要求内部工具出现在 τ² 轨迹中。"""
    predicted = cancel_order(backend)
    original = deepcopy(seed_snapshot)
    before = deepcopy(predicted)
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        predicted,
        [
            UserMessage(role="user", content="确认取消"),
            AssistantMessage(role="assistant", content="已经取消。"),
        ],
        termination_reason=TerminationReason.USER_STOP,
    )
    assert result["db_match"] and result["task_success"]
    assert result["reference_hash"] == result["predicted_hash"]
    assert result["reward_info"]["reward_breakdown"] == {"DB": 1.0, "NL_ASSERTION": 1.0}
    assert not result["reward_info"]["action_checks"][0]["action_match"]
    assert seed_snapshot == original and predicted == before


def test_unmodified_snapshot_fails_despite_claimed_success(
    seed_snapshot: dict, cancel_task: Task
) -> None:
    """回复声称办妥不能代替真实业务库的取消与退款终态。"""
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        seed_snapshot,
        [AssistantMessage(role="assistant", content="订单已经取消，退款成功。")],
        termination_reason=TerminationReason.AGENT_STOP,
    )
    assert not result["db_match"] and not result["task_success"]
    assert result["reward_info"]["reward"] == 0.0


def test_entire_database_hash_detects_unrelated_changes(
    backend: RetailBackend, seed_snapshot: dict, cancel_task: Task
) -> None:
    """目标订单正确但误改了其他用户时，完整数据库比较仍判失败。"""
    predicted = cancel_order(backend)
    predicted["users"]["u2"]["address"]["city"] = "北京"
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        predicted,
        [],
        termination_reason=TerminationReason.USER_STOP,
    )
    assert not result["db_match"] and not result["task_success"]


def test_predicted_trajectory_is_never_replayed(
    backend: RetailBackend, seed_snapshot: dict, cancel_task: Task
) -> None:
    """即使输入含额外取消工具调用，预测侧也只读取本体快照。"""
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        cancel_order(backend),
        [
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="unexpected",
                        name="cancel_pending_order",
                        arguments={"order_id": "#W0000004", "reason": "no longer needed"},
                    )
                ],
            )
        ],
        termination_reason=TerminationReason.USER_STOP,
    )
    assert result["db_match"] and result["task_success"]


def test_reference_errors_continue_with_official_endstate_semantics(
    backend: RetailBackend, seed_snapshot: dict, cancel_task: Task
) -> None:
    """参考查询报错后仍执行取消，结果与官方环境判分器一致。"""
    cancel_task.evaluation_criteria.actions.insert(
        0,
        Action(action_id="bad-query", name="get_order_details", arguments={"order_id": "#missing"}),
    )
    reference = get_environment(db=RetailDB.model_validate(seed_snapshot), policy="")
    trajectory = []
    for action in cancel_task.evaluation_criteria.actions:
        call = ToolCall(
            id=action.action_id, name=action.name, arguments=action.arguments, requestor="assistant"
        )
        trajectory.extend(
            [AssistantMessage(role="assistant", tool_calls=[call]), reference.get_response(call)]
        )
    official = EnvironmentEvaluator.calculate_reward(
        get_environment,
        cancel_task,
        trajectory,
        env_kwargs={"db": RetailDB.model_validate(seed_snapshot), "policy": ""},
    )
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        cancel_order(backend),
        [],
        termination_reason=TerminationReason.USER_STOP,
    )
    assert result["reward_info"]["db_check"] == official.db_check.model_dump()
    assert result["reward_info"]["reward_breakdown"]["DB"] == official.reward
    assert result["reference_errors"][0]["action_id"] == "bad-query"
    assert len(result["reference_errors"]) == 1


@pytest.mark.parametrize("met", [True, False])
def test_nl_assertions_gate_success_without_changing_database_metric(
    monkeypatch: pytest.MonkeyPatch,
    backend: RetailBackend,
    seed_snapshot: dict,
    cancel_task: Task,
    met: bool,
) -> None:
    """自然语言断言单独决定完整成功，真实数据库一致指标保持独立。"""
    assertion = "The agent explained the refund."
    cancel_task.evaluation_criteria.nl_assertions = [assertion]
    calls = []

    def generate(**kwargs: object) -> AssistantMessage:
        """记录实际模型请求，并提供指定断言的结构化模型输出。"""
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": assertion,
                            "metExpectation": met,
                            "reasoning": "测试依据",
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(evaluation, "generate", generate)
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        cancel_order(backend),
        [AssistantMessage(role="assistant", content="已取消，退款返回原支付方式。")],
        termination_reason=TerminationReason.USER_STOP,
        nl_model="openai/test-judge",
        nl_model_args={"temperature": 0, "timeout": 3},
    )
    assert result["db_match"]
    assert result["task_success"] is met
    assert result["reward_info"]["nl_assertions"] == [
        {"nl_assertion": assertion, "met": met, "justification": "测试依据"}
    ]
    assert calls[0]["model"] == "openai/test-judge"
    assert calls[0]["temperature"] == 0 and calls[0]["timeout"] == 3
    assert calls[0]["call_name"] == "nl_assertions_eval"
    assert "reasoning_effort" not in calls[0]


def test_nl_prompt_and_aggregation_match_pinned_official_evaluator(
    monkeypatch: pytest.MonkeyPatch, seed_snapshot: dict, cancel_task: Task
) -> None:
    """更换显式模型不会改写官方提示词、结果 schema 或断言聚合。"""
    cancel_task.evaluation_criteria.nl_assertions = ["The agent explains the policy."]
    cancel_task.evaluation_criteria.actions = []
    messages = [
        UserMessage(role="user", content="你好"),
        AssistantMessage(role="assistant", content="政策说明。"),
    ]
    calls = []

    def generate(**kwargs: object) -> AssistantMessage:
        """记录官方与适配层的提示词，返回同一条断言判断。"""
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant",
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": cancel_task.evaluation_criteria.nl_assertions[0],
                            "metExpectation": True,
                            "reasoning": "已说明",
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr(evaluation, "generate", generate)
    monkeypatch.setattr(evaluator_nl_assertions, "generate", generate)
    official = evaluator_nl_assertions.NLAssertionsEvaluator.calculate_reward(cancel_task, messages)
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        seed_snapshot,
        messages,
        termination_reason=TerminationReason.USER_STOP,
        nl_model="openai/test-judge",
    )
    assert [message.content for message in calls[0]["messages"]] == [
        message.content for message in calls[1]["messages"]
    ]
    assert result["reward_info"]["nl_assertions"] == [
        item.model_dump() for item in official.nl_assertions
    ]
    assert result["reward_info"]["reward_breakdown"]["NL_ASSERTION"] == official.reward


def test_nl_assertions_require_explicit_model(seed_snapshot: dict, cancel_task: Task) -> None:
    """配置缺失时明确报错，不使用上游默认的其他模型服务。"""
    cancel_task.evaluation_criteria.nl_assertions = ["The agent explained the refund."]
    with pytest.raises(ValueError, match="显式配置判分模型"):
        evaluate_task(
            cancel_task,
            seed_snapshot,
            seed_snapshot,
            [],
            termination_reason=TerminationReason.USER_STOP,
        )


@pytest.mark.parametrize("reply", [{}, {"results": []}])
def test_empty_nl_results_preserve_pinned_official_aggregation(
    monkeypatch: pytest.MonkeyPatch, seed_snapshot: dict, cancel_task: Task, reply: dict
) -> None:
    """记录上游缺项限制：空断言结果沿用官方 all() 聚合，仍会得到通过。"""
    cancel_task.evaluation_criteria.actions = []
    cancel_task.evaluation_criteria.nl_assertions = ["The agent explained the refund."]

    def generate(**kwargs: object) -> AssistantMessage:
        """只替换模型原始响应，让官方聚合处理空数组和缺少字段。"""
        return AssistantMessage(role="assistant", content=json.dumps(reply))

    monkeypatch.setattr(evaluation, "generate", generate)
    monkeypatch.setattr(evaluator_nl_assertions, "generate", generate)
    official = evaluator_nl_assertions.NLAssertionsEvaluator.calculate_reward(cancel_task, [])
    result = evaluate_task(
        cancel_task,
        seed_snapshot,
        seed_snapshot,
        [],
        termination_reason=TerminationReason.USER_STOP,
        nl_model="openai/test-judge",
    )
    assert result["reward_info"]["nl_assertions"] == []
    assert result["reward_info"]["reward"] == official.reward == 1.0


@pytest.mark.parametrize("failure", [TimeoutError("模拟超时"), ValueError("模拟模型响应失败")])
def test_model_errors_remain_incomplete_instead_of_agent_failures(
    monkeypatch: pytest.MonkeyPatch, seed_snapshot: dict, cancel_task: Task, failure: Exception
) -> None:
    """判分模型异常原样上抛，避免把未完成判分记作 Agent 零分。"""
    cancel_task.evaluation_criteria.nl_assertions = ["The agent explained the refund."]

    def generate(**kwargs: object) -> AssistantMessage:
        """在模型边界模拟失败，不替换真实环境比较。"""
        raise failure

    monkeypatch.setattr(evaluation, "generate", generate)
    with pytest.raises(type(failure), match=str(failure)):
        evaluate_task(
            cancel_task,
            seed_snapshot,
            seed_snapshot,
            [],
            termination_reason=TerminationReason.USER_STOP,
            nl_model="openai/test-judge",
        )


@pytest.mark.parametrize("reason", sorted(evaluation.INCOMPLETE_REASONS))
def test_unfinished_runtime_is_not_graded_as_agent_failure(
    seed_snapshot: dict, cancel_task: Task, reason: TerminationReason
) -> None:
    """基础设施或执行异常不落为已完成样本的业务零分。"""
    with pytest.raises(EvaluationIncompleteError, match=reason.value):
        evaluate_task(cancel_task, seed_snapshot, seed_snapshot, [], termination_reason=reason)


@pytest.mark.parametrize("reason", [TerminationReason.MAX_STEPS, TerminationReason.TOO_MANY_ERRORS])
def test_guard_limit_is_failure_even_with_matching_database(
    backend: RetailBackend, seed_snapshot: dict, cancel_task: Task, reason: TerminationReason
) -> None:
    """达到对话护栏上限按官方规则失败，但仍记录独立数据库一致指标。"""
    cancel_task.evaluation_criteria.nl_assertions = ["The agent explained the refund."]
    result = evaluate_task(
        cancel_task, seed_snapshot, cancel_order(backend), [], termination_reason=reason
    )
    assert result["db_match"] and not result["task_success"]
    assert result["reward_info"]["reward"] == 0
    assert result["reward_info"]["nl_assertions"] is None


def test_nonempty_initial_state_is_rejected_explicitly(
    seed_snapshot: dict, cancel_task: Task
) -> None:
    """尚未接入的初始化状态不能被忽略后继续运行或判分。"""
    cancel_task.initial_state = InitialState()
    with pytest.raises(ValueError, match="initial_state"):
        evaluate_task(
            cancel_task,
            seed_snapshot,
            seed_snapshot,
            [],
            termination_reason=TerminationReason.USER_STOP,
        )


@pytest.mark.parametrize("basis", ["ACTION", "COMMUNICATE"])
def test_unsupported_reward_basis_is_not_silently_dropped(
    seed_snapshot: dict, cancel_task: Task, basis: str
) -> None:
    """当前中文任务集之外的判分依据明确报错，避免漏算后发布成功。"""
    cancel_task.evaluation_criteria = EvaluationCriteria(actions=[], reward_basis=[basis])
    with pytest.raises(ValueError, match="reward_basis"):
        evaluate_task(
            cancel_task,
            seed_snapshot,
            seed_snapshot,
            [],
            termination_reason=TerminationReason.USER_STOP,
        )
