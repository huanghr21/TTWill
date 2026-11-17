"""以业务终态接入固定 τ² 版本的零售判分，保留官方自然语言断言语义。"""

import json
from typing import Any

from tau2.data_model.message import Message, SystemMessage, ToolCall, UserMessage
from tau2.data_model.simulation import (
    DBCheck,
    EnvAssertionCheck,
    NLAssertionCheck,
    RewardInfo,
    TerminationReason,
)
from tau2.data_model.tasks import RewardType, Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment
from tau2.environment.toolkit import get_tool_types
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.utils.llm_utils import generate

# 提示词与聚合来源：固定提交 c06b07d 的 evaluator_nl_assertions.py。
# 上游未提供模型参数，局部子类只替换模型调用参数，不修改其全局常量。
NL_SYSTEM_PROMPT = (
    "\n"
    "        TASK\n"
    "        - You will be given a list of expected outcomes and a conversation that was "
    "collected during a test case run.\n"
    "        - The conversation is between an agent and a customer.\n"
    "        - Your job is to evaluate whether the agent satisfies each of the expected "
    "outcomes.\n"
    "        - Grade each expected outcome individually.\n"
    "\n"
    "        FORMAT\n"
    "        - Your response should be a JSON object with the following fields:\n"
    "        - `reasoning`: a short explanation for your classification\n"
    "        - `metExpectation`: `true` if the agent satisfies the expected outcomes, "
    "`false` otherwise\n"
    "        - `expectedOutcome`: repeat the expectation from the input that you are "
    "grading\n"
    "        \n"
    "        Example response structure:\n"
    "        {\n"
    '            "results": [\n'
    "                {\n"
    '                    "expectedOutcome": "<one of the expected outcomes from the '
    'input>",\n'
    '                    "reasoning": "<reasoning trace>",\n'
    '                    "metExpectation": <false or true>,\n'
    "                }\n"
    "            ]\n"
    "        }\n"
    "        "
)

INCOMPLETE_REASONS = {
    TerminationReason.TIMEOUT,
    TerminationReason.INFRASTRUCTURE_ERROR,
    TerminationReason.AGENT_ERROR,
    TerminationReason.USER_ERROR,
    TerminationReason.UNEXPECTED_ERROR,
}
SUPPORTED_BASES = {RewardType.DB, RewardType.ENV_ASSERTION, RewardType.NL_ASSERTION}


class EvaluationIncompleteError(RuntimeError):
    """表示运行未完成，需要保留尝试并补跑，不能计作 Agent 业务失败。"""


def _configured_nl_evaluator(
    model: str | None, model_args: dict[str, Any]
) -> type[NLAssertionsEvaluator]:
    """为单次判分绑定模型配置，避免不同样本共享可变类配置。

    Args:
        model: 显式配置的 LiteLLM 模型标识；没有自然语言断言时允许为 None。
        model_args: 传给官方 generate 的模型参数。

    Returns:
        继承官方 calculate_reward 聚合规则的局部判分器类。
    """

    class ConfiguredNLAssertionsEvaluator(NLAssertionsEvaluator):
        """保留官方提示词与结果结构，仅为本次判分绑定模型调用参数。"""

        @classmethod
        def evaluate_nl_assertions(
            cls, trajectory: list[Message], nl_assertions: list[str]
        ) -> list[NLAssertionCheck]:
            """用显式配置的模型判断对话断言。

            Args:
                trajectory: 供判分读取的原始对话文本消息。
                nl_assertions: 任务中需要逐条判断的自然语言断言。

            Returns:
                官方 NLAssertionCheck 列表，保留模型给出的结论与依据。
                兼容固定上游的缺项行为：缺少 results 视为空列表，不校验断言是否
                全部返回；继承的 all() 聚合会将空列表记为通过。

            Raises:
                ValueError: 没有配置模型，或模型输出不符合官方 JSON 结构。
            """
            if not model:
                raise ValueError("本题包含自然语言断言，必须显式配置判分模型")
            trajectory_str = "\n".join(
                f"{message.role}: {message.content}" for message in trajectory
            )
            user_prompt = (
                f"\n        conversation:\n        {trajectory_str}\n"
                "        \n        expectedOutcomes:\n"
                f"        {nl_assertions}\n        "
            )
            response = generate(
                model=model,
                messages=[
                    SystemMessage(role="system", content=NL_SYSTEM_PROMPT),
                    UserMessage(role="user", content=user_prompt),
                ],
                call_name="nl_assertions_eval",
                **model_args,
            )
            data = json.loads(response.content)
            return [
                NLAssertionCheck(
                    nl_assertion=result["expectedOutcome"],
                    met=result["metExpectation"],
                    justification=result["reasoning"],
                )
                for result in data.get("results", [])
            ]

    return ConfiguredNLAssertionsEvaluator


def evaluate_task(
    task: Task,
    original_database: dict[str, Any],
    predicted_snapshot: dict[str, Any],
    messages: list[Message],
    *,
    termination_reason: TerminationReason,
    nl_model: str | None = None,
    nl_model_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较本体终态与官方参考终态，并按任务依据聚合完整任务判分。

    参考动作使用官方环境真实执行；预测环境直接加载本体快照，不重放任何对话
    工具。环境比较与 reward_basis 乘法规则取自固定版本 evaluator_env.py 和
    evaluator.py，自然语言判分继承官方聚合。只接入当前 retail_50 使用的格式。

    Args:
        task: 官方任务模型，initial_state 必须为 None。
        original_database: 本次运行开始前的原始业务数据库，使用 τ² 零售格式。
        predicted_snapshot: 本体运行结束后导出的完整业务数据库快照。
        messages: 官方类型的实际对话；仅用于自然语言与动作诊断，不重放到业务库。
        termination_reason: 对话结束原因；正常结束才能获得完整任务成功。
        nl_model: 显式配置的判分模型标识，无适用断言时允许为 None。
        nl_model_args: 传给官方 generate 的参数，不隐式沿用官方默认服务或参数。

    Returns:
        包含 task_id、db_match、reference_hash、predicted_hash、task_success、
        termination_reason、reference_errors 和官方 RewardInfo JSON 的报告。
        db_match 独立记录终态一致；task_success 同时要求适用断言通过和正常结束。

    Raises:
        EvaluationIncompleteError: 运行因超时、基础设施或执行异常而未完成。
        ValueError: 初始状态或判分依据不受支持，数据库格式错误，或缺少判分模型。
        Exception: 官方模型调用异常原样上抛，调用方应保留为未完成的判分尝试。
    """
    if termination_reason in INCOMPLETE_REASONS:
        raise EvaluationIncompleteError(f"运行未完成判分：{termination_reason.value}")
    if task.initial_state is not None:
        raise ValueError("当前 retail_50 判分入口只支持 initial_state 为 None 的任务")
    criteria = task.evaluation_criteria
    if criteria is None:
        raise ValueError("当前 retail_50 判分入口要求任务提供 evaluation_criteria")
    unsupported = set(criteria.reward_basis) - SUPPORTED_BASES
    if unsupported:
        raise ValueError(f"当前判分入口不支持这些 reward_basis：{sorted(unsupported)}")

    predicted = get_environment(db=RetailDB.model_validate(predicted_snapshot), policy="")
    reference = get_environment(db=RetailDB.model_validate(original_database), policy="")
    reference_errors = []
    for action in criteria.actions or []:
        # 官方 get_response 保留参考动作报错后继续的行为；retail.sync_tools 不修改数据。
        response = reference.get_response(
            ToolCall(
                id=action.action_id,
                name=action.name,
                arguments=action.arguments,
                requestor=action.requestor,
            )
        )
        if response.error:
            reference_errors.append(
                {"action_id": action.action_id, "name": action.name, "error": response.content}
            )

    reference_hash, predicted_hash = reference.get_db_hash(), predicted.get_db_hash()
    db_match = (
        reference_hash == predicted_hash
        and reference.get_user_db_hash() == predicted.get_user_db_hash()
    )
    checks = [
        EnvAssertionCheck(
            env_assertion=assertion,
            met=(met := predicted.run_env_assertion(assertion, raise_assertion_error=False)),
            reward=float(met),
        )
        for assertion in criteria.env_assertions or []
    ]
    breakdown = {}
    if RewardType.DB in criteria.reward_basis:
        breakdown[RewardType.DB] = float(db_match)
    if RewardType.ENV_ASSERTION in criteria.reward_basis:
        breakdown[RewardType.ENV_ASSERTION] = float(all(check.met for check in checks))

    normal_stop = termination_reason in {TerminationReason.USER_STOP, TerminationReason.AGENT_STOP}
    nl_info = None
    if normal_stop and RewardType.NL_ASSERTION in criteria.reward_basis:
        evaluator = _configured_nl_evaluator(nl_model, nl_model_args or {})
        nl_info = evaluator.calculate_reward(task, messages)
        breakdown.update(nl_info.reward_breakdown or {})
    action_info = ActionEvaluator.calculate_reward(task, messages, get_tool_types(reference.tools))
    reward = float(normal_stop)
    for component in breakdown.values():
        reward *= component
    reward_info = RewardInfo(
        reward=reward,
        db_check=DBCheck(db_match=db_match, db_reward=float(db_match)),
        env_assertions=checks,
        action_checks=action_info.action_checks,
        nl_assertions=nl_info.nl_assertions if nl_info is not None else None,
        reward_basis=criteria.reward_basis,
        reward_breakdown=breakdown,
        info={
            "env": None,
            "nl": nl_info.info if nl_info is not None else None,
            "action": action_info.info,
            "note": None if normal_stop else "对话提前结束，按官方规则判为任务失败",
        },
    )
    return {
        "task_id": task.id,
        "db_match": db_match,
        "reference_hash": reference_hash,
        "predicted_hash": predicted_hash,
        "task_success": reward == 1.0,
        "termination_reason": termination_reason.value,
        "reference_errors": reference_errors,
        "reward_info": reward_info.model_dump(mode="json"),
    }
