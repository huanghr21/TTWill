"""用真实官方调度器、持久化运行时和离线 HTTP 验证单题评测入口。"""

import json
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后验证官方评测入口")

import litellm  # noqa: E402
from openai import OpenAI  # noqa: E402
from tau2.utils import llm_utils  # noqa: E402

from ttwill.bench import runner  # noqa: E402
from ttwill.bench.evaluation import EvaluationIncompleteError  # noqa: E402
from ttwill.cli import main  # noqa: E402
from ttwill.config import ConfigError, ModelConfig, load_config  # noqa: E402
from ttwill.core.context import STATE_NOTICE  # noqa: E402
from ttwill.llm import FakeModel, LLMError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "68_zh_identity"
SECRET = "runner-test-credential-must-stay-in-memory"


def tool_response(name: str, args: dict) -> dict:
    """构造交给真实协调者或子 Agent 校验的模型工具提议。

    Args:
        name: 本体已有的工具名称。
        args: 模型提出的参数，不含代码维护的任务字段。

    Returns:
        含一次工具调用和固定用量的 Chat Completions 响应。
    """
    return {
        "model": "offline-agent",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": uuid4().hex,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


def official_response(model: str, content: str) -> dict:
    """返回官方 SDK 可读取的完整文本响应，保留明确用量。"""
    return {
        "id": uuid4().hex,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


@pytest.fixture
def offline_runner(tmp_path, monkeypatch):
    """让本体使用假模型，官方模拟与判分仍经真实 LiteLLM 和 SDK 请求。

    Args:
        tmp_path: 本测试独立产物目录。
        monkeypatch: 结束时恢复配置、模型构造与传输入口。

    Yields:
        含本体请求、官方 HTTP 请求及可选择失败阶段的测试控制字典。
    """
    base = load_config(ROOT / "config.toml")
    model = ModelConfig(
        model="offline-agent",
        base_url="https://unit.invalid/v1",
        api_key_env="TEST_KEY",
        input_cost_per_million=1,
        output_cost_per_million=2,
    )
    config = replace(
        base,
        runs_dir=tmp_path,
        llm=replace(base.llm, max_retries=0, retry_delay_seconds=0),
        models={
            "flash": model,
            "user_simulator": replace(model, model="offline-user"),
            "evaluator": replace(model, model="offline-judge"),
        },
    )
    control = {"fail_role": None, "official_requests": [], "user_calls": 0}

    def script():
        """按真实查询流程验身份、建任务、读取订单并返回事实。"""
        yield tool_response(
            "find_user_id_by_name_zip",
            {
                "first_name": "浩然",
                "last_name": "宋",
                "zip": "116021",
            },
        )
        if control.get("agent_error"):
            yield LLMError("模型传输失败", retryable=True)
            return
        yield tool_response(
            "update_tasks",
            {
                "changes": [
                    {
                        "operation": "add",
                        "capability": "order_service",
                        "action": "get_order_details",
                        "order_id": "#W6729841",
                        "instruction": "查询订单总金额。",
                    }
                ]
            },
        )
        yield tool_response("complete_plan", {})
        snapshot = next(
            json.loads(message["content"].removeprefix(STATE_NOTICE))
            for message in fake.requests[-1]["messages"]
            if message["role"] == "system" and message["content"].startswith(STATE_NOTICE)
        )
        yield tool_response("dispatch_tasks", {"task_ids": [snapshot["tasks"][0]["id"]]})
        yield tool_response("get_order_details", {"order_id": "#W6729841"})
        yield tool_response("finish", {"missing": [], "note": "订单总额为 ¥5,806.01。"})
        yield tool_response("reply", {"message": "请查看订单金额。"})

    fake = FakeModel(script())
    control["agent"] = fake

    @asynccontextmanager
    async def fake_client(env_path):
        """替换模型传输生命周期，保持真实运行时和全部 SQLite 组件。"""
        assert env_path == ROOT / ".env"
        yield fake

    def model_args(app_config, selected):
        """以虚拟密钥替代本地 .env 解析，不接触用户真实凭据。"""
        return {"api_key": SECRET, "api_base": selected.base_url, "timeout": 5}

    def forbid_dotenv(*args, **kwargs):
        """一旦入口误读真实 .env，立即使测试失败。"""
        raise AssertionError("离线端到端测试不能读取 .env")

    def transport(request: httpx.Request) -> httpx.Response:
        """按角色产生用户原话与判分结果，指定阶段可返回真实 HTTP 限流。"""
        body = json.loads(request.content)
        control["official_requests"].append(deepcopy(body))
        is_user = body["model"] == "offline-user"
        if is_user:
            control["user_calls"] += 1
        if control["fail_role"] == ("user_simulator" if is_user else "evaluator") and (
            not is_user or control["user_calls"] == 2
        ):
            return httpx.Response(429, json={"error": {"message": "限流", "type": "rate_limit"}})
        if is_user:
            content = (
                "我是宋浩然，邮编 116021，想查最近订单花了多少钱。"
                if control["user_calls"] == 1
                else "谢谢，已经清楚了。###STOP###"
            )
        else:
            content = json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": (
                                "Agent should tell the user the order total is ¥5,806.01."
                            ),
                            "metExpectation": True,
                            "reasoning": "对话给出了实际订单与付款信息。",
                        }
                    ]
                }
            )
        return httpx.Response(200, json=official_response(body["model"], content))

    client = OpenAI(
        api_key=SECRET,
        base_url="https://unit.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )

    def completion(**kwargs):
        """只在官方调用边界注入假 HTTP 客户端，保留 LiteLLM 回调全流程。"""
        return litellm.completion(**kwargs, client=client)

    monkeypatch.setattr(runner, "load_config", lambda path: config)
    monkeypatch.setattr(runner, "_model_args", model_args)
    monkeypatch.setattr(runner, "dotenv_values", forbid_dotenv)
    monkeypatch.setattr(runner, "CompatibleModel", fake_client)
    monkeypatch.setattr(llm_utils, "completion", completion)
    try:
        yield control
    finally:
        client.close()


def read_json(path: Path) -> dict:
    """读取本次运行的一份 JSON 产物。"""
    return json.loads(path.read_text())


def assert_artifacts(output: Path, *, completed: bool) -> tuple[dict, list[dict]]:
    """核对独立数据库、完整关联和原文引用，禁止凭据落入任何运行文件。

    Args:
        output: 本次尝试的独立目录。
        completed: 是否预期已完成全部判分。

    Returns:
        本次报告与全部调用账本条目。
    """
    metadata = read_json(output / "metadata.json")
    report = read_json(output / "report.json")
    assert report["completed"] is completed
    for name in ("business.sqlite", "checkpoints.sqlite", "archive.sqlite"):
        assert (output / name).stat().st_size > 0
    assert read_json(output / "initial_database.json") == read_json(output / "final_database.json")
    events = [
        json.loads(line)
        for path in (output / "traces").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    usage = [json.loads(line) for line in (output / "usage.jsonl").read_text().splitlines()]
    for record in events + usage:
        for key in ("run_id", "task_id", "attempt_id", "repeat"):
            assert record[key] == metadata[key]
    for entry in usage:
        assert (output / entry["record_ref"]).is_file()
    for path in output.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes()
    return report, usage


def test_official_runner_scores_real_runtime_and_keeps_hidden_task_out_of_agent(
    tmp_path, offline_runner
) -> None:
    """官方用户、调度器与判分完整运行，本体只接收用户文本及自身工具结果。"""
    output = tmp_path / "normal"
    report = runner.run_single(ROOT / "config.toml", TASK_ID, output, seed=4, repeat=3)
    saved, usage = assert_artifacts(output, completed=True)
    assert report == saved
    assert report["task_success"] is True and report["db_match"] is True
    assert report["termination_reason"] == "user_stop"
    assert len(offline_runner["agent"].requests) == 7
    assert len(offline_runner["official_requests"]) == 3
    assert {entry.get("role") for entry in usage} == {None, "user_simulator", "evaluator"}
    messages = read_json(output / "messages.json")
    assert all(message["role"] in {"assistant", "user"} for message in messages)
    assert all(not message.get("tool_calls") for message in messages)
    assert all("update_tasks" not in message["content"] for message in messages)
    raw_task = read_json(output / "task.json")
    model_inputs = json.dumps(offline_runner["agent"].requests, ensure_ascii=False)
    assert raw_task["user_scenario"]["instructions"]["reason_for_call"] not in model_inputs
    assert raw_task["evaluation_criteria"]["nl_assertions"][0] not in model_inputs
    assert TASK_ID not in model_inputs
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run_single(ROOT / "config.toml", TASK_ID, output)
    assert (output / "report.json").read_bytes() == before


@pytest.mark.parametrize("role", ["user_simulator", "evaluator"])
def test_transport_failure_preserves_partial_run_without_fabricated_score(
    tmp_path, offline_runner, role
) -> None:
    """模拟或判分阶段限流保留已发生调用和业务终态，不伪造成功或零分结果。"""
    offline_runner["fail_role"] = role
    output = tmp_path / role
    with pytest.raises(litellm.RateLimitError):
        runner.run_single(ROOT / "config.toml", TASK_ID, output)
    report, usage = assert_artifacts(output, completed=False)
    assert report["error_type"] == "RateLimitError"
    assert "task_success" not in report and "db_match" not in report
    failed = [entry for entry in usage if entry["status"] == "error"]
    assert len(failed) == 1 and failed[0]["role"] == role
    assert failed[0]["cost_usd"] is None
    assert len(offline_runner["official_requests"]) == (2 if role == "user_simulator" else 3)
    assert len(read_json(output / "messages.json")) >= 3


def test_agent_transport_failure_remains_unfinished_after_visible_failure_reply(
    tmp_path, offline_runner
) -> None:
    """本体将网络失败转成用户回复后，评测仍须识别未完成并跳过成功判分。"""
    offline_runner["agent_error"] = True
    output = tmp_path / "agent_failure"
    with pytest.raises(EvaluationIncompleteError):
        runner.run_single(ROOT / "config.toml", TASK_ID, output)
    report, usage = assert_artifacts(output, completed=False)
    assert report["error_type"] == "EvaluationIncompleteError"
    assert "task_success" not in report and "db_match" not in report
    assert len(offline_runner["official_requests"]) == 2
    failed = [entry for entry in usage if entry["status"] == "error"]
    assert len(failed) == 1 and failed[0]["tier"] == "flash"
    assert failed[0]["cost_usd"] is None
    assert read_json(output / "simulation.json")["termination_reason"] == "user_stop"
    assert len(read_json(output / "messages.json")) == 4


def test_missing_model_roles_fail_before_reading_credentials_or_creating_run(
    tmp_path, monkeypatch
) -> None:
    """默认假模型配置缺少评测档位时明确报错，不能开始读密钥或创建运行。"""

    def reject_credentials(*args, **kwargs):
        """缺档位的配置不应进入凭据解析。"""
        raise AssertionError("不应解析凭据")

    monkeypatch.setattr(runner, "dotenv_values", reject_credentials)
    output = tmp_path / "missing_config"
    with pytest.raises(ConfigError, match="评测缺少模型档位.*evaluator.*user_simulator"):
        runner.run_single(ROOT / "config.toml", TASK_ID, output)
    assert not output.exists()


def test_official_model_args_keep_credentials_local_and_reject_callback_overrides(
    monkeypatch,
) -> None:
    """密钥只取无插值的指定 .env 结果，模型参数不能覆盖回调与重试约定。"""
    config = load_config(ROOT / "config.toml")
    model = ModelConfig(
        model="offline-user",
        base_url="https://unit.invalid/v1",
        api_key_env="TEST_SECRET",
        parameters={"temperature": 0.2},
    )

    def local_values(path, *, interpolate):
        """模拟指定文件的密钥读取，检查未启用环境变量插值。"""
        assert path == ROOT / ".env" and interpolate is False
        return {"TEST_SECRET": SECRET}

    monkeypatch.setattr(runner, "dotenv_values", local_values)
    monkeypatch.setenv("TEST_SECRET", "不能采用进程环境里的密钥")
    args = runner._model_args(config, model)
    assert args["api_key"] == SECRET and args["temperature"] == 0.2
    assert args["caching"] is False
    with pytest.raises(ConfigError, match="不能覆盖") as caught:
        runner._model_args(config, replace(model, parameters={"success_callback": [SECRET]}))
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize(
    "values", [{"seed": True}, {"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")}]
)
def test_invalid_public_run_limits_fail_before_loading_resources(
    tmp_path, monkeypatch, values
) -> None:
    """错误种子类型和非有限超时必须在读取配置、凭据与资源之前拒绝。"""

    def reject_config(path):
        """非法运行参数不能继续加载项目配置。"""
        raise AssertionError("不应加载配置")

    monkeypatch.setattr(runner, "load_config", reject_config)
    with pytest.raises(ValueError):
        runner.run_single(tmp_path / "unused.toml", TASK_ID, tmp_path / "invalid", **values)
    assert not (tmp_path / "invalid").exists()


def test_evaluate_cli_passes_explicit_options_and_prints_report(
    tmp_path, monkeypatch, capsys
) -> None:
    """CLI 把模型档位和运行参数交给单题入口，并输出实际完成报告。"""
    calls = []
    output = tmp_path / "cli"

    def run_single(config_path, task_id, destination, **kwargs):
        """记录 CLI 实参，省略已被端到端用例覆盖的数据库执行。"""
        calls.append((config_path, task_id, destination, kwargs))
        return {"completed": True, "task_success": True, "db_match": True}

    monkeypatch.setattr(runner, "run_single", run_single)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ttwill",
            "evaluate",
            "--config",
            str(ROOT / "config.toml"),
            "--task-id",
            TASK_ID,
            "--output",
            str(output),
            "--user-tier",
            "caller",
            "--judge-tier",
            "judge",
            "--seed",
            "7",
            "--repeat",
            "2",
            "--max-steps",
            "20",
            "--timeout-seconds",
            "90",
        ],
    )
    main()
    assert calls == [
        (
            ROOT / "config.toml",
            TASK_ID,
            output,
            {
                "user_tier": "caller",
                "judge_tier": "judge",
                "seed": 7,
                "repeat": 2,
                "max_steps": 20,
                "timeout_seconds": 90.0,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["completed"] is True


@pytest.mark.parametrize(
    "error",
    [
        EvaluationIncompleteError(SECRET),
        LLMError(SECRET),
        litellm.RateLimitError(SECRET, llm_provider="openai", model="offline"),
    ],
)
def test_evaluate_cli_hides_sdk_error_text_and_points_to_partial_report(
    tmp_path, monkeypatch, capsys, error
) -> None:
    """CLI 只显示已捕获评测错误的类型和报告位置，不能打印携密钥的错误原文。"""
    output = tmp_path / "cli_failure"

    def run_single(config_path, task_id, destination, **kwargs):
        """模拟单题入口先保存未完成报告，再原样抛出传输或判分错误。"""
        destination.mkdir()
        (destination / "report.json").write_text(json.dumps({"completed": False}))
        raise error

    monkeypatch.setattr(runner, "run_single", run_single)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ttwill",
            "evaluate",
            "--task-id",
            TASK_ID,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    printed = capsys.readouterr()
    assert SECRET not in printed.out + printed.err
    assert type(error).__name__ in printed.err
    assert str(output / "report.json") in printed.err
