"""用假单题运行验证固定开发题单、隔离目录和未完成样本的汇总口径。"""

import ast
import json
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后验证批量评测入口")

import litellm  # noqa: E402
from openai import APIError, OpenAI  # noqa: E402
from tau2.utils import llm_utils  # noqa: E402

from ttwill.bench import batch, runner, tasks  # noqa: E402
from ttwill.bench.evaluation import EvaluationIncompleteError  # noqa: E402
from ttwill.bench.retail import RetailBundle, load_retail_bundle  # noqa: E402
from ttwill.cli import main  # noqa: E402
from ttwill.config import ModelConfig, load_config  # noqa: E402
from ttwill.llm import FakeModel, LLMError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SECRET = "batch-test-error-text-must-not-be-saved"


def read_json(path: Path) -> dict:
    """读取一份测试运行产物。"""
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def offline_batch(monkeypatch):
    """替换任务来源及单题执行，保存真实独立目录与逐题报告。

    Args:
        monkeypatch: 测试结束后恢复模块中的资源与运行入口。

    Returns:
        可指定逐题结果或异常、查看调用参数的控制字典。
    """
    base = load_config(ROOT / "config.toml")
    config = replace(
        base,
        models={
            "flash": base.models["flash"],
            "user_simulator": base.models["flash"],
            "evaluator": base.models["flash"],
        },
    )
    bundle = RetailBundle(
        database={},
        tasks=[{"id": task_id} for task_id in ("c", "a", "d", "b")],
        policy="",
        metadata={},
        settings={},
        manifest={"commit": "fixed", "files": {"dev_64.json": {"sha256": "source-hash"}}},
        language_pack="",
    )
    success = {"completed": True, "task_success": True, "db_match": True, "reward": 1.0}
    control = {
        "calls": [],
        "loads": [],
        "outcomes": [success.copy() for _ in bundle.tasks],
        "bundle": bundle,
    }

    def load_bundle(source_dir, task_dir, dataset):
        """记录固定开发清单的加载参数。"""
        control["loads"].append((source_dir, task_dir, dataset))
        return control["bundle"]

    def fake_single(config_path, task_id, output, **kwargs):
        """每题创建独立状态并按脚本返回结果或抛出异常，不调用模型。"""
        index = len(control["calls"])
        control["calls"].append((config_path, task_id, output, kwargs))
        output.mkdir(parents=True, exist_ok=False)
        (output / "business.sqlite").write_text(task_id, encoding="utf-8")
        report = {
            "run_id": f"run-{index}",
            "attempt_id": f"attempt-{index}",
            "completed": False,
        }
        outcome = control["outcomes"][index]
        if isinstance(outcome, BaseException):
            report["error_type"] = type(outcome).__name__
        else:
            report.update(outcome)
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        if isinstance(outcome, BaseException):
            raise outcome
        return report

    def provenance(app_config, loaded_bundle):
        """提供明确的代码摘要，避开真实 Git 与官方安装查询。"""
        assert app_config is config and loaded_bundle is control["bundle"]
        return {"git_head": "test-commit", "sha256": {"source.py": "code-hash"}}

    monkeypatch.setattr(batch, "load_config", lambda path: config)
    monkeypatch.setattr(batch, "load_task_bundle", load_bundle)
    monkeypatch.setattr(batch, "_provenance", provenance)
    monkeypatch.setattr(batch, "run_single", fake_single)
    return control


def test_batch_keeps_fixed_order_independent_attempts_and_completed_denominators(
    tmp_path, offline_batch
) -> None:
    """业务失败计入完成数，服务失败继续下一题，种子、清单与独立目录完整保存。"""
    offline_batch["outcomes"][1] = {
        "completed": True,
        "task_success": False,
        "db_match": True,
        "reward": 0,
    }
    offline_batch["outcomes"][2] = LLMError(SECRET)
    output = tmp_path / "batch"
    summary = batch.run_batch(ROOT / "config.toml", output, seed=13, max_steps=40)
    assert summary == read_json(output / "summary.json")
    assert summary["status"] == "finished"
    assert summary["planned"] == 4 and summary["completed"] == 3
    assert summary["attempted"] == 4 and summary["pending"] == 0
    assert summary["completion_rate"] == 0.75
    assert summary["task_success_rate"] == 2 / 3
    assert summary["db_match_rate"] == 1.0
    assert summary["infrastructure_incomplete"] == 1
    assert [item["status"] for item in summary["results"]] == [
        "business_success",
        "business_failure",
        "infrastructure_incomplete",
        "business_success",
    ]
    failed = summary["results"][2]
    assert failed["error_type"] == "LLMError"
    assert "task_success" not in failed and "db_match" not in failed
    plan = read_json(output / "plan.json")
    assert plan["source"] == offline_batch["bundle"].manifest
    assert plan["provenance"]["git_head"] == "test-commit"
    assert plan["seed"] == 13 and plan["repeat"] == 1
    assert [sample["task_id"] for sample in plan["samples"]] == ["c", "a", "d", "b"]
    assert offline_batch["loads"] == [
        (ROOT / "data" / "source", ROOT / "data" / "tasks", "dev_quick")
    ]
    for index, (config, task_id, destination, kwargs) in enumerate(offline_batch["calls"], 1):
        assert config == ROOT / "config.toml"
        assert destination == output / "attempts" / f"{index:03d}"
        assert (destination / "business.sqlite").read_text() == task_id
        assert kwargs == {
            "dataset": "dev_quick",
            "user_tier": "user_simulator",
            "judge_tier": "evaluator",
            "seed": 13,
            "repeat": 1,
            "max_steps": 40,
            "timeout_seconds": 600,
            "expected": plan,
        }
        assert summary["results"][index - 1]["attempt_id"] == f"attempt-{index - 1}"
    journal = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert journal == summary["results"]
    for path in output.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text()


@pytest.mark.parametrize(
    "error",
    [
        APIError(SECRET, request=httpx.Request("POST", "https://unit.invalid"), body=None),
        LLMError(SECRET),
        EvaluationIncompleteError(SECRET),
        TimeoutError(SECRET),
    ],
)
def test_known_service_failures_continue_and_zero_completed_has_no_success_rate(
    tmp_path, offline_batch, error
) -> None:
    """四类已知服务失败都保留为未完成，无完成样本时成功率不能伪造成零。"""
    offline_batch["outcomes"] = [error] * 4
    summary = batch.run_batch(ROOT / "config.toml", tmp_path / "unfinished", dataset="dev_64")
    assert len(offline_batch["calls"]) == 4
    assert summary["dataset"] == "dev_64" and summary["completed"] == 0
    assert summary["completion_rate"] == 0
    assert summary["task_success_rate"] is None and summary["db_match_rate"] is None
    assert summary["infrastructure_incomplete"] == 4
    assert all(item["error_type"] == type(error).__name__ for item in summary["results"])


@pytest.mark.parametrize(
    "error", [ValueError("程序错误"), OSError("读写错误"), KeyboardInterrupt()]
)
def test_programming_io_and_user_interrupt_propagate_with_prior_results_intact(
    tmp_path, offline_batch, error
) -> None:
    """编程错误、读写失败和用户中断终止批次，已完成样本与剩余题单可追溯。"""
    offline_batch["outcomes"][1] = error
    output = tmp_path / "interrupted"
    with pytest.raises(type(error)):
        batch.run_batch(ROOT / "config.toml", output)
    assert len(offline_batch["calls"]) == 2
    summary = read_json(output / "summary.json")
    assert summary["status"] == "interrupted"
    assert summary["completed"] == 1 and summary["attempted"] == 2 and summary["pending"] == 2
    assert summary["completion_rate"] == 0.25 and summary["task_success_rate"] == 1
    assert summary["infrastructure_incomplete"] == 0
    assert summary["results"][1]["status"] == "interrupted"
    assert read_json(output / "attempts/001/report.json")["completed"] is True


def test_existing_batch_directory_is_never_overwritten(tmp_path, offline_batch) -> None:
    """再次使用同一批次目录必须拒绝，并完整保留原计划与逐题记录。"""
    output = tmp_path / "existing"
    batch.run_batch(ROOT / "config.toml", output)
    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        batch.run_batch(ROOT / "config.toml", output)
    assert len(offline_batch["calls"]) == 4
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "retail_50"},
        {"seed": True},
        {"max_steps": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": 0},
    ],
)
def test_invalid_batch_parameters_fail_before_loading_sources(
    tmp_path, monkeypatch, kwargs
) -> None:
    """运行参数无效时不读来源、不读凭据，也不创建任何批次目录。"""

    def reject_config(path):
        """非法参数不得继续读取配置。"""
        raise AssertionError("不应读取配置")

    monkeypatch.setattr(batch, "load_config", reject_config)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        batch.run_batch(ROOT / "config.toml", output, **kwargs)
    assert not output.exists()


def test_source_validation_failure_creates_no_batch(tmp_path, offline_batch, monkeypatch) -> None:
    """来源摘要校验失败必须在创建目录及任何单题运行之前传播。"""

    def bad_provenance(config, bundle):
        """模拟固定来源与本地依赖不一致。"""
        raise ValueError("来源版本不一致")

    monkeypatch.setattr(batch, "_provenance", bad_provenance)
    output = tmp_path / "invalid_source"
    with pytest.raises(ValueError, match="来源版本"):
        batch.run_batch(ROOT / "config.toml", output)
    assert not output.exists() and not offline_batch["calls"]


@pytest.mark.parametrize("dataset", ["dev_64", "dev_quick"])
def test_single_development_task_uses_selected_bundle_before_credentials(
    tmp_path, monkeypatch, dataset
) -> None:
    """单题开发入口使用指定转换清单，旧 retail_50 加载路径不会吞掉数据集参数。"""
    bundle = load_retail_bundle(ROOT / "data" / "source")
    task = deepcopy(bundle.tasks[0])
    task["id"] = "development-task"
    selected = replace(bundle, tasks=[task])
    calls = []

    def load_selected(source_dir, task_dir, name):
        """提供含独有题号的开发数据包，用于区分旧来源。"""
        calls.append((source_dir, task_dir, name))
        return selected

    def no_retail(source_dir):
        """开发题单不得走仅加载 retail_50 的入口。"""
        raise AssertionError("不应读取 retail_50")

    def stop_before_credentials(config, model):
        """证明指定任务校验通过后立即停止，不读取任何凭据。"""
        raise RuntimeError("已到模型参数边界")

    monkeypatch.setattr(tasks, "load_task_bundle", load_selected)
    monkeypatch.setattr(runner, "load_retail_bundle", no_retail)
    monkeypatch.setattr(runner, "_model_args", stop_before_credentials)
    output = tmp_path / "unused"
    with pytest.raises(RuntimeError, match="已到模型参数边界"):
        runner.run_single(
            ROOT / "config.toml",
            "development-task",
            output,
            dataset=dataset,
            user_tier="flash",
            judge_tier="flash",
        )
    assert calls == [(ROOT / "data" / "source", ROOT / "data" / "tasks", dataset)]
    assert not output.exists()


@pytest.mark.parametrize("changed", ["dataset", "config", "source", "git_head", "sha256"])
def test_single_rejects_changed_batch_snapshot_before_credentials_or_new_attempt(
    tmp_path, monkeypatch, changed
) -> None:
    """批次计划与本题实际来源、配置或代码不符时，读取密钥与启动模型之前拒绝。"""
    config = load_config(ROOT / "config.toml")
    bundle = load_retail_bundle(ROOT / "data/source")
    provenance = {
        "tau2_commit": bundle.manifest["commit"],
        "git_head": "same-commit",
        "git_status": [],
        "sha256": {"module.py": "same-code"},
    }
    public_config = asdict(config)
    public_config.update(root=str(config.root), runs_dir=str(config.runs_dir))
    expected = {
        "dataset": "retail_50",
        "config": public_config,
        "source": deepcopy(bundle.manifest),
        "provenance": deepcopy(provenance),
    }
    if changed in {"git_head", "sha256"}:
        expected["provenance"][changed] = "different"
    else:
        expected[changed] = "different"

    def no_credentials(*args, **kwargs):
        """快照不符不能继续读取任何凭据。"""
        raise AssertionError("不应读取凭据")

    monkeypatch.setattr(runner, "_provenance", lambda app, data: provenance)
    monkeypatch.setattr(runner, "_model_args", no_credentials)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="批次配置、来源或代码已变化"):
        runner.run_single(
            ROOT / "config.toml",
            bundle.tasks[0]["id"],
            output,
            user_tier="flash",
            judge_tier="flash",
            expected=expected,
        )
    assert not output.exists()


def test_batch_snapshot_ignores_unrelated_git_status_changes(tmp_path, monkeypatch) -> None:
    """未忽略的输出目录只改变 Git 状态，不能误判代码或输入版本已变化。"""
    config = load_config(ROOT / "config.toml")
    bundle = load_retail_bundle(ROOT / "data/source")
    provenance = {
        "tau2_commit": bundle.manifest["commit"],
        "git_head": "same-commit",
        "git_status": ["?? batch-output/"],
        "sha256": {"module.py": "same-code"},
    }
    public_config = asdict(config)
    public_config.update(root=str(config.root), runs_dir=str(config.runs_dir))
    expected = {
        "dataset": "retail_50",
        "config": public_config,
        "source": bundle.manifest,
        "provenance": {**provenance, "git_status": []},
    }

    def stop_at_credentials(*args, **kwargs):
        """用本地边界异常证明快照检查已通过，不读取真实 .env。"""
        raise RuntimeError("已通过快照检查")

    monkeypatch.setattr(runner, "_provenance", lambda app, data: provenance)
    monkeypatch.setattr(runner, "_model_args", stop_at_credentials)
    output = tmp_path / "same"
    with pytest.raises(RuntimeError, match="已通过快照检查"):
        runner.run_single(
            ROOT / "config.toml",
            bundle.tasks[0]["id"],
            output,
            user_tier="flash",
            judge_tier="flash",
            expected=expected,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("completion", "success", "exit_code"), [(1.0, 1.0, 0), (1.0, 0.5, 1), (0.5, 1.0, 2)]
)
def test_batch_cli_forwards_options_and_distinguishes_business_and_infrastructure_failure(
    tmp_path, monkeypatch, capsys, completion, success, exit_code
) -> None:
    """CLI 传递固定题单与运行参数，成功、业务失败及未完成使用不同退出码。"""
    calls = []

    def fake_batch(config_path, output, **kwargs):
        """记录命令行参数并返回已有汇总口径。"""
        calls.append((config_path, output, kwargs))
        return {"completion_rate": completion, "task_success_rate": success}

    output = tmp_path / "cli"
    monkeypatch.setattr(batch, "run_batch", fake_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ttwill",
            "evaluate-batch",
            "--config",
            str(ROOT / "config.toml"),
            "--output",
            str(output),
            "--dataset",
            "dev_64",
            "--user-tier",
            "caller",
            "--judge-tier",
            "judge",
            "--seed",
            "8",
            "--max-steps",
            "20",
            "--timeout-seconds",
            "90",
        ],
    )
    if exit_code:
        with pytest.raises(SystemExit) as caught:
            main()
        assert caught.value.code == exit_code
    else:
        main()
    assert calls == [
        (
            ROOT / "config.toml",
            output,
            {
                "dataset": "dev_64",
                "user_tier": "caller",
                "judge_tier": "judge",
                "seed": 8,
                "max_steps": 20,
                "timeout_seconds": 90.0,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["completion_rate"] == completion


@pytest.fixture
def offline_development_runtime(monkeypatch):
    """仅替换模型传输，以真实开发题单、官方组件和持久化运行时执行批次。

    Args:
        monkeypatch: 测试结束后恢复模型、配置及 HTTP 调用入口。

    Yields:
        本体假模型实例列表和官方模型请求计数，可核对每题隔离。
    """
    base = load_config(ROOT / "config.toml")
    model = ModelConfig(
        model="offline-agent",
        base_url="https://unit.invalid/v1",
        api_key_env="TEST_KEY",
        input_cost_per_million=0,
        output_cost_per_million=0,
    )
    config = replace(
        base,
        llm=replace(base.llm, max_retries=0, retry_delay_seconds=0),
        models={
            "flash": model,
            "user_simulator": replace(model, model="offline-user"),
            "evaluator": replace(model, model="offline-judge"),
        },
    )
    control = {"agents": [], "user_calls": 0, "judge_calls": 0}
    database = load_retail_bundle(ROOT / "data/source").database
    identity_email = next(iter(database["users"].values()))["email"]
    control["identity_email"] = identity_email

    def tool_response(name: str, arguments: dict) -> dict:
        """为公开用户输入构造身份查询、结束计划和回复的工具调用。

        Args:
            name: 协调者的已有工具名称。
            arguments: 用户可见信息或空计划参数，不读取隐藏任务目标。

        Returns:
            带用量的单次工具调用响应。
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
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    @asynccontextmanager
    async def fake_client(env_path):
        """每题按用户提供的邮箱验证身份后回复，真实保存检查点，不读取 .env。"""
        assert env_path == ROOT / ".env"
        fake = FakeModel(
            [
                tool_response("find_user_id_by_email", {"email": identity_email}),
                tool_response("complete_plan", {}),
                tool_response("reply", {"message": "身份已验证，请说明需要办理的业务。"}),
            ]
        )
        control["agents"].append(fake)
        yield fake

    def model_args(app_config, selected):
        """提供仅供本地 MockTransport 使用的虚拟凭据。"""
        return {"api_key": SECRET, "api_base": selected.base_url, "timeout": 5}

    def reject_credentials(*args, **kwargs):
        """离线批次不得接触真实凭据文件。"""
        raise AssertionError("离线测试不能读取 .env")

    def transport(request: httpx.Request) -> httpx.Response:
        """用户主动提供邮箱、验证后退出；判分逐条返回未满足，不假造业务成功。"""
        body = json.loads(request.content)
        if body["model"] == "offline-user":
            control["user_calls"] += 1
            content = (
                f"你好，我需要帮助，我的账户邮箱是 {identity_email}。"
                if control["user_calls"] % 2
                else "这次先不办理了。###STOP###"
            )
        else:
            assert body["model"] == "offline-judge"
            control["judge_calls"] += 1
            assertions = ast.literal_eval(
                body["messages"][-1]["content"].split("expectedOutcomes:\n", 1)[1].strip()
            )
            content = json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": assertion,
                            "metExpectation": False,
                            "reasoning": "离线链路测试只询问信息，未办理业务。",
                        }
                        for assertion in assertions
                    ]
                }
            )
        return httpx.Response(
            200,
            json={
                "id": uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        )

    client = OpenAI(
        api_key=SECRET,
        base_url="https://unit.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )

    def completion(**kwargs):
        """保留真实 LiteLLM 回调和 SDK，仅将网络请求交给本地传输。"""
        return litellm.completion(**kwargs, client=client)

    monkeypatch.setattr(batch, "load_config", lambda path: config)
    monkeypatch.setattr(runner, "load_config", lambda path: config)
    monkeypatch.setattr(runner, "_model_args", model_args)
    monkeypatch.setattr(runner, "dotenv_values", reject_credentials)
    monkeypatch.setattr(runner, "CompatibleModel", fake_client)
    monkeypatch.setattr(llm_utils, "completion", completion)
    try:
        yield control
    finally:
        client.close()


def test_real_quick_set_batch_completes_offline_with_isolated_runtime_and_official_scores(
    tmp_path, offline_development_runtime
) -> None:
    """固定快速集全部题目经真实批次和判分入口完成，模型传输全为本地替身。"""
    output = tmp_path / "offline-quick"
    summary = batch.run_batch(ROOT / "config.toml", output)
    plan = read_json(output / "plan.json")
    expected = tasks.load_task_bundle(ROOT / "data/source", ROOT / "data/tasks", "dev_quick")
    assert 15 <= len(expected.tasks) <= 20
    assert summary["planned"] == summary["completed"] == len(expected.tasks)
    assert summary["completion_rate"] == 1 and summary["infrastructure_incomplete"] == 0
    assert [item["task_id"] for item in plan["samples"]] == [task["id"] for task in expected.tasks]
    assert len(offline_development_runtime["agents"]) == len(expected.tasks)
    assert all(len(agent.requests) == 3 for agent in offline_development_runtime["agents"])
    assert all(
        any(
            message["role"] == "user"
            and offline_development_runtime["identity_email"] in message["content"]
            for message in agent.requests[0]["messages"]
        )
        for agent in offline_development_runtime["agents"]
    )
    assert offline_development_runtime["user_calls"] == len(expected.tasks) * 2
    assert offline_development_runtime["judge_calls"] == sum(
        bool(task["evaluation_criteria"]["nl_assertions"]) for task in expected.tasks
    )
    assert len({item["run_id"] for item in summary["results"]}) == len(expected.tasks)
    assert len({item["attempt_id"] for item in summary["results"]}) == len(expected.tasks)
    for item in summary["results"]:
        destination = output / item["output"]
        report = read_json(destination / "report.json")
        metadata = read_json(destination / "metadata.json")
        assert report["completed"] is True and metadata["dataset"] == "dev_quick"
        assert report["run_id"] == item["run_id"]
        assert metadata["source"] == expected.manifest
        for name in ("business.sqlite", "checkpoints.sqlite", "archive.sqlite"):
            assert (destination / name).stat().st_size > 0
        assert read_json(destination / "initial_database.json") == read_json(
            destination / "final_database.json"
        )
        assert len((destination / "usage.jsonl").read_text().splitlines()) >= 3
        for path in destination.rglob("*.json"):
            assert SECRET not in path.read_text()
