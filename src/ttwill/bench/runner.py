"""在独立业务库中运行一题官方模拟对话，保留终态、判分和全部尝试记录。"""

import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

# 官方依赖的价格表不能在导入阶段额外请求网络；实际费用使用本体配置。
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.domains.retail.data_model import RetailDB  # noqa: E402
from tau2.domains.retail.environment import get_environment  # noqa: E402
from tau2.environment.regional import DomainRegionalSettings  # noqa: E402
from tau2.orchestrator.orchestrator import Orchestrator  # noqa: E402
from tau2.runner.build import build_user  # noqa: E402
from tau2.utils.llm_utils import llm_log_dir  # noqa: E402

from ttwill.agents.retail import build_registry  # noqa: E402
from ttwill.backend import RetailBackend  # noqa: E402
from ttwill.bench.agent import TTWillAgent  # noqa: E402
from ttwill.bench.evaluation import EvaluationIncompleteError, evaluate_task  # noqa: E402
from ttwill.bench.persona import build_persona  # noqa: E402
from ttwill.bench.retail import RetailBundle, load_retail_bundle  # noqa: E402
from ttwill.bench.usage import TauUsageCallback  # noqa: E402
from ttwill.bootstrap import load_skills  # noqa: E402
from ttwill.config import AppConfig, ConfigError, ModelConfig, load_config  # noqa: E402
from ttwill.core.context import ContextCompiler  # noqa: E402
from ttwill.core.coordinator import Coordinator  # noqa: E402
from ttwill.core.runtime import PersistentRuntime  # noqa: E402
from ttwill.core.subagent import SubAgentRunner  # noqa: E402
from ttwill.gateway import ToolGateway  # noqa: E402
from ttwill.llm import CompatibleModel, LLMClient  # noqa: E402
from ttwill.persistence import Archive  # noqa: E402
from ttwill.persistence.checkpoints import open_checkpointer  # noqa: E402
from ttwill.tracing import Trace  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    """将一份新的运行产物写成可读 JSON，拒绝覆盖历史文件。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _require_finished_calls(output: Path) -> None:
    """检查本体最终耗尽的传输失败，避免把服务故障计成 Agent 业务失败。

    Args:
        output: 当前尝试的费用账本及模型原文所在目录。

    Raises:
        EvaluationIncompleteError: 某个本体逻辑调用的最后一次尝试仍未取得模型响应。
    """
    ledger = output / "usage.jsonl"
    if not ledger.exists():
        return
    last_attempts = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("tier") is not None:
            last_attempts[item["call_id"]] = item
    for item in last_attempts.values():
        if item["status"] == "error":
            raw = json.loads((output / item["record_ref"]).read_text(encoding="utf-8"))
            if raw["response"] is None:
                raise EvaluationIncompleteError("本体模型调用重试后仍无响应，保留本次尝试待补跑")


def _provenance(config: AppConfig, bundle: RetailBundle) -> dict:
    """校验官方安装版本，记录本体代码与提示资源的实际内容摘要。

    Args:
        config: 含项目根目录的运行配置。
        bundle: 已验证来源的固定中文数据。

    Returns:
        官方提交、本体 Git 提交、工作区状态及逐文件 SHA-256。

    Raises:
        ValueError: 安装的官方依赖与数据清单提交不一致。
        OSError: 代码或 Git 信息无法读取。
    """
    direct_url = json.loads(distribution("tau2").read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != bundle.manifest["commit"]:
        raise ValueError("τ² 安装提交与固定中文来源清单不一致")
    paths = sorted((config.root / "src" / "ttwill").rglob("*.py"))
    paths += sorted(path for path in (config.root / "skills").rglob("*") if path.is_file())
    paths += [config.root / "pyproject.toml", config.root / "uv.lock"]
    hashes = {
        str(path.relative_to(config.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.root, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=config.root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"tau2_commit": commit, "git_head": git_head, "git_status": git_status, "sha256": hashes}


def _model_args(config: AppConfig, model: ModelConfig) -> dict:
    """为官方 LiteLLM 调用解析兼容接口；密钥只留在请求内存中。

    Args:
        config: 指定 .env 所在目录及超时的配置。
        model: 使用兼容 Chat Completions 服务的模型档位。

    Returns:
        服务参数、生成参数及密钥；此对象不能保存到运行产物。

    Raises:
        ConfigError: 服务地址、密钥变量或本地密钥缺失，或参数覆盖传输设置。
    """
    if not model.base_url or not model.api_key_env:
        raise ConfigError("真实评测各模型档位均须配置 base_url 和 api_key_env")
    key = dotenv_values(config.root / ".env", interpolate=False).get(model.api_key_env)
    if not key or not key.strip():
        raise ConfigError(f"本地 .env 缺少变量：{model.api_key_env}")
    reserved = {
        "api_base",
        "base_url",
        "api_key",
        "custom_llm_provider",
        "num_retries",
        "max_retries",
        "success_callback",
        "failure_callback",
        "metadata",
        "caching",
        "cache",
    }
    if reserved & model.parameters.keys():
        raise ConfigError("评测模型 parameters 不能覆盖接口、回调、缓存或重试设置")
    parameters = dict(model.parameters)
    if "thinking" in parameters:
        # OpenAI 兼容入口会把顶层 thinking 当作不支持的 Anthropic 参数丢弃。
        parameters["extra_body"] = {
            **parameters.get("extra_body", {}),
            "thinking": parameters.pop("thinking"),
        }
    return {
        **parameters,
        "api_base": model.base_url,
        "api_key": key,
        "timeout": config.llm.timeout_seconds,
        "caching": False,
    }


@asynccontextmanager
async def _runtime(
    config: AppConfig, bundle: RetailBundle, output: Path, trace: Trace, thread_id: str
) -> AsyncIterator[tuple[PersistentRuntime, RetailBackend]]:
    """组装一个独立评测任务的真实运行时，退出时关闭数据库和模型连接。

    Args:
        config: 本体配置与项目资源根目录。
        bundle: 固定中文种子数据库。
        output: 本次尝试独有的产物目录。
        trace: 带运行关联标识的统一轨迹。
        thread_id: 检查点及档案使用的任务线程标识。

    Yields:
        runtime 为自然对话入口，backend 用于导出实际业务终态。
    """
    backend = RetailBackend(
        output / "business.sqlite", ttl_seconds=config.backend.operation_ttl_seconds
    )
    try:
        backend.initialize(bundle.database)
        skills = load_skills(config.root)
        gateway = ToolGateway(backend, trace, skills)
        async with CompatibleModel(config.root / ".env") as model:
            llm = LLMClient(config, trace, model)
            child = SubAgentRunner(
                build_registry(skills), gateway, llm, max_steps=config.agent.max_steps
            )
            coordinator = Coordinator(gateway, child, llm, max_rounds=config.agent.max_rounds)
            with Archive(output / "archive.sqlite") as archive:
                compiler = ContextCompiler(
                    archive,
                    llm,
                    max_chars=config.context.max_chars,
                    recent_turns=config.context.recent_turns,
                    summary_tier=config.context.summary_tier,
                )
                async with open_checkpointer(output / "checkpoints.sqlite") as saver:
                    yield (
                        PersistentRuntime(
                            coordinator, saver, archive, thread_id=thread_id, compiler=compiler
                        ),
                        backend,
                    )
    finally:
        backend.close()


def run_single(
    config_path: Path,
    task_id: str,
    output: Path,
    *,
    dataset: str = "retail_50",
    user_tier: str = "user_simulator",
    judge_tier: str = "evaluator",
    seed: int = 0,
    repeat: int = 1,
    max_steps: int = 100,
    timeout_seconds: float = 600,
    expected: dict | None = None,
) -> dict:
    """运行一题 D 组对话并判分，每次调用创建新的尝试目录。

    Args:
        config_path: 配置路径；flash、摘要、模拟用户、判分档位都须可实际调用。
        task_id: 指定固定任务集中的题目标识，不据答案筛选任务。
        output: 尚不存在的尝试目录，保留输入、对话、终态、判分和费用。
        dataset: 固定任务集，支持 retail_50、dev_64 和 dev_quick。
        user_tier: 官方模拟用户使用的模型档位。
        judge_tier: 自然语言断言判分使用的模型档位。
        seed: 官方对话和 persona 分配使用的随机种子。
        repeat: 实验重复序号，从 1 开始；补跑使用同序号及新目录。
        max_steps: 官方对话调度步数上限，用户和助手各算一步。
        timeout_seconds: 官方调度器在步骤之间检查的总时限，单位为秒。
        expected: 批次计划中的 dataset、config、source 和 provenance 快照；
            为 None 时独立运行，否则在读取凭据前检查本题来源和配置是否仍一致。

    Returns:
        包含完成状态、独立数据库一致性和完整任务成功判定的报告。

    Raises:
        ConfigError: 模型配置或本地密钥不完整。
        ValueError: 题目、来源提交或运行参数无效，或批次期间配置、来源、代码发生变化。
        FileExistsError: 输出目录已经存在。
        OSError: 资源读取或产物写入失败。
        LLMError: 本体模型调用失败；产物保留为未完成尝试。
    """
    if type(repeat) is not int or repeat < 1 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("repeat 和 max_steps 必须是正整数")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds 必须是有限正数")
    if type(seed) is not int:
        raise ValueError("seed 必须是整数")
    config = load_config(config_path)
    if dataset == "retail_50":
        bundle = load_retail_bundle(config.root / "data" / "source")
    else:
        from ttwill.bench.tasks import load_task_bundle

        bundle = load_task_bundle(
            config.root / "data" / "source", config.root / "data" / "tasks", dataset
        )
    raw_task = next((item for item in bundle.tasks if item["id"] == task_id), None)
    if raw_task is None:
        raise ValueError(f"task_id 不在固定任务集 {dataset} 中")
    task = Task.model_validate(raw_task)
    if task.initial_state is not None:
        raise ValueError("当前单题入口只支持 initial_state 为 None 的任务")
    required_tiers = {"flash", config.context.summary_tier, user_tier, judge_tier}
    if missing := required_tiers - config.models.keys():
        raise ConfigError(f"评测缺少模型档位：{', '.join(sorted(missing))}")
    provenance = _provenance(config, bundle)
    public_config = asdict(config)
    public_config.update(root=str(config.root), runs_dir=str(config.runs_dir))
    if expected is not None and (
        expected["dataset"] != dataset
        or expected["config"] != public_config
        or expected["source"] != bundle.manifest
        or any(
            expected["provenance"][key] != provenance[key]
            for key in ("tau2_commit", "git_head", "sha256")
        )
    ):
        raise ValueError("批次配置、来源或代码已变化，停止本次运行")
    args_by_tier = {tier: _model_args(config, config.models[tier]) for tier in required_tiers}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id, attempt_id = uuid4().hex, uuid4().hex
    trace = Trace(
        output, uuid4().hex, run_id=run_id, task_id=task_id, repeat=repeat, attempt_id=attempt_id
    )
    metadata = {
        "group": "D",
        "dataset": dataset,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "task_id": task_id,
        "repeat": repeat,
        "seed": seed,
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
        "started_at": datetime.now(UTC).isoformat(),
        "config": public_config,
        "roles": {"agent": "flash", "user_simulator": user_tier, "evaluator": judge_tier},
        "source": bundle.manifest,
        "provenance": provenance,
        "tau2_retries": 0,
        "cost_source": "usage.jsonl；未知用量或价格保留 null",
    }
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "task.json", raw_task)
    _write_json(output / "initial_database.json", bundle.database)
    report = {
        "dataset": dataset,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "repeat": repeat,
        "completed": False,
    }
    orchestrator = None
    token = llm_log_dir.set(None)
    try:
        with asyncio.Runner() as loop:
            manager = _runtime(config, bundle, output, trace, run_id)
            runtime, backend = loop.run(manager.__aenter__())
            try:
                # 模拟环境只供官方用户侧调度；本体业务调用始终走独立 SQLite。
                environment = get_environment(
                    db=RetailDB.model_validate(bundle.database),
                    policy=bundle.policy,
                    regional_settings=DomainRegionalSettings(
                        **{
                            key: bundle.settings[key]
                            for key in (
                                "country_code",
                                "currency_code",
                                "currency_scale",
                                "time_zone",
                            )
                        }
                    ),
                )
                user_callback = TauUsageCallback(
                    trace, role="user_simulator", model=config.models[user_tier]
                )
                judge_callback = TauUsageCallback(
                    trace, role="evaluator", model=config.models[judge_tier]
                )
                persona, greeting = build_persona(bundle, task, seed=seed)
                task = task.model_copy(update={"agent_opener": greeting})
                user = build_user(
                    "user_simulator",
                    environment,
                    task,
                    persona_config=persona,
                    llm="openai/" + config.models[user_tier].model,
                    llm_args=args_by_tier[user_tier],
                )
                # 官方构造器深拷贝参数；回调含线程锁，只能在构造后绑定。
                user.llm_args.update(user_callback.request_kwargs())
                _write_json(
                    output / "user_prompt.json",
                    {
                        "system_prompt": user.system_prompt,
                        "sha256": hashlib.sha256(user.system_prompt.encode("utf-8")).hexdigest(),
                        "persona": persona.model_dump(mode="json"),
                        "greeting": greeting,
                    },
                )
                orchestrator = Orchestrator(
                    domain="retail",
                    agent=TTWillAgent(runtime, loop),
                    user=user,
                    environment=environment,
                    task=task,
                    seed=seed,
                    max_steps=max_steps,
                    timeout=timeout_seconds,
                    simulation_id=attempt_id,
                    validate_communication=True,
                )
                with trace.span("session", "官方单题对话与终态判分"):
                    with trace.span("simulation", "官方用户模拟"), user_callback.capture():
                        simulation = orchestrator.run()
                    _write_json(output / "simulation.json", simulation.model_dump(mode="json"))
                    report["termination_reason"] = simulation.termination_reason.value
                    _require_finished_calls(output)
                    with (
                        trace.span("evaluation", "本体终态与自然语言断言"),
                        judge_callback.capture(),
                    ):
                        score = evaluate_task(
                            task,
                            bundle.database,
                            backend.export_snapshot(),
                            simulation.messages,
                            termination_reason=simulation.termination_reason,
                            nl_model="openai/" + config.models[judge_tier].model,
                            nl_model_args={
                                **args_by_tier[judge_tier],
                                **judge_callback.request_kwargs(),
                            },
                        )
                    report.update(completed=True, **score)
            finally:
                try:
                    _write_json(output / "final_database.json", backend.export_snapshot())
                    if orchestrator is not None:
                        _write_json(
                            output / "messages.json",
                            [
                                message.model_dump(mode="json")
                                for message in orchestrator.get_messages()
                            ],
                        )
                finally:
                    loop.run(manager.__aexit__(*sys.exc_info()))
    finally:
        llm_log_dir.reset(token)
        error = sys.exception()
        if error is not None:
            report["error_type"] = type(error).__name__
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(output / "report.json", report)
    return report
