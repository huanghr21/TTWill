"""按固定开发题单逐题运行 D 组，区分业务失败与基础设施未完成。"""

import json
import math
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from openai import APIError

from ttwill.bench.runner import (
    EvaluationIncompleteError,
    _provenance,
    _write_json,
    run_single,
)
from ttwill.bench.tasks import load_task_bundle
from ttwill.config import ConfigError, load_config
from ttwill.llm import LLMError


def _partial_report(output: Path, error_type: str) -> dict:
    """保留单题已写报告中的关联标识，未完成时不采用任何分数。

    Args:
        output: 可能已经创建的单题目录。
        error_type: 错误类型名称，不包含服务异常原文或凭据。

    Returns:
        仅含运行标识、结束原因及未完成状态的报告。
    """
    path = output / "report.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        **{
            key: saved[key]
            for key in ("run_id", "attempt_id", "termination_reason")
            if key in saved
        },
        "completed": False,
        "error_type": error_type,
    }


def _result(sample: dict, report: dict, *, interrupted: bool = False) -> dict:
    """把单题报告转成批次记录，不复制可能包含服务原文的附加字段。"""
    completed = report["completed"]
    status = (
        "interrupted"
        if interrupted
        else "business_success"
        if completed and report["task_success"]
        else "business_failure"
        if completed
        else "infrastructure_incomplete"
    )
    fields = ["run_id", "attempt_id", "termination_reason", "error_type"]
    if completed:
        fields += ["task_success", "db_match", "reward"]
    return {
        **sample,
        "status": status,
        "completed": completed,
        **{key: report[key] for key in fields if key in report},
    }


def _save_summary(output: Path, plan: dict, results: list[dict], status: str) -> dict:
    """原子保存批次进展，成功率只使用已完成判分的样本作分母。

    Args:
        output: 批次独立目录。
        plan: 包含批次身份、固定题单和运行参数的计划。
        results: 按题单顺序追加的每题结果。
        status: running、finished 或 interrupted，表示批次调度状态。

    Returns:
        完成率、已完成样本成功率及逐题结果；无完成样本时成功率为 null。
    """
    finished = [result for result in results if result["completed"]]
    completed = len(finished)
    successes = sum(result["task_success"] for result in finished)
    matches = sum(result["db_match"] for result in finished)
    planned = len(plan["samples"])
    summary = {
        "group": "D",
        "dataset": plan["dataset"],
        "batch_id": plan["batch_id"],
        "seed": plan["seed"],
        "repeat": 1,
        "status": status,
        "planned": planned,
        "attempted": len(results),
        "completed": completed,
        "infrastructure_incomplete": sum(
            result["status"] == "infrastructure_incomplete" for result in results
        ),
        "pending": planned - len(results),
        "completion_rate": completed / planned,
        "task_success_count": successes,
        "task_success_rate": successes / completed if completed else None,
        "db_match_count": matches,
        "db_match_rate": matches / completed if completed else None,
        "updated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    temporary = output / "summary.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output / "summary.json")
    return summary


def _record_result(output: Path, result: dict) -> None:
    """逐题追加结果，使后续中断仍能查到此前已经完成的样本。"""
    with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


def run_batch(
    config_path: Path,
    output: Path,
    *,
    dataset: str = "dev_quick",
    user_tier: str = "user_simulator",
    judge_tier: str = "evaluator",
    seed: int = 0,
    max_steps: int = 100,
    timeout_seconds: float = 600,
) -> dict:
    """按固定清单串行运行开发集，每题一次，每题使用新的业务库和运行目录。

    Args:
        config_path: 模型档位及项目根目录所在的配置文件。
        output: 尚不存在的批次目录，保存计划、逐题记录及进展汇总。
        dataset: dev_quick 或 dev_64；使用已冻结的清单顺序。
        user_tier: 官方模拟用户的模型档位。
        judge_tier: 自然语言断言判分模型的档位。
        seed: 每题官方调度和用户人设分配的固定随机种子。
        max_steps: 每题官方对话调度步数上限。
        timeout_seconds: 每题调度软时限，单位为秒，在官方调度步骤之间检查。

    Returns:
        计划数、完成率、已完成样本的成功率与终态一致率，以及按题结果。
        基础设施未完成样本保留原始尝试，不自动补跑。

    Raises:
        ValueError: 数据集、运行参数或固定来源校验失败，或题间配置、来源、代码变化。
        ConfigError: 配置缺少要求的模型档位。
        FileExistsError: 批次目录已存在，拒绝覆盖历史结果。
        OSError: 来源或产物读写失败，已完成的逐题记录保留。
    """
    if dataset not in {"dev_quick", "dev_64"}:
        raise ValueError("批量开发评测只支持 dev_quick 和 dev_64")
    if type(seed) is not int or type(max_steps) is not int or max_steps < 1:
        raise ValueError("seed 必须是整数，max_steps 必须是正整数")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds 必须是有限正数")
    config = load_config(config_path)
    bundle = load_task_bundle(
        config.root / "data" / "source", config.root / "data" / "tasks", dataset
    )
    required_tiers = {"flash", config.context.summary_tier, user_tier, judge_tier}
    if missing := required_tiers - config.models.keys():
        raise ConfigError(f"评测缺少模型档位：{', '.join(sorted(missing))}")
    task_ids = [task["id"] for task in bundle.tasks]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("固定开发题单必须非空且题目标识不重复")
    provenance = _provenance(config, bundle)
    samples = [
        {"task_id": task_id, "repeat": 1, "output": f"attempts/{index:03d}"}
        for index, task_id in enumerate(task_ids, start=1)
    ]
    public_config = asdict(config)
    public_config.update(root=str(config.root), runs_dir=str(config.runs_dir))
    plan = {
        "group": "D",
        "dataset": dataset,
        "batch_id": uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "repeat": 1,
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
        "roles": {"agent": "flash", "user_simulator": user_tier, "evaluator": judge_tier},
        "config": public_config,
        "source": bundle.manifest,
        "provenance": provenance,
        "samples": samples,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    results = []
    _save_summary(output, plan, results, "running")
    for sample in samples:
        destination = output / sample["output"]
        try:
            report = run_single(
                config_path,
                sample["task_id"],
                destination,
                dataset=dataset,
                user_tier=user_tier,
                judge_tier=judge_tier,
                seed=seed,
                repeat=1,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
                expected=plan,
            )
        except (APIError, LLMError, EvaluationIncompleteError, TimeoutError) as exc:
            report = _partial_report(destination, type(exc).__name__)
        finally:
            if (error := sys.exception()) is not None:
                interrupted = _result(
                    sample, _partial_report(destination, type(error).__name__), interrupted=True
                )
                _record_result(output, interrupted)
                results.append(interrupted)
                _save_summary(output, plan, results, "interrupted")
        result = _result(sample, report)
        _record_result(output, result)
        results.append(result)
        _save_summary(output, plan, results, "running")
    return _save_summary(output, plan, results, "finished")
