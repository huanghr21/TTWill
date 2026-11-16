"""从命令行验证两轮请求、展示、确认与真实 SQLite 提交核验。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend
from ttwill.bootstrap import run_coordinator_demo
from ttwill.llm import CompatibleModel

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo_config(tmp_path: Path) -> Path:
    """在项目外提供运行配置，并链接只读种子与 Skill 资源。

    Args:
        tmp_path: 当前测试独占的临时目录，运行产物保存在其中。

    Returns:
        可从任意工作目录使用的 TOML 配置路径。
    """
    config = tmp_path / "config.toml"
    config.write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("skills", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    return config


def assert_completed_demo(output: dict) -> None:
    """核对两轮快照、实际提交、单次退款与贯穿全程的轨迹证据。

    Args:
        output: run_coordinator_demo 或命令行输出的完整结果对象。
    """
    first, second = output["turns"]
    assert len(first["tasks"]) == len(second["tasks"]) == 1
    first_task = first["tasks"][0]
    second_task = second["tasks"][0]
    assert first_task["status"] == "pending_confirm"
    assert second_task["status"] == "done"
    assert first_task["id"] == second_task["id"]
    assert first["confirmation_results"] == []
    assert len(second["confirmation_results"]) == 1
    assert len(first["displayed"]) == 1
    assert second["displayed"] == []
    shown = first["displayed"][0]
    operation_id = shown["operation_id"]
    assert shown["content"] in first["reply"]
    assert first_task["operation_ids"] == second_task["operation_ids"] == [operation_id]
    backend = RetailBackend(Path(output["database"]))
    try:
        snapshot = backend.export_snapshot()
        order = snapshot["orders"][second_task["order_id"]]
        operation = backend.get_operation(order["user_id"], operation_id=operation_id)
        assert operation["status"] == "committed"
        assert operation["action"] == "cancel_pending_order"
        assert operation["args"]["order_id"] == second_task["order_id"]
        assert order["status"] == "cancelled"
        payments = [
            entry for entry in order["payment_history"] if entry["transaction_type"] == "payment"
        ]
        refunds = [
            entry for entry in order["payment_history"] if entry["transaction_type"] == "refund"
        ]
        assert len(payments) == len(refunds) == 1
        assert refunds[0]["amount"] == payments[0]["amount"]
        assert refunds[0]["payment_method_id"] == payments[0]["payment_method_id"]
    finally:
        backend.close()
    events = [json.loads(line) for line in Path(output["trace"]).read_text().splitlines()]
    ended = [event for event in events if event["event"] == "end"]
    assert len([event for event in ended if event["kind"] == "turn"]) == 2
    assert {"coordinator", "task_update", "dispatch", "confirmation", "verification", "reply"} <= {
        event["kind"] for event in ended
    }
    confirmations = [event for event in ended if event["kind"] == "confirmation"]
    confirmed = next(event for event in confirmations if event["attrs"]["user_message"] == "确认")
    assert confirmed["attrs"]["source"] == "rule"
    assert confirmed["attrs"]["displayed"] == [
        {"operation_id": operation_id, "content": shown["content"]}
    ]
    assert confirmed["attrs"]["results"][0]["decision"] == "confirm"
    assert all(event["status"] == "ok" for event in ended if event["kind"] == "verification")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in Path(output["database"]).parent.glob("models/*.json")
    ]
    assert len(records) == 11
    assert all(record["response"]["model"] == "fake-demo" for record in records)
    assert all(
        tool["function"]["name"] != "commit"
        for record in records
        for tool in record["request"].get("tools", [])
    )


def test_coordinator_demo_from_other_directory(demo_config: Path, tmp_path: Path) -> None:
    """项目外 CLI 请求停在待确认，第二轮确认后由代码提交并核验为完成。"""
    process = subprocess.run(
        [
            str(Path(sys.executable).parent / "ttwill"),
            "coordinator-demo",
            "--config",
            str(demo_config),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout)
    assert_completed_demo(output)
    assert Path(output["database"]).is_relative_to(tmp_path / "runs")
    assert Path(output["trace"]).is_relative_to(tmp_path / "runs")


def test_coordinator_demo_never_creates_real_model_client(demo_config: Path, monkeypatch) -> None:
    """两轮演示完全使用假模型，所有真实服务客户端构造均应不可达。"""

    def reject_real_client(*args, **kwargs):
        """任何真实模型客户端构造都使测试立即失败。"""
        raise AssertionError("假模型演示不能创建真实服务客户端")

    monkeypatch.setattr(CompatibleModel, "__init__", reject_real_client)
    output = asyncio.run(run_coordinator_demo(demo_config))
    assert_completed_demo(output)
