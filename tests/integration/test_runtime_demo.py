"""验证关闭连接后续办同一业务任务的命令行演示与持久化证据。"""

import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend
from ttwill.bootstrap import run_runtime_demo
from ttwill.llm import CompatibleModel
from ttwill.persistence import Archive

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo_config(tmp_path: Path) -> Path:
    """在项目外建立使用固定种子与 Skill 的独立演示配置。

    Args:
        tmp_path: 当前测试的临时目录，独立保存所有运行文件。

    Returns:
        可从其他工作目录执行演示的配置文件绝对路径。
    """
    config = tmp_path / "config.toml"
    config.write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("skills", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    return config


def assert_persistent_demo(output: dict) -> None:
    """核对跨会话重验与重新展示，以及唯一业务操作、退款和持久化记录。

    Args:
        output: run_runtime_demo 或 runtime-demo 命令输出的完整结果。
    """
    first, resumed, confirmed = output["turns"]
    tasks = [turn["tasks"][0] for turn in output["turns"]]
    assert all(len(turn["tasks"]) == 1 for turn in output["turns"])
    assert [task["status"] for task in tasks] == ["pending_confirm", "pending_confirm", "done"]
    assert len({task["id"] for task in tasks}) == 1
    assert {task["revision"] for task in tasks} == {1}
    operation_id = first["displayed"][0]["operation_id"]
    assert all(task["operation_ids"] == [operation_id] for task in tasks)
    assert first["confirmation_results"] == resumed["confirmation_results"] == []
    assert len(first["displayed"]) == len(resumed["displayed"]) == 1
    assert confirmed["displayed"] == []
    assert first["displayed"][0]["session_id"] != resumed["displayed"][0]["session_id"]
    assert first["displayed"][0]["content"] == resumed["displayed"][0]["content"]
    assert first["displayed"][0]["content"] in first["reply"]
    assert resumed["displayed"][0]["content"] in resumed["reply"]
    assert confirmed["confirmation_results"][0]["decision"] == "confirm"
    assert tasks[-1]["facts"][-1]["verification"]["verified"] is True

    database = Path(output["database"])
    backend = RetailBackend(database)
    try:
        snapshot = backend.export_snapshot()
        order = snapshot["orders"][tasks[-1]["order_id"]]
        operation = backend.get_operation(order["user_id"], operation_id=operation_id)
        assert operation["status"] == "committed"
        assert order["status"] == "cancelled"
        assert [entry["transaction_type"] for entry in order["payment_history"]] == [
            "payment",
            "refund",
        ]
        payment, refund = order["payment_history"]
        assert payment["amount"] == refund["amount"]
        assert payment["payment_method_id"] == refund["payment_method_id"]
    finally:
        backend.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM operations").fetchone()[0] == 1
    with sqlite3.connect(output["checkpoint"]) as connection:
        assert connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0] > 0
    with Archive(Path(output["archive"])) as archive:
        messages = archive.messages(database.parent.name, order["user_id"])
        user_messages = [
            entry["payload"]["content"] for entry in messages if entry["payload"]["role"] == "user"
        ]
        assert "继续办理" in user_messages
        assert "确认" in user_messages
        assistant_messages = [
            entry["payload"].get("content")
            for entry in messages
            if entry["payload"]["role"] == "assistant"
        ]
        assert all(turn["reply"] in assistant_messages for turn in output["turns"])

    assert len(set(output["traces"])) == 2
    sessions = [
        [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        for path in output["traces"]
    ]
    ended = [[event for event in events if event["event"] == "end"] for events in sessions]
    for session in ended:
        assert any(
            event["kind"] == "tool"
            and event["name"] == "find_user_id_by_email"
            and event["attrs"]["result"] == order["user_id"]
            for event in session
        )
    assert not any(
        event["kind"] == "tool" and event["attrs"].get("kind") == "commit" for event in ended[0]
    )
    commits = [
        event
        for event in ended[1]
        if event["kind"] == "tool" and event["attrs"].get("kind") == "commit"
    ]
    assert len(commits) == 1
    assert commits[0]["attrs"]["operation_id"] == operation_id
    assert any(event["kind"] == "recovery" for event in ended[1])
    decisions = [event for event in ended[1] if event["kind"] == "confirmation"]
    assert len(decisions) == 1
    assert decisions[0]["attrs"]["user_message"] == "确认"
    assert decisions[0]["attrs"]["displayed"] == [
        {"operation_id": operation_id, "content": resumed["displayed"][0]["content"]}
    ]
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in database.parent.glob("models/*.json")
    ]
    assert records
    assert all(record["response"]["model"] == "fake-demo" for record in records)
    assert all(
        tool["function"]["name"] != "commit"
        for record in records
        for tool in record["request"].get("tools", [])
    )


def test_runtime_demo_from_other_directory(demo_config: Path, tmp_path: Path) -> None:
    """项目外命令行演示真实重开持久化文件，跨会话重新展示后才能确认。"""
    process = subprocess.run(
        [
            str(Path(sys.executable).parent / "ttwill"),
            "runtime-demo",
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
    assert_persistent_demo(output)
    for key in ("database", "checkpoint", "archive"):
        assert Path(output[key]).is_relative_to(tmp_path / "runs")
    assert all(Path(path).is_relative_to(tmp_path / "runs") for path in output["traces"])


def test_runtime_demo_never_builds_real_model_client(demo_config: Path, monkeypatch) -> None:
    """恢复、上下文编译与三轮对话完全使用假模型。"""

    def reject_real_client(*args, **kwargs):
        """构造真实模型客户端即使未联网也让测试失败。"""
        raise AssertionError("持久化演示不能创建真实模型客户端")

    monkeypatch.setattr(CompatibleModel, "__init__", reject_real_client)
    assert_persistent_demo(asyncio.run(run_runtime_demo(demo_config)))
