"""通过真实命令行进程验收查询、准备、提交与操作恢复。"""

import json
import subprocess
import sys
from pathlib import Path

from ttwill.backend import RetailBackend

ROOT = Path(__file__).resolve().parents[2]


def invoke(
    tmp_path: Path, action: str, *arguments: str, email: str = "u1@example.com"
) -> subprocess.CompletedProcess:
    """从项目外执行一条业务命令，并将数据库和轨迹写入测试目录。

    Args:
        tmp_path: 本项测试的工作目录，含 business.sqlite，并接收配置与轨迹。
        action: business 命令下的子命令名称。
        *arguments: 追加到命令末尾的参数，各元素对应一个命令行参数。
        email: 本次命令验证身份使用的邮箱。

    Returns:
        子进程执行结果，包含退出码以及文本形式的标准输出和标准错误。
    """
    config = tmp_path / "config.toml"
    config.write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("skills", "data"):
        if not (tmp_path / name).exists():
            (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    return subprocess.run(
        [
            str(Path(sys.executable).parent / "ttwill"),
            "business",
            action,
            "--db",
            str(tmp_path / "business.sqlite"),
            "--config",
            str(config),
            "--email",
            email,
            *arguments,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_commit_and_retry_across_cli_processes(
    tmp_path: Path, backend: RetailBackend
) -> None:
    """分进程准备和提交同一操作，准备不改库，重试不产生第二次退款。"""
    args = json.dumps({"order_id": "#W0000001", "reason": "no longer needed"})
    prepared = invoke(
        tmp_path, "prepare", "--tool", "cancel_pending_order", "--task-id", "task-1", "--args", args
    )
    assert prepared.returncode == 0, prepared.stderr
    operation = json.loads(prepared.stdout)
    assert operation["status"] == "prepared"
    query = invoke(
        tmp_path, "query", "--tool", "get_order_details", "--args", '{"order_id":"#W0000001"}'
    )
    assert query.returncode == 0, query.stderr
    assert json.loads(query.stdout)["status"] == "pending"
    lookup = invoke(tmp_path, "operation", "--request-key", operation["request_key"])
    assert json.loads(lookup.stdout)["operation_id"] == operation["operation_id"]
    invalid = invoke(
        tmp_path, "commit", "--operation-id", operation["operation_id"], "--args", "{}"
    )
    assert invalid.returncode == 2 and "参数不一致" in invalid.stderr
    for _ in range(2):
        commit = invoke(
            tmp_path, "commit", "--operation-id", operation["operation_id"], "--args", args
        )
        assert commit.returncode == 0, commit.stderr
        assert json.loads(commit.stdout)["status"] == "committed"
    order = backend.query("u1", "get_order_details", {"order_id": "#W0000001"})
    assert order["status"] == "cancelled"
    assert len(order["payment_history"]) == 2
    traces = list((tmp_path / "runs").glob("*/traces/*.jsonl"))
    assert len(traces) == 6
    assert not (tmp_path / "models").exists()


def test_query_rejects_other_users_order(tmp_path: Path, backend: RetailBackend) -> None:
    """命令行重新验证当前用户，不能按参数读取其他用户的订单。"""
    result = invoke(
        tmp_path, "query", "--tool", "get_order_details", "--args", '{"order_id":"#W0000003"}'
    )
    assert result.returncode == 2
    assert "order_id 不合法" in result.stderr
    assert json.loads(result.stderr.split("合法值：", 1)[1]) == [
        "#W0000001",
        "#W0000002",
        "#W0000004",
    ]
    assert "#W0000003" not in result.stderr
    assert backend.export_snapshot()["orders"]["#W0000003"]["status"] == "pending"


def test_reject_then_recover_operation(tmp_path: Path, backend: RetailBackend) -> None:
    """拒绝后的操作可被新进程按 ID 找回，且未改业务订单。"""
    prepared = invoke(
        tmp_path,
        "prepare",
        "--tool",
        "cancel_pending_order",
        "--task-id",
        "task-1",
        "--args",
        '{"order_id":"#W0000001","reason":"ordered by mistake"}',
    )
    operation_id = json.loads(prepared.stdout)["operation_id"]
    result = invoke(tmp_path, "reject", "--operation-id", operation_id, "--reason", "用户撤销")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "rejected"
    recovered = invoke(tmp_path, "operation", "--operation-id", operation_id)
    assert json.loads(recovered.stdout)["reason"] == "用户撤销"
    assert backend.export_snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_missing_database_is_reported_without_creating_file(tmp_path: Path) -> None:
    """路径拼错时显示初始化提示，不生成空业务库。"""
    result = invoke(
        tmp_path, "query", "--tool", "get_order_details", "--args", '{"order_id":"#W0000001"}'
    )
    assert result.returncode == 2 and "请先初始化" in result.stderr
    assert not (tmp_path / "business.sqlite").exists()


def test_non_object_json_is_rejected_before_opening_database(tmp_path: Path) -> None:
    """非对象 JSON 在命令行参数边界被拒绝。"""
    result = invoke(tmp_path, "query", "--tool", "get_order_details", "--args", "[]")
    assert result.returncode == 2 and "JSON 对象" in result.stderr
    assert not (tmp_path / "business.sqlite").exists()
