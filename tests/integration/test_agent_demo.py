"""通过命令行验证两种子 Agent 演示及超限后的准备清理。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("capability", "steps", "status"),
    [
        ("order_service", 12, "need_confirm"),
        ("after_sales", 12, "need_confirm"),
        ("order_service", 3, "failed"),
    ],
)
def test_agent_demo_from_other_directory(
    tmp_path: Path, capability: str, steps: int, status: str
) -> None:
    """从项目外派工，结果有真实操作依据；准备和超限清理都不改变订单。"""
    config = tmp_path / "config.toml"
    config.write_text(
        (ROOT / "config.toml").read_text().replace("max_steps = 12", f"max_steps = {steps}")
    )
    for name in ("skills", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    process = subprocess.run(
        [
            str(Path(sys.executable).parent / "ttwill"),
            "agent-demo",
            "--config",
            str(config),
            "--capability",
            capability,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    output = json.loads(process.stdout)
    result = output["result"]
    assert result["status"] == status
    assert result["steps"] == min(steps, 4)
    action = next(action for action in result["actions"] if action["status"] == "prepared")
    backend = RetailBackend(Path(output["database"]))
    try:
        operation = backend.get_operation(result["user_id"], operation_id=action["operation_id"])
        assert operation["status"] == ("prepared" if status == "need_confirm" else "rejected")
        order = backend.query(
            result["user_id"], "get_order_details", {"order_id": action["args"]["order_id"]}
        )
        assert order["status"] == ("pending" if capability == "order_service" else "delivered")
    finally:
        backend.close()
    if status == "need_confirm":
        assert result["pending_confirmation"][0]["operation_id"] == operation["operation_id"]
    else:
        assert result["pending_confirmation"] == []
        assert "上限" in result["reason"]
    events = [json.loads(line) for line in Path(output["trace"]).read_text().splitlines()]
    assert {"session", "turn", "dispatch", "subagent", "llm", "tool", "result"} <= {
        event["kind"] for event in events
    }
    assert not any(event["name"] == "commit" for event in events)
    records = list(Path(output["database"]).parent.glob("models/*.json"))
    assert len(records) == min(steps, 4)
