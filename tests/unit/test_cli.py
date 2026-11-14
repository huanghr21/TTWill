import json
import subprocess
import sys
from pathlib import Path


def test_demo_and_trace_commands_work_outside_project(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    config = config_dir / "config.toml"
    config.write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    command = [str(Path(sys.executable).parent / "ttwill")]
    demo = subprocess.run(
        [*command, "demo", "--config", str(config)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "假模型调用成功" in demo.stdout
    assert "session" in demo.stdout and "└─ llm" in demo.stdout
    traces = list((config_dir / "runs").glob("*/traces/*.jsonl"))
    assert len(traces) == 1
    events = [json.loads(line) for line in traces[0].read_text().splitlines()]
    assert len(events) == 6
    llm = next(event for event in events if event["kind"] == "llm" and event["event"] == "end")
    assert llm["status"] == "ok" and llm["attrs"]["model"] == "fake-demo"
    run_dir = traces[0].parents[1]
    raw = json.loads((run_dir / llm["attrs"]["record_ref"]).read_text())
    assert raw["request"]["messages"] == [{"role": "user", "content": "你好"}]
    assert raw["response"]["choices"][0]["message"]["content"] == "假模型调用成功。"
    usage = json.loads((run_dir / "usage.jsonl").read_text())
    assert usage["input_tokens"] == 12 and usage["output_tokens"] == 6
    assert usage["cost_usd"] == 0
    view = subprocess.run(
        [*command, "trace", str(traces[0])],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert view.stdout in demo.stdout
    assert not (tmp_path / "runs").exists()
