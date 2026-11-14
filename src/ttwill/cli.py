"""基础设施演示与离线轨迹查看。"""

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from ttwill.config import load_config
from ttwill.llm import FakeModel, LLMClient
from ttwill.tracing import Trace, render_tree


async def run_demo(config_path: Path) -> Path:
    """按指定配置运行一次假模型调用，打印轨迹树并返回会话 JSONL 路径。

    配置须包含 flash 档位。每次调用新建运行目录，不读取密钥或访问模型服务。
    """
    config = load_config(config_path)
    run_id = f"demo-{uuid4().hex}"
    session_id = uuid4().hex
    trace = Trace(config.runs_dir / run_id, session_id, run_id=run_id)
    fake = FakeModel(
        [
            {
                "model": "fake-demo",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "假模型调用成功。"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            }
        ]
    )
    client = LLMClient(config, trace, fake)
    with trace.span("session", "基础设施演示"):
        with trace.span("turn", "第 1 轮", turn=1):
            response = await client.complete("flash", [{"role": "user", "content": "你好"}])
    print(response.message["content"])
    path = config.runs_dir / run_id / "traces" / f"{session_id}.jsonl"
    print(f"轨迹：{path}")
    print(render_tree(path))
    return path


def main() -> None:
    """解析命令行参数，执行假模型演示或离线轨迹查看。"""
    parser = argparse.ArgumentParser(prog="ttwill", description="TTWill 命令行入口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="用假模型验证调用、记录和轨迹树")
    demo.add_argument("--config", type=Path, default=Path("config.toml"), help="TOML 配置文件")
    trace = subparsers.add_parser("trace", help="离线查看会话 JSONL 轨迹树")
    trace.add_argument("path", type=Path, help="会话轨迹文件")
    args = parser.parse_args()
    if args.command == "demo":
        asyncio.run(run_demo(args.config))
    else:
        print(render_tree(args.path))


if __name__ == "__main__":
    main()
