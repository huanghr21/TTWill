"""提供本体演示、业务操作、任务转换、离线重放、模型评测和轨迹查看。"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from ttwill.backend import BusinessError, RetailBackend
from ttwill.bootstrap import (
    business_runtime,
    load_demo_config,
    run_agent_demo,
    run_coordinator_demo,
    run_runtime_demo,
)
from ttwill.gateway import AFTER_SALES_WRITES, READ_TOOLS, WRITE_TOOLS, CallContext
from ttwill.llm import FakeModel, LLMClient
from ttwill.tracing import Trace, render_tree


async def run_demo(config_path: Path) -> Path:
    """运行一次假模型调用，在新运行目录记录结果并打印轨迹树。

    Args:
        config_path: TOML 运行配置路径，须包含 flash 模型档位。

    Returns:
        本次会话的 JSONL 轨迹文件路径。

    Raises:
        ConfigError: 配置字段缺失或取值不符合约定。
        ValueError: 未配置 flash 档位。
        OSError: 配置无法读取，或运行记录无法写入。
    """
    config = load_demo_config(config_path)
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


def _json_object(value: str) -> dict:
    """将命令行参数解析为 JSON 对象。

    Args:
        value: 命令行传入的 JSON 文本。

    Returns:
        解析后的参数字典。

    Raises:
        argparse.ArgumentTypeError: 文本不是合法 JSON，或顶层值不是对象。
    """
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("参数必须是合法 JSON") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("参数必须是 JSON 对象")
    return result


def _business_parser(subparsers) -> None:
    """向主解析器注册初始化、查询、准备、提交、拒绝和操作对账命令。

    Args:
        subparsers: 主解析器通过 add_subparsers 创建的子命令注册入口。
    """
    business = subparsers.add_parser("business", help="查询业务及执行准备/提交")
    actions = business.add_subparsers(dest="action", required=True)
    initialize = actions.add_parser("init", help="将固定中文种子数据导入空库")
    initialize.add_argument("--source", type=Path, default=Path("data/source"))
    initialize.add_argument("--db", type=Path, required=True)
    for name, description in (
        ("query", "查询已验证用户的业务数据"),
        ("prepare", "预演并保存写操作"),
        ("commit", "提交已展示且参数一致的操作"),
        ("reject", "拒绝或撤销待提交操作"),
        ("operation", "读取操作的当前状态"),
    ):
        action = actions.add_parser(name, help=description)
        action.add_argument("--db", type=Path, required=True)
        action.add_argument("--config", type=Path, default=Path("config.toml"))
        action.add_argument("--email", required=True)
        if name in {"query", "prepare"}:
            action.add_argument(
                "--tool",
                choices=sorted(READ_TOOLS if name == "query" else WRITE_TOOLS),
                required=True,
            )
        if name in {"query", "prepare", "commit"}:
            action.add_argument("--args", type=_json_object, required=True)
        if name == "prepare":
            action.add_argument("--task-id", required=True)
            action.add_argument("--revision", type=int, default=1)
        if name in {"commit", "reject"}:
            action.add_argument("--operation-id", required=True)
        if name == "reject":
            action.add_argument("--reason", required=True)
        if name == "operation":
            identifier = action.add_mutually_exclusive_group(required=True)
            identifier.add_argument("--operation-id")
            identifier.add_argument("--request-key")


def run_business(args: argparse.Namespace) -> dict:
    """执行业务子命令并记录轨迹；初始化导入种子，其余操作重新验证身份。

    Args:
        args: business 子命令的解析结果，包含动作、业务库路径和动作所需字段；
            查询和写入命令还包含配置路径与验证身份用的邮箱。

    Returns:
        init 返回业务库路径和实体数量；query 返回工具查询结果；
        prepare、commit、reject、operation 返回含状态和参数的后端操作记录。

    Raises:
        BusinessError: 身份、数据归属、操作状态或业务规则不满足要求。
        ValueError: 种子来源校验失败、配置无效或工具参数不符合约定。
        OSError: 配置或来源文件无法读取、业务库文件不存在，或运行记录无法写入。
    """
    if args.action == "init":
        from ttwill.bench.retail import load_retail_bundle

        bundle = load_retail_bundle(args.source)
        backend = RetailBackend(args.db)
        try:
            backend.initialize(bundle.database)
        finally:
            backend.close()
        return {
            "database": str(args.db.resolve()),
            "entities": {name: len(entities) for name, entities in bundle.database.items()},
        }
    with business_runtime(args.config, args.db) as (gateway, session):
        context = CallContext(session, "coordinator", uuid4().hex)
        gateway.call(context, "find_user_id_by_email", {"email": args.email})
        if args.action in {"query", "prepare"}:
            capability = "after_sales" if args.tool in AFTER_SALES_WRITES else "order_service"
            context = CallContext(
                session,
                capability,
                uuid4().hex,
                getattr(args, "task_id", None),
                getattr(args, "revision", 1),
            )
            try:
                if args.action == "prepare":
                    gateway.call(
                        context, "load_skill", {"name": gateway.skills.required_skill(args.tool)}
                    )
                return gateway.call(context, args.tool, args.args)
            finally:
                gateway.release_instance(context)
        if args.action == "commit":
            return gateway.commit(context, args.operation_id, args.args)
        if args.action == "reject":
            return gateway.reject(context, args.operation_id, args.reason)
        with gateway.trace.span("tool", "get_operation") as span:
            result = gateway.backend.get_operation(
                session.user_id, operation_id=args.operation_id, request_key=args.request_key
            )
            span.attrs["result"] = result
            return result


def main() -> None:
    """解析进程命令行参数，并向标准输出写入 JSON 或轨迹树。

    Raises:
        SystemExit: 参数或业务运行错误时退出码为 2；参考动作存在差异时为 1。
    """
    parser = argparse.ArgumentParser(prog="ttwill", description="TTWill 命令行入口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="用假模型验证调用、记录和轨迹树")
    demo.add_argument("--config", type=Path, default=Path("config.toml"), help="TOML 配置文件")
    agent_demo = subparsers.add_parser("agent-demo", help="用假模型演示单次子 Agent 派工")
    agent_demo.add_argument("--config", type=Path, default=Path("config.toml"))
    agent_demo.add_argument(
        "--capability", choices=["order_service", "after_sales"], default="order_service"
    )
    coordinator_demo = subparsers.add_parser(
        "coordinator-demo", help="用假模型演示请求、确认与核验的两轮对话"
    )
    coordinator_demo.add_argument("--config", type=Path, default=Path("config.toml"))
    runtime_demo = subparsers.add_parser(
        "runtime-demo", help="用假模型演示跨会话恢复、重新展示与确认核验"
    )
    runtime_demo.add_argument("--config", type=Path, default=Path("config.toml"))
    trace = subparsers.add_parser("trace", help="离线查看会话 JSONL 轨迹树")
    trace.add_argument("path", type=Path, help="会话轨迹文件")
    _business_parser(subparsers)
    bench = subparsers.add_parser("bench", help="离线参考动作重放，需要 bench 可选依赖")
    bench.add_argument("--source", type=Path, default=Path("data/source"))
    bench.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    bench.add_argument(
        "--dataset", choices=["retail_50", "dev_64", "dev_quick"], default="retail_50"
    )
    bench.add_argument("--output", type=Path, required=True, help="新的重放产物目录")
    convert = subparsers.add_parser("convert-tasks", help="离线转换开发题并校验官方 50 题")
    convert.add_argument("--source", type=Path, default=Path("data/source"))
    convert.add_argument("--output", type=Path, default=Path("data/tasks"))
    quick = subparsers.add_parser(
        "freeze-quick-set", help="按固定种子生成开发快速集，不覆盖已有清单"
    )
    quick.add_argument("--tasks", type=Path, default=Path("data/tasks"))
    quick.add_argument("--seed", type=int, default=0)
    quick.add_argument("--size", type=int, default=20)
    evaluate = subparsers.add_parser("evaluate", help="运行一题官方用户模拟与终态判分")
    evaluate.add_argument("--config", type=Path, default=Path("config.toml"))
    evaluate.add_argument("--task-id", required=True, help="所选固定任务集中的题目标识")
    evaluate.add_argument(
        "--dataset", choices=["retail_50", "dev_64", "dev_quick"], default="retail_50"
    )
    evaluate.add_argument("--output", type=Path, required=True, help="尚不存在的尝试目录")
    evaluate.add_argument("--user-tier", default="user_simulator")
    evaluate.add_argument("--judge-tier", default="evaluator")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--repeat", type=int, default=1)
    evaluate.add_argument("--max-steps", type=int, default=100)
    evaluate.add_argument("--timeout-seconds", type=float, default=600)
    batch = subparsers.add_parser("evaluate-batch", help="按固定开发清单逐题运行 D 组并汇总")
    batch.add_argument("--config", type=Path, default=Path("config.toml"))
    batch.add_argument("--dataset", choices=["dev_64", "dev_quick"], default="dev_quick")
    batch.add_argument("--output", type=Path, required=True, help="尚不存在的批量运行目录")
    batch.add_argument("--user-tier", default="user_simulator")
    batch.add_argument("--judge-tier", default="evaluator")
    batch.add_argument("--seed", type=int, default=0)
    batch.add_argument("--max-steps", type=int, default=100)
    batch.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            asyncio.run(run_demo(args.config))
        elif args.command == "agent-demo":
            print(
                json.dumps(
                    asyncio.run(run_agent_demo(args.config, args.capability)),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "coordinator-demo":
            print(
                json.dumps(
                    asyncio.run(run_coordinator_demo(args.config)), ensure_ascii=False, indent=2
                )
            )
        elif args.command == "runtime-demo":
            print(
                json.dumps(asyncio.run(run_runtime_demo(args.config)), ensure_ascii=False, indent=2)
            )
        elif args.command == "trace":
            print(render_tree(args.path))
        elif args.command == "business":
            print(json.dumps(run_business(args), ensure_ascii=False, indent=2))
        elif args.command == "convert-tasks":
            from ttwill.bench.conversion import build_development_tasks

            report = build_development_tasks(args.source, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "freeze-quick-set":
            from ttwill.bench.tasks import generate_quick_set

            report = generate_quick_set(args.tasks, seed=args.seed, size=args.size)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "evaluate-batch":
            from ttwill.bench.batch import run_batch

            report = run_batch(
                args.config,
                args.output,
                dataset=args.dataset,
                user_tier=args.user_tier,
                judge_tier=args.judge_tier,
                seed=args.seed,
                max_steps=args.max_steps,
                timeout_seconds=args.timeout_seconds,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if report["completion_rate"] < 1:
                raise SystemExit(2)
            if report["task_success_rate"] < 1:
                raise SystemExit(1)
        elif args.command == "evaluate":
            from openai import APIError

            from ttwill.bench.evaluation import EvaluationIncompleteError
            from ttwill.bench.runner import run_single
            from ttwill.llm import LLMError

            try:
                report = run_single(
                    args.config,
                    args.task_id,
                    args.output,
                    user_tier=args.user_tier,
                    judge_tier=args.judge_tier,
                    seed=args.seed,
                    repeat=args.repeat,
                    max_steps=args.max_steps,
                    timeout_seconds=args.timeout_seconds,
                    **({"dataset": args.dataset} if args.dataset != "retail_50" else {}),
                )
            except (APIError, LLMError, EvaluationIncompleteError, TimeoutError) as exc:
                parser.exit(
                    2,
                    f"评测未完成：{type(exc).__name__}；"
                    f"尝试记录：{(args.output / 'report.json').resolve()}\n",
                )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["task_success"]:
                raise SystemExit(1)
        else:
            from ttwill.bench.replay import replay_tasks
            from ttwill.bench.tasks import load_task_bundle

            report = replay_tasks(
                load_task_bundle(args.source, args.tasks, args.dataset), args.output
            )
            print(
                json.dumps(
                    {
                        "report": str((args.output / "report.json").resolve()),
                        "task_count": report["task_count"],
                        "hash_equal_count": report["hash_equal_count"],
                    },
                    ensure_ascii=False,
                )
            )
            if (
                report["hash_equal_count"] != report["task_count"]
                or report["actions_equal_count"] != report["task_count"]
                or not all(check["matched"] for check in report["read_tool_checks"])
            ):
                raise SystemExit(1)
    except (BusinessError, ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
