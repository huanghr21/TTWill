"""组装命令行业务入口使用的后端、网关和会话轨迹。"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ttwill.agents.retail import build_registry
from ttwill.backend import RetailBackend
from ttwill.bench.policy import extract_policy
from ttwill.bench.retail import load_retail_bundle
from ttwill.config import load_config
from ttwill.core.subagent import DispatchTask, SubAgentRunner
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.llm import FakeModel, LLMClient
from ttwill.tracing import Trace


def load_skills(root: Path) -> SkillCatalog:
    """从配置根目录读取 Skill 声明及固定来源中的政策原文。

    Args:
        root: 含 skills/ 与 data/source/ 的项目根目录。

    Returns:
        已校验章节引用并应用协议条款排除清单的静态 Skill 目录。

    Raises:
        OSError: 来源文件或 Skill 目录无法读取。
        ValueError: 来源校验失败、政策结构或 Skill 声明无效。
    """
    bundle = load_retail_bundle(root / "data" / "source")
    sections, exclusions = extract_policy(bundle.policy)
    return SkillCatalog(root / "skills", sections, exclusions)


async def run_agent_demo(config_path: Path, capability: str) -> dict:
    """在独立业务库上演示子 Agent 查询、加载 Skill 和准备操作。

    Args:
        config_path: 含 flash 档位的配置路径，资源目录相对于其所在目录。
        capability: order_service 演示取消待发货订单，after_sales 演示退货。

    Returns:
        含 result、database 和 trace 的字典；结果停在待确认状态，业务数据未提交。

    Raises:
        ValueError: 能力名称、配置、种子或 Skill 声明无效。
        OSError: 资源无法读取，或业务库、运行记录无法写入。
    """
    if capability not in {"order_service", "after_sales"}:
        raise ValueError(f"未知演示能力：{capability}")
    config = load_config(config_path)
    bundle = load_retail_bundle(config.root / "data" / "source")
    skills = load_skills(config.root)
    desired_status = "pending" if capability == "order_service" else "delivered"
    order = next(
        order for order in bundle.database["orders"].values() if order["status"] == desired_status
    )
    user = bundle.database["users"][order["user_id"]]
    action = (
        "cancel_pending_order" if capability == "order_service" else "return_delivered_order_items"
    )
    arguments = {"order_id": order["order_id"]}
    if capability == "order_service":
        arguments["reason"] = "no longer needed"
    else:
        arguments.update(
            item_ids=[item["item_id"] for item in order["items"]],
            payment_method_id=order["payment_history"][0]["payment_method_id"],
        )
    calls = [
        ("get_order_details", {"order_id": order["order_id"]}),
        ("load_skill", {"name": skills.required_skill(action)}),
        (action, arguments),
        ("finish", {"missing": [], "message_hint": "已准备操作，请核对详情后确认。"}),
    ]
    responses = [
        {
            "model": "fake-demo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"demo-{index}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args, ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }
        for index, (name, args) in enumerate(calls)
    ]
    run_id = f"agent-demo-{uuid4().hex}"
    session = SessionContext(uuid4().hex)
    trace = Trace(config.runs_dir / run_id, session.session_id, run_id=run_id)
    database = trace.run_dir / "business.sqlite"
    backend = RetailBackend(database, ttl_seconds=config.backend.operation_ttl_seconds)
    try:
        backend.initialize(bundle.database)
        gateway = ToolGateway(backend, trace, skills)
        llm = LLMClient(config, trace, FakeModel(responses))
        runner = SubAgentRunner(
            build_registry(skills), gateway, llm, max_steps=config.agent.max_steps
        )
        with trace.span("session", "子 Agent 假模型演示"):
            with trace.span("turn", "第 1 轮", turn=1):
                gateway.call(
                    CallContext(session, "coordinator", uuid4().hex),
                    "find_user_id_by_email",
                    {"email": user["email"]},
                )
                result = await runner.run(
                    session,
                    capability,
                    DispatchTask(
                        task_id="demo-task",
                        instruction=f"准备 {action}：{json.dumps(arguments, ensure_ascii=False)}",
                        is_write=True,
                    ),
                )
        return {"result": result, "database": str(database), "trace": str(trace.path)}
    finally:
        backend.close()


@contextmanager
def business_runtime(
    config_path: Path, database: Path
) -> Iterator[tuple[ToolGateway, SessionContext]]:
    """打开已有业务库，为本次命令创建会话与轨迹，结束时释放数据库连接。

    Args:
        config_path: TOML 运行配置路径，轨迹目录以该配置所在目录为基准解析。
        database: 已初始化的 SQLite 业务库文件路径。

    Yields:
        按顺序包含以下对象的二元组：
            gateway: 连接业务库并记录工具调用轨迹的网关。
            session: 本次命令使用的会话，初始未验证身份。

    Raises:
        FileNotFoundError: 业务库或配置文件不存在。
        ConfigError: 配置字段缺失或取值不符合约定。
    """
    if not database.is_file():
        raise FileNotFoundError(f"业务库不存在，请先初始化：{database}")
    config = load_config(config_path)
    skills = load_skills(config.root)
    backend = RetailBackend(database, ttl_seconds=config.backend.operation_ttl_seconds)
    run_id = f"business-{uuid4().hex}"
    session = SessionContext(uuid4().hex)
    try:
        trace = Trace(config.runs_dir / run_id, session.session_id, run_id=run_id)
        with trace.span("session", "命令行业务操作"):
            with trace.span("turn", "第 1 轮", turn=1):
                yield ToolGateway(backend, trace, skills), session
    finally:
        backend.close()
