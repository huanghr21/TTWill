"""组装场景验收所需的真实组件，保存逐轮原话、结果和完整轨迹。"""

import json
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from ttwill.agents import build_registry
from ttwill.backend import RetailBackend
from ttwill.config import AppConfig
from ttwill.core.confirmation import ConfirmationJudge
from ttwill.core.context import ContextCompiler
from ttwill.core.coordinator import Coordinator
from ttwill.core.runtime import PersistentRuntime
from ttwill.core.subagent import SubAgentRunner
from ttwill.gateway import ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.llm import FakeModel, LLMClient
from ttwill.persistence import Archive
from ttwill.persistence.checkpoints import open_checkpointer
from ttwill.tracing import Trace

USER_ID = "junjie_liu_6635"
PAYMENT = "paypal_6151711"
PENDING = "#W8770097"
ITEMS_ORDER = "#W5183325"
DELIVERED = "#W6893533"


def tool(name: str, args: dict) -> dict:
    """生成一次脚本化工具提议，实际权限与业务规则仍由本体校验。

    Args:
        name: 协调者或子 Agent 的工具名称。
        args: 模型提出的工具参数，不包含代码维护的任务状态。

    Returns:
        含唯一调用标识的完整假模型响应。
    """
    return {
        "model": "fake-demo",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                },
            }
        ],
    }


def text(content: str) -> dict:
    """生成独立确认判定或摘要所用的假模型文本响应。"""
    return {
        "model": "fake-demo",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def add_task(action: str, order_id: str, instruction: str, **fields) -> dict:
    """构造模型新增诉求，不填写任务标识、状态或执行结果。

    Args:
        action: 具体读写动作名称。
        order_id: 目标订单。
        instruction: 交给子 Agent 的自然语言办理说明。
        **fields: condition、missing 等允许由模型提出的附加字段。

    Returns:
        update_tasks 接受的一项新增提议。
    """
    return {
        "operation": "add",
        "capability": "after_sales" if "delivered" in action else "order_service",
        "action": action,
        "order_id": order_id,
        "instruction": instruction,
        **fields,
    }


def task_ids(
    tasks: Callable[[], list[dict]], *, status: str | None = None, action: str | None = None
) -> list[str]:
    """读取代码维护的任务标识，供下一条脚本化模型提议引用。

    Args:
        tasks: 取得当前真实任务快照的回调。
        status: 可选状态筛选；None 表示所有状态。
        action: 可选动作筛选；None 表示所有动作。

    Returns:
        满足条件的任务标识，保持创建顺序。
    """
    return [
        task["id"]
        for task in tasks()
        if (status is None or task["status"] == status)
        and (action is None or task["action"] == action)
    ]


def prepare_steps(action: str, args: dict) -> list[dict]:
    """让子 Agent 查询原单、加载真实 Skill、准备操作并结束派工。

    Args:
        action: 要准备的业务动作。
        args: 完整业务参数，原样交给网关和真实后端。

    Returns:
        四个模型响应；结束响应只含回复要点与信息缺口。
    """
    skill = {
        "return_delivered_order_items": "return_delivered_order",
        "exchange_delivered_order_items": "exchange_delivered_order",
    }.get(action, action)
    return [
        tool("get_order_details", {"order_id": args["order_id"]}),
        tool("load_skill", {"name": skill}),
        tool(action, args),
        tool("finish", {"missing": [], "note": "请核对操作详情后确认。"}),
    ]


def reply(show_task_ids: list[str] | None = None) -> dict:
    """请求按真实任务结果回复，可明确限制本次展示的任务范围。"""
    return tool("reply", {"message": "请查看办理结果。", "show_task_ids": show_task_ids})


class AcceptanceCase:
    """为一个独立业务场景保存运行文件和多段聊天的观察记录。"""

    def __init__(
        self, run_dir: Path, name: str, seed: dict, config: AppConfig, skills: SkillCatalog
    ) -> None:
        """初始化独立业务库，保留默认运行参数与原始开发种子。

        Args:
            run_dir: pytest 分配的本场景独立目录。
            name: 场景标识，同时用于检查点线程与轨迹关联。
            seed: 固定中文业务数据的小型子集，不含参考动作或答案。
            config: 项目默认配置，循环和上下文预算原样使用。
            skills: 真实政策与项目 Skill 目录。
        """
        self.run_dir, self.name, self.seed = run_dir, name, deepcopy(seed)
        self.user = deepcopy(seed["users"][USER_ID])
        self.config, self.skills = replace(config, runs_dir=run_dir), skills
        self.turns: list[dict] = []
        self.sessions: list[dict] = []
        self.database = run_dir / "business.sqlite"
        backend = RetailBackend(self.database, ttl_seconds=config.backend.operation_ttl_seconds)
        try:
            backend.initialize(seed)
        finally:
            backend.close()
        self._save()

    @asynccontextmanager
    async def session(
        self,
        script: Callable,
        children: Iterable[dict],
        *,
        session_id: str = "chat1",
        judge: Callable | Iterable[dict] | None = None,
    ) -> AsyncIterator[PersistentRuntime]:
        """打开一段真实聊天，先验身份，退出时关闭所有连接。

        Args:
            script: 接收 tasks 回调并逐条产出协调模型响应的函数。
            children: 子 Agent 模型响应序列。
            session_id: 本场景中唯一的聊天标识，续办时必须换一个。
            judge: 独立确认模型响应序列，或接收 tasks 回调的生成器函数。
                None 使用本组场景固定原话的预设判断，未登记输入使测试失败。

        Yields:
            已通过邮箱查找验证身份的 PersistentRuntime。
        """
        backend = RetailBackend(
            self.database, ttl_seconds=self.config.backend.operation_ttl_seconds
        )
        trace = Trace(
            self.run_dir,
            session_id,
            run_id=self.name,
            task_id=self.name,
            repeat=1,
            attempt_id="offline-1",
        )

        def tasks() -> list[dict]:
            """取得当前节点的任务副本，只供脚本引用代码生成的标识。"""
            return runtime.state.tasks.snapshot()

        coordinator_script = iter(script(tasks))
        child_script = iter(children)

        def default_judgments():
            """为验收场景的固定原话返回判定，并核对实际回复与候选卡片一致。"""
            expected = {
                ("好的", 2): "unrelated",
                ("全部确认", 2): "confirm",
                ("确认", 1): "confirm",
            }
            consumed = 0
            while len(judge_fake.requests) > consumed:
                data = json.loads(judge_fake.requests[consumed]["messages"][-1]["content"])
                consumed += 1
                key = data["user_message"], len(data["displayed"])
                assert key in expected, f"未配置的确认场景：{key}"
                assert all(
                    shown["content"] in data["assistant_reply"] for shown in data["displayed"]
                )
                yield text(
                    json.dumps(
                        [
                            {
                                "operation_id": shown["operation_id"],
                                "decision": expected[key],
                                "reason": "场景预设的明确授权范围",
                            }
                            for shown in data["displayed"]
                        ],
                        ensure_ascii=False,
                    )
                )

        judge_script = iter(
            default_judgments() if judge is None else judge(tasks) if callable(judge) else judge
        )
        judge_fake = FakeModel(judge_script)
        gateway = ToolGateway(backend, trace, self.skills)
        llm = LLMClient(self.config, trace, FakeModel(coordinator_script))
        runner = SubAgentRunner(
            build_registry(self.skills),
            gateway,
            LLMClient(self.config, trace, FakeModel(child_script)),
            max_steps=self.config.agent.max_steps,
        )
        coordinator = Coordinator(
            gateway,
            runner,
            llm,
            max_rounds=self.config.agent.max_rounds,
            judge=ConfirmationJudge(LLMClient(self.config, trace, judge_fake)),
        )
        try:
            with Archive(self.run_dir / "archive.sqlite") as archive:
                async with open_checkpointer(self.run_dir / "checkpoints.sqlite") as saver:
                    runtime = PersistentRuntime(
                        coordinator,
                        saver,
                        archive,
                        thread_id=self.name,
                        compiler=ContextCompiler(
                            archive,
                            llm,
                            max_chars=self.config.context.max_chars,
                            recent_turns=self.config.context.recent_turns,
                            summary_tier=self.config.context.summary_tier,
                        ),
                    )
                    with trace.span("session", "本体场景验收"):
                        opened = await runtime.open_session(
                            "find_user_id_by_email", {"email": self.user["email"]}
                        )
                        self.sessions.append(deepcopy(opened))
                        self._save()
                        yield runtime
                        for responses in (coordinator_script, child_script, judge_script):
                            assert next(responses, None) is None, "场景结束后仍有未执行的模型脚本"
        finally:
            backend.close()
            self._save()

    async def send(self, runtime: PersistentRuntime, user_message: str) -> dict:
        """将真实原话送入运行时，并保存可对照业务库的逐轮观察记录。

        Args:
            runtime: 当前已验证的聊天运行时。
            user_message: 手写场景中的本轮用户原话。

        Returns:
            运行时真实返回的回复、任务、展示项与确认结果。
        """
        result = await runtime.send_message(user_message)
        state = await runtime.snapshot()
        self.turns.append(
            {
                "session_id": state.session.session_id,
                "turn": state.turn,
                "user_message": user_message,
                "result": deepcopy(result),
            }
        )
        self._save()
        return result

    def snapshot(self) -> dict:
        """重读当前业务终态，不使用模型回复或任务状态推断业务结果。"""
        backend = RetailBackend(self.database)
        try:
            return backend.export_snapshot()
        finally:
            backend.close()

    def operation(self, operation_id: str) -> dict:
        """按当前用户读取真实操作状态及参数。

        Args:
            operation_id: 网关返回的操作标识。

        Returns:
            属于当前用户的后端操作记录。
        """
        backend = RetailBackend(self.database)
        try:
            return backend.get_operation(USER_ID, operation_id=operation_id)
        finally:
            backend.close()

    def _save(self) -> None:
        """覆盖本场景的只读观察报告，不参与运行时恢复或任务推进。"""
        report = {
            "scenario": self.name,
            "mode": "scripted_fake_models",
            "parameters": {
                "agent": asdict(self.config.agent),
                "context": asdict(self.config.context),
            },
            "sessions": self.sessions,
            "turns": self.turns,
        }
        (self.run_dir / "dialogue.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
