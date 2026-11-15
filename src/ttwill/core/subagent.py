"""用有上限的模型工具循环执行单个任务，并由代码汇总可信结果。"""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from ttwill.core.capabilities import Capability, CapabilityRegistry
from ttwill.gateway import (
    READ_TOOLS,
    WRITE_TOOLS,
    BusinessError,
    CallContext,
    SessionContext,
    ToolGateway,
)
from ttwill.llm.client import LLMClient
from ttwill.llm.types import LLMError


@dataclass(frozen=True)
class DispatchTask:
    """保存协调代码交给一个实例的任务说明。

    Attributes:
        task_id: 稳定任务标识，跨实例重试时沿用。
        instruction: 本次子任务的具体目标与要求。
        revision: 请求修订号；任务内容改变时由协调代码递增。
        is_write: 是否需要准备业务写操作，影响结果状态的判定。
        facts: 由协调代码确认的已有事实；不包含旧聊天记录。
    """

    task_id: str
    instruction: str
    revision: int = 1
    is_write: bool = False
    facts: list[dict[str, Any]] = field(default_factory=list)


class FinishPayload(BaseModel):
    """限制模型只报告信息缺口和回复要点。

    Attributes:
        missing: 继续办理所缺少的信息，空列表表示模型未发现缺口。
        message_hint: 供协调者组织回复的要点，不作为业务执行证据。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    missing: list[str]
    message_hint: str


FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "结束当前任务，只提供缺少的信息和回复要点；须单独调用。",
        "parameters": FinishPayload.model_json_schema(),
    },
}


class SubAgentRunner:
    """为每次任务分派创建独立上下文，执行工具循环并释放实例 Skill。"""

    def __init__(
        self,
        registry: CapabilityRegistry,
        gateway: ToolGateway,
        llm: LLMClient,
        *,
        max_steps: int,
    ) -> None:
        """绑定能力、工具网关、模型与循环预算。

        Args:
            registry: 按能力名称保存的固定定义。
            gateway: 执行业务工具、记录结果并管理 Skill 解锁的网关。
            llm: 所有模型请求共用的调用层。
            max_steps: 每个实例允许的最大步骤数，须为正整数；每项工具调用
                消耗一步，没有工具调用的模型响应也消耗一步。

        Raises:
            ValueError: max_steps 不是正整数。
        """
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("子 Agent 步数上限必须为正整数")
        self.registry = registry
        self.gateway = gateway
        self.llm = llm
        self.max_steps = max_steps

    async def run(
        self,
        session: SessionContext,
        capability_name: str,
        task: DispatchTask,
        *,
        model_tier: str | None = None,
    ) -> dict[str, Any]:
        """执行一次任务分派，使用实际工具记录组装结构化结果。

        新实例只接收本次任务和已确认事实；写工具仅准备操作，正常结束时
        保留待确认操作。超限或异常退出时拒绝本实例的待提交操作，所有退出
        路径都会清除实例 Skill 加载记录。

        Args:
            session: 已由协调代码验证身份的会话上下文。
            capability_name: 注册表中的能力名称。
            task: 任务标识、说明、修订号、读写类型与已确认事实。
            model_tier: 本次使用的模型档位；None 表示使用能力默认档位。

        Returns:
            包含任务与实例标识、status、actions、facts、missing、
            pending_confirmation 和 message_hint 的字典。待确认详情来自
            后端记录；写任务不会在本阶段返回 done。

        Raises:
            ValueError: 能力、模型档位、任务标识或修订号无效，或身份尚未验证。
            LLMError: 模型调用或响应校验失败。
            TimeoutError: 模型请求在重试结束后仍超时。
            asyncio.CancelledError: 当前实例被取消。
            OSError: 轨迹无法写入。
        """
        capability = self.registry.get(capability_name)
        if session.user_id is None:
            raise ValueError("派发子 Agent 前须验证会话身份")
        if not task.task_id.strip() or type(task.revision) is not int or task.revision < 1:
            raise ValueError("任务标识须非空，修订号须为正整数")
        context = CallContext(session, capability.name, uuid4().hex, task.task_id, task.revision)
        attrs = {
            "task_id": task.task_id,
            "revision": task.revision,
            "user_id": session.user_id,
            "capability": capability.name,
            "instance_id": context.instance_id,
        }
        completed = False
        exit_reason = "子 Agent 异常退出，作废尚未提交的操作"
        try:
            with self.gateway.trace.span("dispatch", capability.name, attrs=attrs):
                with self.gateway.trace.span("subagent", task.task_id, attrs=attrs) as span:
                    payload, steps, failure = await self._loop(
                        context,
                        capability,
                        task,
                        capability.model_tier if model_tier is None else model_tier,
                    )
                    if failure:
                        self._cleanup(context, failure)
                    else:
                        self._refresh_operations(context)
                    result = self._result(context, task, payload, steps, failure)
                    span.attrs.update(status=result["status"], steps=steps)
                    with self.gateway.trace.span("result", task.task_id, attrs=result):
                        pass
            completed = True
            return result
        except (LLMError, TimeoutError, asyncio.CancelledError) as exc:
            exit_reason = f"子 Agent 因 {type(exc).__name__} 退出，作废尚未提交的操作"
            self._blocked(context, "subagent_interrupted", exit_reason)
            raise
        finally:
            try:
                if not completed:
                    self._cleanup(context, exit_reason)
            finally:
                self.gateway.release_instance(context)

    def _messages(self, capability: Capability, task: DispatchTask) -> list[dict[str, Any]]:
        """构造新实例的固定政策、Skill 目录和本次任务。

        Args:
            capability: 当前能力的提示词和已声明 Skill 名称。
            task: 本次任务说明与协调代码确认的事实。

        Returns:
            两条 Chat Completions 消息：system 包含能力提示词、通用政策和
            可用 Skill 目录；user 包含任务说明与已有事实，不附带历史对话。
        """
        catalog = [
            entry
            for entry in self.gateway.skills.catalog(capability.name)
            if entry["name"] in capability.skills
        ]
        system = (
            f"{capability.prompt}\n\n{self.gateway.skills.common_policy}\n\n"
            f"可用 Skill 目录：{json.dumps(catalog, ensure_ascii=False)}\n"
            "调用写工具前先加载对应 Skill。写工具只准备操作，等待用户确认后才能提交。"
            "完成查询、准备操作或发现信息缺口后，单独调用 finish，参数仅含 missing"
            "（字符串列表）和 message_hint（回复要点字符串）。也可返回同结构 JSON。"
            "任务状态、动作、业务事实与待确认详情由运行代码填写。"
        )
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {"instruction": task.instruction, "facts": task.facts}, ensure_ascii=False
                ),
            },
        ]

    async def _loop(
        self, context: CallContext, capability: Capability, task: DispatchTask, tier: str
    ) -> tuple[FinishPayload | None, int, str | None]:
        """逐调用消耗预算，向模型反馈可修正的工具或格式错误。

        Args:
            context: 本实例的固定调用身份和任务关联。
            capability: 本次派发使用的能力定义。
            task: 构造初始上下文的任务与已有事实。
            tier: 统一调用层中的模型档位名称。

        Returns:
            按顺序返回合法结束内容、已消耗步数、失败原因；预算耗尽时结束
            内容为 None，正常结束时失败原因为 None。
        """
        messages = self._messages(capability, task)
        tools = [
            schema
            for schema in self.gateway.schemas(context)
            if schema["function"]["name"] in capability.tools
        ] + [deepcopy(FINISH_TOOL)]
        steps = 0
        while steps < self.max_steps:
            response = await self.llm.complete(tier, messages, tools=tools)
            messages.append(deepcopy(response.message))
            calls = response.message.get("tool_calls") or []
            if not calls:
                steps += 1
                try:
                    return (
                        FinishPayload.model_validate_json(response.message.get("content") or ""),
                        steps,
                        None,
                    )
                except ValidationError as exc:
                    reason = f"结束内容须只包含 missing 和 message_hint：{exc}"
                    self._blocked(context, "invalid_finish", reason)
                    messages.append({"role": "user", "content": reason})
                    continue
            for call in calls:
                if steps == self.max_steps:
                    break
                steps += 1
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                    if not isinstance(args, dict):
                        raise ValueError("工具参数必须是 JSON 对象")
                    if name == "finish":
                        if len(calls) != 1:
                            raise ValueError("finish 须单独调用")
                        return FinishPayload.model_validate(args), steps, None
                    if name not in capability.tools:
                        raise ValueError(f"当前能力没有此工具：{name}")
                    if name == "load_skill" and args.get("name") not in capability.skills:
                        raise ValueError("当前能力未声明此 Skill")
                    outcome = self.gateway.call(context, name, args)
                except (ValueError, BusinessError) as exc:
                    if isinstance(exc, BusinessError):
                        outcome = {"error": exc.code, "reason": str(exc)}
                    else:
                        self._blocked(context, "invalid_tool_call", str(exc))
                        outcome = {"error": "invalid_tool_call", "reason": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(outcome, ensure_ascii=False),
                    }
                )
        reason = f"子 Agent 达到步骤上限 {self.max_steps}，尚未返回合法结束内容"
        self._blocked(context, "subagent_step_limit", reason)
        return None, steps, reason

    def _records(self, context: CallContext) -> list[dict[str, Any]]:
        """读取仅属于当前调用身份的网关记录。

        Args:
            context: 用于筛选会话、用户、能力与实例四项身份的调用上下文。

        Returns:
            按原始顺序保存的网关记录引用列表，包含实际参数、结果或错误。
        """
        return [
            record
            for record in self.gateway.records
            if record["session_id"] == context.session.session_id
            and record["instance_id"] == context.instance_id
            and record["user_id"] == context.session.user_id
            and record["capability"] == context.capability
        ]

    def _operations(self, context: CallContext) -> dict[str, dict[str, Any]]:
        """按操作 ID 汇总本实例触及的操作，以最后一次实际结果为准。

        Args:
            context: 用于定位当前实例网关记录的调用上下文。

        Returns:
            以操作 ID 为键、完整后端操作记录为值的字典；包含准备、对账、
            提交或拒绝返回的最新记录。
        """
        operations = {}
        for record in self._records(context):
            if record.get("operation_id") and "result" in record:
                operations[record["operation_id"]] = record["result"]
        return operations

    def _cleanup(self, context: CallContext, reason: str) -> None:
        """对账后拒绝本实例仍待提交的操作，释放订单占用。

        Args:
            context: 定位本实例操作并执行对账与拒绝的调用上下文。
            reason: 写入后端操作记录的失败或退出原因。

        Raises:
            BusinessError: 对账时身份或操作归属不满足要求；拒绝阶段的业务
                异常单独写入 cleanup_failed 护栏，不中断其他操作的清理。
            OSError: 对账或拒绝的轨迹无法写入。
        """
        self._refresh_operations(context)
        for operation_id, operation in self._operations(context).items():
            if operation["status"] == "prepared":
                try:
                    self.gateway.reject(context, operation_id, reason)
                except BusinessError as exc:
                    self._blocked(context, "cleanup_failed", str(exc))

    def _refresh_operations(self, context: CallContext) -> None:
        """重新读取本实例触及的操作，将后端当前状态追加到网关记录。

        Args:
            context: 定位本实例操作并执行代码对账的调用上下文。

        Raises:
            BusinessError: 对账时操作不存在、身份未验证或操作归属不符。
            OSError: 对账轨迹无法写入。
        """
        for operation_id in self._operations(context):
            self.gateway.get_operation(context, operation_id)

    def _blocked(self, context: CallContext, name: str, reason: str) -> None:
        """记录包含实例关联的格式、预算或清理护栏结果。

        Args:
            context: 当前任务与实例标识的来源。
            name: 护栏名称，用于区分格式、预算和退出等事件。
            reason: 本次拒绝或失败的具体原因。
        """
        with self.gateway.trace.span(
            "guard",
            name,
            attrs={
                "allowed": False,
                "reason": reason,
                "task_id": context.task_id,
                "instance_id": context.instance_id,
            },
        ) as span:
            span.status = "blocked"

    def _result(
        self,
        context: CallContext,
        task: DispatchTask,
        payload: FinishPayload | None,
        steps: int,
        failure: str | None,
    ) -> dict[str, Any]:
        """将工具事实与模型的两项允许字段合并，计算本次执行状态。

        Args:
            context: 结果中的固定执行身份，以及网关记录的筛选范围。
            task: 任务标识、修订号和读写类型。
            payload: 经严格校验的信息缺口与回复要点；None 表示循环未正常结束。
            steps: 已实际消耗的模型响应或工具调用步数。
            failure: 循环生成的失败原因；None 表示根据工具记录判断执行状态。

        Returns:
            包含固定身份、状态、写尝试 actions、查询事实 facts、信息缺口
            missing、待确认列表 pending_confirmation、回复要点、失败原因和
            步数的字典。actions 保留实际调用结果；待确认操作采用对账后的
            参数与预览，代码对账读取本身不计作新的写尝试。
        """
        actions = []
        facts = []
        for index, record in enumerate(self._records(context)):
            if record["action"] in WRITE_TOOLS and record["kind"] != "operation":
                actions.append(
                    {
                        "kind": record["kind"],
                        "tool": record["action"],
                        "args": deepcopy(record["args"]),
                        "status": record["status"],
                        **{
                            key: deepcopy(record[key])
                            for key in ("result", "operation_id", "error_code", "reason")
                            if key in record
                        },
                    }
                )
            elif record["action"] in READ_TOOLS and record.get("status") == "executed":
                facts.append(
                    {
                        "tool": record["action"],
                        "args": deepcopy(record["args"]),
                        "result": deepcopy(record["result"]),
                        "source": {
                            "session_id": context.session.session_id,
                            "instance_id": context.instance_id,
                            "record_index": index,
                        },
                    }
                )
        operations = self._operations(context)
        pending = [
            {
                "operation_id": operation["operation_id"],
                "action": operation["action"],
                "args": deepcopy(operation["args"]),
                "preview": deepcopy(operation["preview"]),
                "expires_at": operation["expires_at"],
            }
            for operation in operations.values()
            if operation["status"] == "prepared"
        ]
        missing = payload.missing if payload else []
        if failure:
            status = "failed"
            pending = []
        elif pending:
            status = "need_confirm"
        elif any(operation["status"] == "committed" for operation in operations.values()):
            status = "needs_verification"
        elif missing:
            status = "need_info"
        elif not task.is_write and facts:
            status = "ready"
        else:
            status = "failed"
            if operations:
                states = "、".join(
                    sorted({operation["status"] for operation in operations.values()})
                )
                failure = f"本次准备的操作已失效：{states}"
            else:
                failure = (
                    "写任务未准备操作，也未说明信息缺口"
                    if task.is_write
                    else "未取得业务查询结果，也未说明信息缺口"
                )
            self._blocked(context, "subagent_no_progress", failure)
        return {
            "task_id": task.task_id,
            "revision": task.revision,
            "user_id": context.session.user_id,
            "capability": context.capability,
            "instance_id": context.instance_id,
            "status": status,
            "actions": actions,
            "facts": facts,
            "missing": deepcopy(missing),
            "pending_confirmation": pending,
            "message_hint": payload.message_hint if payload else "",
            "reason": failure,
            "steps": steps,
        }
