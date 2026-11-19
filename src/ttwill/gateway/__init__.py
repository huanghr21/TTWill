"""以会话身份和能力白名单约束业务工具，并记录实际调用结果。"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ttwill.backend import BusinessError, RetailBackend, request_key
from ttwill.backend.arguments import ARGUMENT_MODELS, validate_arguments
from ttwill.gateway.identifiers import normalize_arguments, normalize_order_id
from ttwill.gateway.presentation import operation_content
from ttwill.gateway.skills import SkillCatalog
from ttwill.gateway.verification import verification_query, verify_business_result
from ttwill.tracing import Trace

IDENTITY_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
ORDER_WRITES = frozenset(
    {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }
)
AFTER_SALES_WRITES = frozenset({"exchange_delivered_order_items", "return_delivered_order_items"})
WRITE_TOOLS = ORDER_WRITES | AFTER_SALES_WRITES
READ_TOOLS = frozenset(ARGUMENT_MODELS) - WRITE_TOOLS - IDENTITY_TOOLS
CAPABILITY_TOOLS = {
    "coordinator": IDENTITY_TOOLS | READ_TOOLS,
    "order_service": READ_TOOLS | ORDER_WRITES | {"load_skill"},
    "after_sales": READ_TOOLS | AFTER_SALES_WRITES | {"load_skill"},
}


class LoadSkillArguments(BaseModel):
    """加载当前能力的一项 Skill，读取政策及动作办理要点。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, description="本能力 Skill 目录中的名称")


TOOL_MODELS = {**ARGUMENT_MODELS, "load_skill": LoadSkillArguments}


@dataclass
class SessionContext:
    """由代码保存当前聊天的身份验证结果。

    Attributes:
        session_id: 当前聊天会话的唯一标识。
        user_id: 已验证的用户 ID；None 表示尚未验证身份。
    """

    session_id: str
    user_id: str | None = None


@dataclass(frozen=True)
class CallContext:
    """派出工具调用时附带会话、能力、实例和任务信息。

    Attributes:
        session: 当前聊天的身份上下文。
        capability: 当前调用方的能力名称，决定可调用的工具集合。
        instance_id: 发起调用的实例标识，用于区分工具记录。
        task_id: 关联任务的稳定标识；写操作必须提供，其他调用可为 None。
        revision: 任务修订号，初始为 1，请求内容变化时递增。
    """

    session: SessionContext
    capability: str
    instance_id: str
    task_id: str | None = None
    revision: int = 1


class ToolGateway:
    """提供模型业务工具，以及代码专用的提交、拒绝和核验入口。

    Attributes:
        backend: 执行业务查询与准备、提交、拒绝操作的后端。
        trace: 当前会话的轨迹记录器。
        skills: 按能力组织政策和动作说明的 Skill 目录。
        records: 按调用顺序保存的事实记录，包含上下文、参数及执行结果或错误原因。
    """

    def __init__(self, backend: RetailBackend, trace: Trace, skills: SkillCatalog) -> None:
        """连接业务后端、会话轨迹与 Skill 目录，并检查写工具的政策覆盖。

        Args:
            backend: 本次运行使用的业务后端实例。
            trace: 用于记录工具调用和护栏检查的会话轨迹。
            skills: 含全部写工具对应政策的 Skill 目录。

        Raises:
            ValueError: 写工具缺少 Skill，或 Skill 声明的工具超出其能力权限。
        """
        self.backend = backend
        self.trace = trace
        self.skills = skills
        self.records: list[dict[str, Any]] = []
        self._loaded: dict[tuple[str, str | None, str, str], set[str]] = {}
        self._validate_skills()

    @property
    def database_path(self) -> Path:
        """返回当前业务库的绝对路径，供运行时绑定与核对检查点范围。

        Returns:
            解析符号链接后的 SQLite 文件路径；该属性不允许重新绑定业务库。
        """
        return Path(self.backend.engine.url.database).resolve()

    def _validate_skills(self) -> None:
        """启动时核对能力与写工具的 Skill 声明，读取元数据不解锁实例。"""
        covered = set()
        for capability, allowed in CAPABILITY_TOOLS.items():
            for name in self.skills.names(capability):
                skill = self.skills.load(capability, name)
                tools = set(skill["tools"])
                if not tools or not tools <= allowed & WRITE_TOOLS:
                    raise ValueError(f"Skill {name} 的关联写工具不符合能力 {capability}")
                for tool in tools:
                    if self.skills.required_skill(tool) != name:
                        raise ValueError(f"写工具 {tool} 的 Skill 声明不一致")
                covered.update(tools)
        if missing := WRITE_TOOLS - covered:
            raise ValueError(f"写工具缺少 Skill：{', '.join(sorted(missing))}")

    @staticmethod
    def _instance_key(context: CallContext) -> tuple[str, str | None, str, str]:
        """用代码提供的会话、用户、能力和实例标识定位加载记录。"""
        return (
            context.session.session_id,
            context.session.user_id,
            context.capability,
            context.instance_id,
        )

    def loaded_skills(self, context: CallContext) -> frozenset[str]:
        """读取当前实例已经成功加载的 Skill 名称。

        Args:
            context: 指定会话身份、能力和实例的代码上下文。

        Returns:
            已加载名称的不可变集合；尚未加载或实例已释放时为空集合。
        """
        return frozenset(self._loaded.get(self._instance_key(context), ()))

    def release_instance(self, context: CallContext) -> None:
        """清除已结束实例的 Skill 加载记录。

        Args:
            context: 即将销毁的实例使用的会话身份和能力上下文。
        """
        self._loaded.pop(self._instance_key(context), None)

    def schemas(self, context: CallContext) -> list[dict[str, Any]]:
        """构造当前身份和能力可用的 Chat Completions 工具定义。

        Args:
            context: 当前调用的会话身份和能力上下文。

        Returns:
            可直接传入统一模型调用层的工具定义列表，每项包含工具名称、说明
            和参数 JSON Schema；没有可用工具时返回空列表。
        """
        allowed = CAPABILITY_TOOLS.get(context.capability, frozenset())
        if context.session.user_id is None:
            allowed = allowed & IDENTITY_TOOLS
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": model.__doc__.strip(),
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in TOOL_MODELS.items()
            if name in allowed
        ]

    def _guard(self, allowed: bool, name: str, reason: str) -> None:
        """记录一次访问判定，并拦截未通过的调用。

        Args:
            allowed: 本次检查是否允许调用继续执行。
            name: 护栏名称，同时作为拦截时的业务错误码。
            reason: 拦截时写入轨迹和异常的原因。

        Raises:
            BusinessError: allowed 为 False。
        """
        with self.trace.span("guard", name) as span:
            span.attrs["allowed"] = allowed
            if not allowed:
                span.status = "blocked"
                span.attrs["reason"] = reason
        if not allowed:
            raise BusinessError(reason, code=name)

    def _identity(self, context: CallContext) -> str:
        """检查当前会话已验证身份，并返回其用户 ID。"""
        self._guard(context.session.user_id is not None, "identity_required", "请先验证用户身份")
        return context.session.user_id

    def normalize_order_id(self, context: CallContext, value: str) -> str:
        """在任务入口规范订单号，并核对当前已验证用户的权威订单列表。

        Args:
            context: 提供代码保存的已验证用户身份。
            value: 模型给出的订单号。

        Returns:
            通过格式及归属校验的规范订单号。

        Raises:
            BusinessError: 未验证身份、格式错误或订单不属于当前用户；标识符
                错误包含当前用户的合法订单列表。
        """
        user_id = self._identity(context)
        user = self.backend.query(user_id, "get_user_details", {"user_id": user_id})
        return normalize_order_id(value, user["orders"])

    def _attrs(self, context: CallContext, name: str, args: dict, kind: str) -> dict:
        """构造调用记录和工具轨迹共用的上下文信息。

        Args:
            context: 本次调用的会话、能力、实例和任务上下文。
            name: 工具名称或代码操作名称。
            args: 调用参数；写入记录时复制，避免后续修改影响轨迹。
            kind: 调用类别，为 call、commit、reject、operation（操作对账）、
                verification（核验）、verification_read（核验中的业务重读）或
                presentation_read（确认展示中的原订单读取）。

        Returns:
            包含调用类别、上下文标识、动作名称、参数副本和读写类别的属性字典。
        """
        return {
            "kind": kind,
            "session_id": context.session.session_id,
            "user_id": context.session.user_id,
            "capability": context.capability,
            "instance_id": context.instance_id,
            "task_id": context.task_id,
            "revision": context.revision,
            "action": name,
            "args": deepcopy(args),
            "read_write": "write"
            if name in WRITE_TOOLS or kind in {"commit", "reject"}
            else "read",
        }

    def call(self, context: CallContext, name: str, args: dict) -> Any:
        """执行一次模型工具调用，并记录参数与实际结果。

        身份查找成功后绑定本会话用户。写工具只准备操作，同一任务修订与动作
        跨实例沿用请求标识。

        Args:
            context: 代码构造的调用上下文；写操作须包含任务 ID 和有效修订号。
            name: 当前能力允许使用的业务工具名称，或 load_skill。
            args: 工具的参数字典，字段与类型须符合对应参数模型。

        Returns:
            身份工具返回用户 ID；读工具返回实体或目录字典；写工具返回包含
            operation_id、status、preview 等字段的操作记录，重复请求沿用已有记录；
            load_skill 返回包含名称、关联工具、政策章节和正文的 Skill 内容。

        Raises:
            BusinessError: 工具名称、能力、身份或参数检查失败，缺少有效任务关联，
                关联 Skill 尚未加载、Skill 加载失败，或后端业务规则不满足。
        """
        attrs = self._attrs(context, name, args, "call")
        with self.trace.span("tool", name, attrs=attrs) as span:
            try:
                self._guard(name in TOOL_MODELS, "unknown_tool", "未知业务工具")
                self._guard(
                    name in CAPABILITY_TOOLS.get(context.capability, ()),
                    "capability_denied",
                    "当前能力不能调用该工具",
                )
                if name not in IDENTITY_TOOLS:
                    self._identity(context)
                try:
                    arguments = (
                        LoadSkillArguments.model_validate(args).model_dump()
                        if name == "load_skill"
                        else validate_arguments(name, args)
                    )
                except ValueError as exc:
                    self._guard(False, "invalid_arguments", str(exc))
                if name == "load_skill":
                    skill_name = arguments["name"]
                    self._guard(
                        skill_name in self.skills.names(context.capability),
                        "skill_denied",
                        "当前能力不能加载该 Skill",
                    )
                    try:
                        result = self.skills.load(context.capability, skill_name)
                    except ValueError as exc:
                        self._guard(False, "skill_unavailable", str(exc))
                    self._loaded.setdefault(self._instance_key(context), set()).add(skill_name)
                elif name in IDENTITY_TOOLS:
                    result = getattr(self.backend, name)(**arguments)
                    self._guard(
                        context.session.user_id in (None, result),
                        "identity_locked",
                        "当前会话不能切换用户",
                    )
                    context.session.user_id = result
                    span.attrs["user_id"] = result
                elif name in WRITE_TOOLS:
                    required = self.skills.required_skill(name)
                    self._guard(
                        required in self.loaded_skills(context),
                        "skill_required",
                        f"当前实例须先加载 Skill：{required}",
                    )
                    self._guard(bool(context.task_id), "task_required", "准备写操作需要任务 ID")
                    key = request_key(context.task_id, context.revision, name)
                    span.attrs["request_key"] = key
                    arguments = normalize_arguments(
                        self.backend, context.session.user_id, arguments, request_key=key
                    )
                    span.attrs["args"] = deepcopy(arguments)
                    result = self.backend.prepare(context.session.user_id, name, arguments, key)
                else:
                    arguments = normalize_arguments(
                        self.backend, context.session.user_id, arguments
                    )
                    span.attrs["args"] = deepcopy(arguments)
                    result = self.backend.query(context.session.user_id, name, arguments)
            except BusinessError as exc:
                span.attrs.update(status="failed", error_code=exc.code, reason=str(exc))
                self.records.append(deepcopy(span.attrs))
                raise
            span.attrs["result"] = deepcopy(result)
            span.attrs["status"] = result["status"] if name in WRITE_TOOLS else "executed"
            if name in WRITE_TOOLS:
                span.attrs["operation_id"] = result["operation_id"]
            self.records.append(deepcopy(span.attrs))
            return result

    def commit(self, context: CallContext, operation_id: str, expected_args: dict) -> dict:
        """由确认代码提交操作，核对身份及展示参数并记录执行结果。

        此方法不注册为模型工具；确认护栏负责检查真实用户授权和展示有效期。

        Args:
            context: 包含已验证会话身份的调用上下文。
            operation_id: 待提交的后端操作 ID。
            expected_args: 已展示给用户的准备参数，须与后端保存的 args 完全一致。

        Returns:
            状态为 committed 的操作记录，result 为提交后的业务实体；重复提交
            沿用原记录和结果。

        Raises:
            BusinessError: 身份未验证、操作不存在或归属不符、展示参数不一致、
                操作已失效，或提交时的业务复核失败。
        """
        return self._resolve(context, "commit", operation_id, expected_args)

    def reject(self, context: CallContext, operation_id: str, reason: str) -> dict:
        """由代码拒绝或作废操作，记录结果并释放后端占用。

        Args:
            context: 包含已验证会话身份的调用上下文。
            operation_id: 待拒绝或作废的后端操作 ID。
            reason: 拒绝或作废原因，须包含非空白字符。

        Returns:
            后端的当前操作记录；待提交操作变为 rejected，已失效操作保持原状态。

        Raises:
            ValueError: reason 为空或仅含空白。
            BusinessError: 身份未验证、操作不存在、不属于当前用户，或已经提交。
        """
        return self._resolve(context, "reject", operation_id, {"reason": reason})

    def get_operation(self, context: CallContext, operation_id: str) -> dict:
        """由运行代码读取操作当前状态，记录到期等后端变更以供对账。

        此方法不注册为模型工具。后端会将超过有效期的待提交操作更新为 expired。

        Args:
            context: 包含已验证会话身份的调用上下文。
            operation_id: 需要核对当前状态的后端操作 ID。

        Returns:
            当前完整操作记录，包含状态、原参数、预览、提交结果及过期时间。

        Raises:
            BusinessError: 身份尚未验证、操作不存在，或操作不属于当前用户。
        """
        return self._resolve(context, "operation", operation_id, {})

    def operation_content(self, context: CallContext, operation: dict) -> str:
        """核对准备记录并读取原始商品，生成只包含本次改动的确认正文。

        此方法由回复代码调用，不注册为模型工具；完整后端参数与预览保持原状。

        Args:
            context: 含已验证会话身份的代码调用上下文。
            operation: 需要展示的准备记录，使用 operation_id 从后端核对归属和状态。

        Returns:
            用户可确认的中文正文；修改商品时按请求逐项展示原商品与新规格。

        Raises:
            BusinessError: 身份未验证、操作归属不符、操作不再待确认，或原单查询失败。
            ValueError: 原订单条目与准备时的商品替换记录不一致。
        """
        current = self.get_operation(context, operation["operation_id"])
        if current["status"] != "prepared":
            raise BusinessError("只能展示待确认操作", code="operation_not_prepared")
        original_items = None
        if current["action"] == "modify_pending_order_items":
            args = {"order_id": current["args"]["order_id"]}
            attrs = self._attrs(context, "get_order_details", args, "presentation_read")
            with self.trace.span("tool", "get_order_details", attrs=attrs) as span:
                try:
                    order = self.backend.query(context.session.user_id, "get_order_details", args)
                except BusinessError as exc:
                    span.attrs.update(status="failed", error_code=exc.code, reason=str(exc))
                    self.records.append(deepcopy(span.attrs))
                    raise
                span.attrs.update(status="executed", result=deepcopy(order))
                self.records.append(deepcopy(span.attrs))
            original_items = order["items"]
        return operation_content(current, original_items=original_items)

    def find_operation(self, context: CallContext, action: str) -> dict | None:
        """由恢复代码重算请求标识，找回尚未写入检查点的后端操作。

        此方法不注册为模型工具；请求标识只取稳定任务 ID、修订号与动作，
        不包含会话或实例 ID。查询成功不代表用户授权提交。

        Args:
            context: 已验证身份的代码上下文，包含恢复的任务 ID 和修订号。
            action: 当前任务的写动作名称。

        Returns:
            当前请求的完整操作记录；后端尚无该请求时返回 None。

        Raises:
            BusinessError: 身份未验证、请求关联字段无效，或已有操作属于其他用户。
        """
        attrs = self._attrs(context, action, {}, "operation")
        with self.trace.span("tool", "find_operation", attrs=attrs) as span:
            try:
                user_id = self._identity(context)
                key = request_key(context.task_id, context.revision, action)
                span.attrs["request_key"] = key
                try:
                    result = self.backend.get_operation(user_id, request_key=key)
                except BusinessError as exc:
                    if exc.code != "not_found":
                        raise
                    span.attrs["status"] = "not_found"
                    self.records.append(deepcopy(span.attrs))
                    return None
            except BusinessError as exc:
                span.attrs.update(status="failed", error_code=exc.code, reason=str(exc))
                self.records.append(deepcopy(span.attrs))
                raise
            span.attrs.update(
                operation_id=result["operation_id"],
                result=deepcopy(result),
                status=result["status"],
            )
            self.records.append(deepcopy(span.attrs))
            return result

    def verify_operation(self, context: CallContext, operation_id: str) -> dict:
        """由代码重读已提交操作的业务实体，核验后返回可信结果。

        此方法不注册为模型工具。核验只读取业务数据，不修改任务状态；调用方
        只有收到 verified 为 True 的结果才可将写任务标为完成。

        Args:
            context: 包含已验证会话身份的代码调用上下文。
            operation_id: 需要在提交后立即核验的后端操作 ID。

        Returns:
            含 operation_id、action、args、verified、expected、actual、reason
            和 checked_fields 的核验结果；不一致时 verified 为 False。金额检查
            基于订单商品、收付款明细与换货差额，不包括账户余额。

        Raises:
            BusinessError: 身份未验证、操作归属不符或不存在、尚未提交、动作
                没有核验规则，或重读业务实体失败。
        """
        attrs = self._attrs(
            context, "verify_operation", {"operation_id": operation_id}, "verification"
        )
        attrs["operation_id"] = operation_id
        with self.trace.span("verification", "verify_operation", attrs=attrs) as span:
            try:
                operation = self.get_operation(context, operation_id)
                if operation["status"] != "committed":
                    raise BusinessError("只能核验已提交操作", code="operation_not_committed")
                query, query_args = verification_query(operation)
                actual = self._read_for_verification(context, query, query_args)
                result = verify_business_result(operation, actual)
            except BusinessError as exc:
                with self.trace.span("guard", "verification_allowed") as guard:
                    guard.status = "blocked"
                    guard.attrs.update(allowed=False, reason=str(exc), error_code=exc.code)
                span.attrs.update(status="failed", error_code=exc.code, reason=str(exc))
                self.records.append(deepcopy(span.attrs))
                raise
            with self.trace.span("guard", "verification_matches") as guard:
                guard.attrs.update(allowed=result["verified"], reason=result["reason"])
                if not result["verified"]:
                    guard.status = "blocked"
                    span.status = "blocked"
            span.attrs.update(
                action=operation["action"],
                result=deepcopy(result),
                status="verified" if result["verified"] else "verification_failed",
                reason=result["reason"],
            )
            self.records.append(deepcopy(span.attrs))
            return result

    def _read_for_verification(self, context: CallContext, name: str, args: dict) -> dict:
        """执行代码选择的业务查询，记录核验真正读取的实体。

        Args:
            context: 已通过核验入口身份检查的代码调用上下文。
            name: 核验规则选择的订单或用户查询名称。
            args: 由已校验操作记录构造的查询参数。

        Returns:
            当前业务库中的订单或用户实体。

        Raises:
            BusinessError: 业务实体不存在或不属于当前用户。
        """
        attrs = self._attrs(context, name, args, "verification_read")
        with self.trace.span("tool", name, attrs=attrs) as span:
            try:
                result = self.backend.query(context.session.user_id, name, args)
            except BusinessError as exc:
                span.attrs.update(status="failed", error_code=exc.code, reason=str(exc))
                self.records.append(deepcopy(span.attrs))
                raise
            span.attrs.update(status="executed", result=deepcopy(result))
            self.records.append(deepcopy(span.attrs))
            return result

    def _resolve(self, context: CallContext, kind: str, operation_id: str, args: dict) -> dict:
        """记录代码提交、拒绝或对账操作的身份检查、参数与执行结果。

        Args:
            context: 本次代码操作的调用上下文。
            kind: commit 表示提交，reject 表示拒绝，operation 表示读取当前状态。
            operation_id: 待处理的后端操作 ID。
            args: 提交时为已展示的准备参数；拒绝时为包含 reason 的字典；
                对账时为空字典。

        Returns:
            后端操作记录，包含动作名称、请求标识、当前状态和业务结果。

        Raises:
            ValueError: 拒绝原因为空或仅含空白。
            BusinessError: 身份或归属检查失败，或后端拒绝提交、拒绝操作。
        """
        attrs = self._attrs(context, kind, args, kind)
        attrs["operation_id"] = operation_id
        with self.trace.span("tool", kind, attrs=attrs) as span:
            try:
                user_id = self._identity(context)
                if kind == "commit":
                    result = self.backend.commit(user_id, operation_id, args)
                elif kind == "operation":
                    result = self.backend.get_operation(user_id, operation_id=operation_id)
                else:
                    result = self.backend.reject(user_id, operation_id, args["reason"])
            except BusinessError as exc:
                span.attrs.update(status="failed", error_code=exc.code, reason=str(exc))
                self.records.append(deepcopy(span.attrs))
                raise
            span.attrs.update(
                action=result["action"],
                result=deepcopy(result),
                status=result["status"],
                request_key=result["request_key"],
            )
            self.records.append(deepcopy(span.attrs))
            return result
