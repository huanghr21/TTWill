"""验证写工具的政策前置条件、实例隔离和真实加载事实。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ttwill.backend import BusinessError, RetailBackend
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace


@pytest.fixture
def gateway(backend: RetailBackend, tmp_path: Path, skill_catalog: SkillCatalog) -> ToolGateway:
    """为政策门控测试连接独立的业务库、轨迹与 Skill 目录。

    Args:
        backend: 当前测试使用的小型 SQLite 业务库。
        tmp_path: 保存调用轨迹的临时目录。
        skill_catalog: 覆盖七项写操作的项目 Skill 目录。

    Returns:
        尚未给任何实例加载 Skill 的网关。
    """
    return ToolGateway(backend, Trace(tmp_path / "run", "chat"), skill_catalog)


@pytest.fixture
def context() -> CallContext:
    """提供由代码构造、已验证 u1 身份的订单服务实例。"""
    return CallContext(SessionContext("chat", "u1"), "order_service", "instance-1", "task-1")


def load_for(gateway: ToolGateway, context: CallContext, tool: str) -> dict:
    """通过公开模型工具为某项写操作读取政策。

    Args:
        gateway: 记录本次政策加载事实的网关。
        context: 当前调用的身份、能力与实例上下文。
        tool: 本次准备操作所需的写工具名称。

    Returns:
        网关返回的 Skill 元数据和政策正文。
    """
    return gateway.call(context, "load_skill", {"name": gateway.skills.required_skill(tool)})


@pytest.mark.parametrize(
    ("capability", "tool", "args"),
    [
        (
            "order_service",
            "cancel_pending_order",
            {"order_id": "#1", "reason": "no longer needed"},
        ),
        (
            "order_service",
            "modify_pending_order_address",
            {
                "order_id": "#1",
                "address1": "新地址",
                "address2": "",
                "city": "北京",
                "state": "北京",
                "country": "中国",
                "zip": "100000",
            },
        ),
        (
            "order_service",
            "modify_user_address",
            {
                "user_id": "u1",
                "address1": "新地址",
                "address2": "",
                "city": "北京",
                "state": "北京",
                "country": "中国",
                "zip": "100000",
            },
        ),
        (
            "order_service",
            "modify_pending_order_items",
            {
                "order_id": "#1",
                "item_ids": ["a"],
                "new_item_ids": ["b"],
                "payment_method_id": "card",
            },
        ),
        (
            "order_service",
            "modify_pending_order_payment",
            {"order_id": "#1", "payment_method_id": "card"},
        ),
        (
            "after_sales",
            "return_delivered_order_items",
            {"order_id": "#2", "item_ids": ["a"], "payment_method_id": "card"},
        ),
        (
            "after_sales",
            "exchange_delivered_order_items",
            {
                "order_id": "#2",
                "item_ids": ["a"],
                "new_item_ids": ["b"],
                "payment_method_id": "card",
            },
        ),
    ],
)
def test_every_write_requires_its_skill(
    gateway: ToolGateway,
    context: CallContext,
    backend: RetailBackend,
    capability: str,
    tool: str,
    args: dict,
) -> None:
    """七项写工具在各自政策加载前全部拦截，并记录明确的护栏原因。"""
    context = replace(context, capability=capability)
    before = backend.export_snapshot()
    with pytest.raises(BusinessError) as error:
        gateway.call(context, tool, args)
    assert error.value.code == "skill_required"
    assert backend.export_snapshot() == before
    assert not gateway.loaded_skills(context)
    assert gateway.records[-1]["error_code"] == "skill_required"
    events = [json.loads(line) for line in gateway.trace.path.read_text().splitlines()]
    assert any(
        event["kind"] == "guard"
        and event["name"] == "skill_required"
        and event["status"] == "blocked"
        for event in events
    )


def test_loading_records_actual_text_and_preserves_tool_allowlist(
    gateway: ToolGateway, context: CallContext, backend: RetailBackend
) -> None:
    """成功加载保留实际政策内容，只解锁本实例，业务工具清单始终相同。"""
    before = gateway.schemas(context)
    assert gateway.loaded_skills(context) == frozenset()
    skill = load_for(gateway, context, "cancel_pending_order")
    assert skill["content"]
    assert gateway.loaded_skills(context) == frozenset({skill["name"]})
    assert gateway.schemas(context) == before
    record = gateway.records[-1]
    assert record["action"] == "load_skill"
    assert record["read_write"] == "read"
    assert record["status"] == "executed"
    assert record["result"] == skill
    skill["content"] = "调用方篡改"
    assert record["result"]["content"] != "调用方篡改"
    operation = gateway.call(
        context, "cancel_pending_order", {"order_id": "#1", "reason": "no longer needed"}
    )
    assert operation["status"] == "prepared"
    assert backend.query("u1", "get_order_details", {"order_id": "#1"})["status"] == "pending"


@pytest.mark.parametrize("dimension", ["session", "user", "capability", "instance"])
def test_loading_is_scoped_to_complete_instance_identity(
    gateway: ToolGateway, context: CallContext, dimension: str
) -> None:
    """会话、用户、能力或实例任一改变，都不能沿用其他实例的加载记录。"""
    skill = load_for(gateway, context, "cancel_pending_order")
    alternative = {
        "session": replace(context, session=SessionContext("other-chat", "u1")),
        "user": replace(context, session=SessionContext("chat", "u2")),
        "capability": replace(context, capability="after_sales"),
        "instance": replace(context, instance_id="instance-2"),
    }[dimension]
    assert gateway.loaded_skills(context) == frozenset({skill["name"]})
    assert gateway.loaded_skills(alternative) == frozenset()
    tool = "return_delivered_order_items" if dimension == "capability" else "cancel_pending_order"
    args = (
        {"order_id": "#2", "item_ids": ["a"], "payment_method_id": "card"}
        if dimension == "capability"
        else {"order_id": "#1", "reason": "no longer needed"}
    )
    with pytest.raises(BusinessError) as error:
        gateway.call(alternative, tool, args)
    assert error.value.code == "skill_required"


def test_releasing_instance_clears_only_its_loading_record(
    gateway: ToolGateway, context: CallContext
) -> None:
    """销毁实例清除其解锁状态，其他实例和已有调用事实保持可查。"""
    other = replace(context, instance_id="instance-2")
    load_for(gateway, context, "cancel_pending_order")
    skill = load_for(gateway, other, "cancel_pending_order")
    records = list(gateway.records)
    gateway.release_instance(context)
    assert gateway.loaded_skills(context) == frozenset()
    assert gateway.loaded_skills(other) == frozenset({skill["name"]})
    assert gateway.records == records
    with pytest.raises(BusinessError) as error:
        gateway.call(
            context, "cancel_pending_order", {"order_id": "#1", "reason": "no longer needed"}
        )
    assert error.value.code == "skill_required"


@pytest.mark.parametrize(
    ("capability", "user_id", "name", "code"),
    [
        ("coordinator", "u1", "cancel_pending_order", "capability_denied"),
        ("order_service", None, "cancel_pending_order", "identity_required"),
        ("order_service", "u1", "return_delivered_order_items", "skill_denied"),
        ("after_sales", "u1", "cancel_pending_order", "skill_denied"),
        ("order_service", "u1", None, "skill_denied"),
    ],
)
def test_invalid_load_requests_do_not_unlock(
    gateway: ToolGateway,
    capability: str,
    user_id: str | None,
    name: str | None,
    code: str,
) -> None:
    """未验证身份、越权或未知的 Skill 加载都不产生解锁记录。"""
    context = CallContext(SessionContext("chat", user_id), capability, "instance-1", "task-1")
    skill = gateway.skills.required_skill(name) if name else "不存在的 Skill"
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "load_skill", {"name": skill})
    assert error.value.code == code
    assert not gateway.loaded_skills(context)
    assert gateway.records[-1]["status"] == "failed"


@pytest.mark.parametrize("extra", [{"loaded": True}, {"user_id": "u1"}, {"instance_id": "other"}])
def test_model_cannot_forge_loading_or_context_fields(
    gateway: ToolGateway, context: CallContext, extra: dict
) -> None:
    """模型在加载参数中伪造权限或调用上下文字段时，参数检查直接拒绝。"""
    name = gateway.skills.required_skill("cancel_pending_order")
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "load_skill", {"name": name, **extra})
    assert error.value.code == "invalid_arguments"
    assert not gateway.loaded_skills(context)
    with pytest.raises(BusinessError) as blocked:
        gateway.call(
            context, "cancel_pending_order", {"order_id": "#1", "reason": "no longer needed"}
        )
    assert blocked.value.code == "skill_required"


def test_failed_catalog_load_does_not_unlock(
    gateway: ToolGateway, context: CallContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """政策内容加载失败时记录实际失败，且不留下实例解锁状态。"""

    def unavailable(capability: str, name: str) -> dict:
        """模拟已登记政策正文不可读取的加载错误。"""
        raise ValueError("政策正文不可用")

    monkeypatch.setattr(gateway.skills, "load", unavailable)
    with pytest.raises(BusinessError) as error:
        load_for(gateway, context, "cancel_pending_order")
    assert error.value.code == "skill_unavailable"
    assert not gateway.loaded_skills(context)
    assert gateway.records[-1]["error_code"] == "skill_unavailable"


def test_loading_unrelated_policy_does_not_unlock_cancellation(
    gateway: ToolGateway, context: CallContext
) -> None:
    """同一能力内读过其他动作政策，也不能越过当前动作的政策前置条件。"""
    load_for(gateway, context, "modify_user_address")
    with pytest.raises(BusinessError) as error:
        gateway.call(
            context, "cancel_pending_order", {"order_id": "#1", "reason": "no longer needed"}
        )
    assert error.value.code == "skill_required"


def test_gateway_startup_rejects_missing_write_policy(
    backend: RetailBackend,
    skill_catalog: SkillCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目录遗漏任一写工具的政策时，启动失败并指出缺少的工具。"""
    names = skill_catalog.names
    missing = skill_catalog.required_skill("cancel_pending_order")
    monkeypatch.setattr(
        skill_catalog,
        "names",
        lambda capability: tuple(n for n in names(capability) if n != missing),
    )
    with pytest.raises(ValueError, match="cancel_pending_order"):
        ToolGateway(backend, Trace(tmp_path / "run", "chat"), skill_catalog)


def test_gateway_startup_rejects_skill_with_foreign_capability_tool(
    backend: RetailBackend,
    skill_catalog: SkillCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """订单服务 Skill 声明售后写工具时，启动阶段拒绝能力权限冲突。"""
    original_load = skill_catalog.load

    def mismatched(capability: str, name: str) -> dict:
        """返回声明了越权工具的目录条目。"""
        result = original_load(capability, name)
        if capability == "order_service":
            result["tools"] = ["return_delivered_order_items"]
        return result

    monkeypatch.setattr(skill_catalog, "load", mismatched)
    with pytest.raises(ValueError, match="关联写工具不符合能力"):
        ToolGateway(backend, Trace(tmp_path / "run", "chat"), skill_catalog)
