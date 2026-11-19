"""用真实 SQLite 回归模型标识符的格式、归属、批量原子性和幂等边界。"""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend
from ttwill.core.tasks import TaskChange, TaskStore
from ttwill.gateway import (
    IDENTITY_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    BusinessError,
    CallContext,
    SessionContext,
    ToolGateway,
)
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace


@pytest.fixture
def boundary(
    tmp_path: Path, seed_snapshot: dict, skill_catalog: SkillCatalog
) -> Iterator[tuple[ToolGateway, CallContext]]:
    """创建包含真实失败订单号及另一未购买商品的独立业务库。

    Args:
        tmp_path: 本项测试的临时目录。
        seed_snapshot: 订单号符合正式格式的小型业务快照。
        skill_catalog: 项目 Skill 目录和离线测试政策。

    Yields:
        已验证用户 u1 的真实网关和协调者上下文，结束后关闭数据库。
    """
    snapshot = deepcopy(seed_snapshot)
    order = snapshot["orders"].pop("#W0000001")
    order["order_id"] = "#W5061109"
    snapshot["orders"][order["order_id"]] = order
    snapshot["users"]["u1"]["orders"][0] = order["order_id"]
    snapshot["products"]["p3"] = {
        "product_id": "p3",
        "name": "书架",
        "variants": {"z": {"item_id": "z", "price": 100.0, "available": True, "options": {}}},
    }
    backend = RetailBackend(tmp_path / "business.sqlite")
    backend.initialize(snapshot)
    gateway = ToolGateway(backend, Trace(tmp_path / "trace", "boundary"), skill_catalog)
    context = CallContext(SessionContext("boundary"), "coordinator", "coordinator")
    gateway.call(context, "find_user_id_by_email", {"email": "u1@example.com"})
    yield gateway, context
    backend.close()


def change(order_id: str = "W5061109", **fields: object) -> TaskChange:
    """构造可按需覆盖字段的订单取消任务提议。

    Args:
        order_id: 模型产出的订单号，默认复现 F2 的无 # 表示。
        **fields: 覆盖任务的允许字段。

    Returns:
        已通过字段结构检查、尚未通过业务标识符校验的提议。
    """
    return TaskChange.model_validate(
        {
            "operation": "add",
            "capability": "order_service",
            "action": "cancel_pending_order",
            "order_id": order_id,
            "instruction": "取消订单",
            **fields,
        }
    )


def child(gateway: ToolGateway, context: CallContext, action: str) -> CallContext:
    """创建带稳定任务标识、已加载指定写 Skill 的子实例上下文。

    Args:
        gateway: 当前测试的真实工具网关。
        context: 已验证的协调者上下文。
        action: 要加载 Skill 的写动作。

    Returns:
        属于相同用户的订单服务子实例上下文。
    """
    result = replace(context, capability="order_service", instance_id="child", task_id="t1")
    gateway.call(result, "load_skill", {"name": action})
    return result


@pytest.mark.parametrize("order_id", ["W5061109", " #w5061109 ", "w5061109", "#W5061109"])
def test_actual_f2_order_normalizes_in_task_and_query_fact(boundary: tuple, order_id: str) -> None:
    """F2 的口述订单号在任务与查询事实中统一，匹配事实时不再误拦。"""
    gateway, context = boundary
    store = TaskStore()
    store.apply([change(order_id, action="get_order_details")], gateway, context)
    gateway.call(context, "get_order_details", {"order_id": order_id})
    task = next(iter(store.tasks.values()))
    fact = gateway.records[-1]
    assert task.order_id == fact["args"]["order_id"] == fact["result"]["order_id"] == "#W5061109"


def test_normalized_same_order_requests_coalesce_before_task_creation(boundary: tuple) -> None:
    """不同合法表示的同单同类诉求在新任务创建前合并为一次请求。"""
    gateway, context = boundary
    store = TaskStore()
    store.apply([change(), change(" #w5061109 ", instruction="不再需要")], gateway, context)
    assert len(store.tasks) == 1
    task = next(iter(store.tasks.values()))
    assert task.order_id == "#W5061109" and task.revision == 1
    assert task.instruction == "取消订单\n不再需要"


@pytest.mark.parametrize(
    "order_id",
    ["#1", "5061109", "#W506110", "#W50611090", "Ｗ5061109", "#W５０６１１０９", "W 5061109"],
)
def test_order_format_does_not_guess_or_accept_partial_ids(boundary: tuple, order_id: str) -> None:
    """只执行文档允许的三种格式转换，不按相似度、前后缀或数字猜订单。"""
    gateway, context = boundary
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "get_order_details", {"order_id": order_id})
    assert error.value.code == "invalid_identifier"
    assert "#W5061109" in str(error.value)
    assert gateway.records[-1]["status"] == "failed"


@pytest.mark.parametrize("order_id", ["#W0000003", "#W9999999"])
def test_order_whitelist_uses_verified_user_and_returns_only_own_choices(
    boundary: tuple, order_id: str
) -> None:
    """他人和虚构订单均只返回当前用户的合法列表，不泄露他人订单信息。"""
    gateway, context = boundary
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "get_order_details", {"order_id": order_id})
    assert error.value.code == "invalid_identifier"
    assert "#W5061109" in str(error.value) and "#W0000002" in str(error.value)
    assert "#W0000003" not in str(error.value)


def test_task_batch_validates_all_ids_before_rejecting_old_operation(boundary: tuple) -> None:
    """后项非法订单使整批失败，前项不增加修订、不发布且不拒绝原准备操作。"""
    gateway, context = boundary
    store = TaskStore()
    store.apply([change()], gateway, context)
    task = next(iter(store.tasks.values()))
    prepared_context = replace(child(gateway, context, task.action), task_id=task.id)
    operation = gateway.call(
        prepared_context, task.action, {"order_id": task.order_id, "reason": "no longer needed"}
    )
    task.operation_ids = [operation["operation_id"]]
    before = store.snapshot()
    with pytest.raises(BusinessError, match="合法值"):
        store.apply(
            [
                change(operation="update", task_id=task.id, instruction="修改请求"),
                change("#W9999999"),
            ],
            gateway,
            context,
        )
    assert store.snapshot() == before
    assert gateway.get_operation(context, operation["operation_id"])["status"] == "prepared"
    assert not any(record["kind"] == "reject" for record in gateway.records)


def test_coordinator_read_access_still_requires_identity_and_excludes_writes(
    boundary: tuple,
) -> None:
    """协调者验证后可直接查询全部业务读工具，但不能准备或提交写操作。"""
    gateway, context = boundary
    names = {schema["function"]["name"] for schema in gateway.schemas(context)}
    assert READ_TOOLS | IDENTITY_TOOLS == names
    assert not names & WRITE_TOOLS and "commit" not in names
    guest = replace(context, session=SessionContext("guest"))
    assert {schema["function"]["name"] for schema in gateway.schemas(guest)} == IDENTITY_TOOLS
    with pytest.raises(BusinessError) as error:
        gateway.call(guest, "get_order_details", {"order_id": "W5061109"})
    assert error.value.code == "identity_required"
    with pytest.raises(BusinessError) as denied:
        gateway.call(
            context, "cancel_pending_order", {"order_id": "W5061109", "reason": "no longer needed"}
        )
    assert denied.value.code == "capability_denied"


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"item_ids": ["z"]}, ["a", "x"]),
        ({"new_item_ids": ["y"]}, ["a", "b"]),
        ({"payment_method_id": "guessed-card"}, ["card", "gift"]),
    ],
)
def test_write_identifier_choices_use_order_product_and_user(
    boundary: tuple, fields: dict, expected: list[str]
) -> None:
    """原规格限定订单，替换规格限定对应商品，支付方式限定已验证账户。"""
    gateway, context = boundary
    context = child(gateway, context, "modify_pending_order_items")
    args = {
        "order_id": "W5061109",
        "item_ids": ["a"],
        "new_item_ids": ["b"],
        "payment_method_id": "card",
        **fields,
    }
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "modify_pending_order_items", args)
    assert error.value.code == "invalid_identifier"
    for value in expected:
        assert f'"{value}"' in str(error.value)
    assert gateway.find_operation(context, "modify_pending_order_items") is None


def test_single_item_query_uses_owned_order_products_but_product_catalog_is_public(
    boundary: tuple,
) -> None:
    """单规格读取可查已购商品的其他规格，商品目录读取仍允许未购买商品。"""
    gateway, context = boundary
    assert gateway.call(context, "get_item_details", {"item_id": " b "})["item_id"] == "b"
    assert gateway.records[-1]["args"] == {"item_id": "b"}
    assert (
        gateway.call(context, "get_product_details", {"product_id": " p3 "})["product_id"] == "p3"
    )
    with pytest.raises(BusinessError) as error:
        gateway.call(context, "get_item_details", {"item_id": "z"})
    assert error.value.code == "invalid_identifier"
    assert all(f'"{value}"' in str(error.value) for value in ("a", "b", "x", "y"))
    assert '"z"' not in str(error.value)


def test_normalized_prepare_and_committed_retry_share_authoritative_parameters(
    boundary: tuple,
) -> None:
    """格式差异不产生新操作，提交后旧规格消失也可按同请求权威记录重放。"""
    gateway, context = boundary
    child_context = child(gateway, context, "modify_pending_order_items")
    args = {
        "order_id": " w5061109 ",
        "item_ids": [" a "],
        "new_item_ids": [" b "],
        "payment_method_id": " card ",
    }
    operation = gateway.call(child_context, "modify_pending_order_items", args)
    normalized = {
        "order_id": "#W5061109",
        "item_ids": ["a"],
        "new_item_ids": ["b"],
        "payment_method_id": "card",
    }
    assert operation["args"] == gateway.records[-1]["args"] == normalized
    assert (
        gateway.call(child_context, "modify_pending_order_items", normalized)["operation_id"]
        == operation["operation_id"]
    )
    gateway.commit(context, operation["operation_id"], normalized)
    retry = gateway.call(child_context, "modify_pending_order_items", args)
    assert retry["operation_id"] == operation["operation_id"] and retry["status"] == "committed"
    assert gateway.records[-1]["args"] == normalized
    with pytest.raises(BusinessError) as denied:
        gateway.call(
            child_context,
            "modify_pending_order_items",
            {**normalized, "item_ids": ["x"], "new_item_ids": ["y"]},
        )
    assert denied.value.code == "request_conflict"


def test_other_user_cannot_replay_existing_request_key(boundary: tuple) -> None:
    """同任务键不能让另一已验证用户读取或重放他人的权威操作。"""
    gateway, context = boundary
    first = child(gateway, context, "cancel_pending_order")
    gateway.call(
        first, "cancel_pending_order", {"order_id": "W5061109", "reason": "no longer needed"}
    )
    other = replace(context, session=SessionContext("other", "u2"))
    other_child = child(gateway, other, "cancel_pending_order")
    with pytest.raises(BusinessError) as denied:
        gateway.call(
            other_child,
            "cancel_pending_order",
            {"order_id": "#W0000003", "reason": "no longer needed"},
        )
    assert denied.value.code == "forbidden"
