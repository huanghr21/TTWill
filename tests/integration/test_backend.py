"""用独立 SQLite 库验证两阶段写入、恢复对账、事务和并发占用。"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import event, update

from ttwill.backend import BusinessError, RetailBackend, request_key
from ttwill.backend.store import orders, products, users

CANCEL = "cancel_pending_order"
CANCEL_ARGS = {"order_id": "#1", "reason": "no longer needed"}
MODIFY = "modify_pending_order_items"
MODIFY_ARGS = {
    "order_id": "#1",
    "item_ids": ["a"],
    "new_item_ids": ["b"],
    "payment_method_id": "gift",
}


@pytest.fixture
def peer(backend: RetailBackend) -> Iterator[RetailBackend]:
    """为同一业务库创建第二个独立后端实例，模拟重启与并发进程。

    Args:
        backend: 已初始化的原后端，提供共享数据库路径。

    Yields:
        指向同一数据库的新后端实例，测试结束后关闭其连接池。
    """
    instance = RetailBackend(Path(backend.engine.url.database))
    yield instance
    instance.close()


def test_prepare_only_records_operation(backend: RetailBackend, seed_snapshot: dict) -> None:
    """准备阶段返回业务预演并登记事件，业务实体保持原样。"""
    operation = backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t1", 1, MODIFY))
    assert operation["status"] == "prepared"
    assert operation["preview"]["items"][0]["item_id"] == "b"
    assert backend.export_snapshot() == seed_snapshot
    assert [
        entry["status"] for entry in backend.operation_events("u1", operation["operation_id"])
    ] == ["prepared"]


def test_prepare_replay_recovers_by_deterministic_key(
    backend: RetailBackend, peer: RetailBackend
) -> None:
    """新实例按恢复的任务字段重算标识，找回已准备但未记下 ID 的同一操作。"""
    operation = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("任务一", 1, CANCEL))
    recalculated = request_key("任务一", 1, CANCEL)
    recovered = peer.get_operation("u1", request_key=recalculated)
    replayed = peer.prepare("u1", CANCEL, CANCEL_ARGS, recalculated)
    assert recovered == replayed == operation
    assert len(peer.operation_events("u1", operation["operation_id"])) == 1


def test_request_key_separates_revision_and_action() -> None:
    """修改任务修订号或动作类型后生成新的请求标识。"""
    original = request_key("任务一", 1, CANCEL)
    assert original != request_key("任务一", 2, CANCEL)
    assert original != request_key("任务一", 1, MODIFY)
    assert original == request_key("任务一", 1, CANCEL)


@pytest.mark.parametrize("revision", [0, -1, True, 1.5])
def test_request_key_requires_positive_integer_revision(revision: object) -> None:
    """关联标识的修订号必须是正整数，布尔值不能代替修订号。"""
    with pytest.raises(BusinessError) as caught:
        request_key("t1", revision, CANCEL)
    assert caught.value.code == "invalid_request_key"


@pytest.mark.parametrize(
    ("action", "args"),
    [(CANCEL, {**CANCEL_ARGS, "reason": "ordered by mistake"}), (MODIFY, MODIFY_ARGS)],
)
def test_prepare_key_conflict_preserves_original(
    backend: RetailBackend, action: str, args: dict
) -> None:
    """同一关联标识不能绑定不同参数或动作，原准备记录保持不变。"""
    key = request_key("t1", 1, CANCEL)
    original = backend.prepare("u1", CANCEL, CANCEL_ARGS, key)
    with pytest.raises(BusinessError) as caught:
        backend.prepare("u1", action, args, key)
    assert caught.value.code == "request_conflict"
    assert backend.get_operation("u1", request_key=key) == original


@pytest.mark.parametrize(
    ("action", "args"),
    [("get_order_details", {"order_id": "#3"}), ("get_user_details", {"user_id": "u2"})],
)
def test_query_denies_other_users_records(backend: RetailBackend, action: str, args: dict) -> None:
    """已验证用户不能读取他人的订单或账户资料。"""
    with pytest.raises(BusinessError) as caught:
        backend.query("u1", action, args)
    assert caught.value.code == "forbidden"


def test_prepare_denies_other_users_order(backend: RetailBackend, seed_snapshot: dict) -> None:
    """订单归属不匹配时不能生成准备记录，也不能改动业务数据。"""
    with pytest.raises(BusinessError) as caught:
        backend.prepare(
            "u1", CANCEL, {**CANCEL_ARGS, "order_id": "#3"}, request_key("t1", 1, CANCEL)
        )
    assert caught.value.code == "forbidden"
    assert backend.export_snapshot() == seed_snapshot


def test_prepare_denies_other_users_default_address(
    backend: RetailBackend, seed_snapshot: dict
) -> None:
    """用户默认地址的写请求必须属于调用身份。"""
    action = "modify_user_address"
    args = {"user_id": "u2", **seed_snapshot["users"]["u2"]["address"]}
    with pytest.raises(BusinessError) as caught:
        backend.prepare("u1", action, args, request_key("t1", 1, action))
    assert caught.value.code == "forbidden"
    assert backend.export_snapshot() == seed_snapshot


@pytest.mark.parametrize("method", ["get_id", "get_key", "commit", "reject", "events", "prepare"])
def test_operation_access_checks_owner(backend: RetailBackend, method: str) -> None:
    """知道另一用户的操作 ID 或请求标识也不能查询、重放或提交其操作。"""
    key = request_key("t1", 1, CANCEL)
    operation = backend.prepare("u1", CANCEL, CANCEL_ARGS, key)
    operation_id = operation["operation_id"]
    with pytest.raises(BusinessError) as caught:
        if method == "get_id":
            backend.get_operation("u2", operation_id=operation_id)
        elif method == "get_key":
            backend.get_operation("u2", request_key=key)
        elif method == "commit":
            backend.commit("u2", operation_id, CANCEL_ARGS)
        elif method == "reject":
            backend.reject("u2", operation_id, "撤销")
        elif method == "events":
            backend.operation_events("u2", operation_id)
        else:
            backend.prepare("u2", CANCEL, CANCEL_ARGS, key)
    assert caught.value.code == "forbidden"
    assert backend.get_operation("u1", operation_id=operation_id)["status"] == "prepared"


def test_commit_retries_return_saved_result(backend: RetailBackend, peer: RetailBackend) -> None:
    """提交后跨实例重试返回同一结果，礼品卡差价只扣一次。"""
    key = request_key("t1", 1, MODIFY)
    prepared = backend.prepare("u1", MODIFY, MODIFY_ARGS, key)
    committed = backend.commit("u1", prepared["operation_id"], MODIFY_ARGS)
    replayed = peer.commit("u1", prepared["operation_id"], MODIFY_ARGS)
    assert committed == replayed
    assert peer.get_operation("u1", request_key=key) == committed
    assert committed["status"] == "committed"
    snapshot = peer.export_snapshot()
    assert snapshot["users"]["u1"]["payment_methods"]["gift"]["balance"] == 290.0
    assert len(snapshot["orders"]["#1"]["payment_history"]) == 2
    assert [entry["status"] for entry in peer.operation_events("u1", prepared["operation_id"])] == [
        "prepared",
        "committed",
    ]


def test_commit_rejects_changed_parameters(backend: RetailBackend, seed_snapshot: dict) -> None:
    """提交时参数必须与准备记录一致，修改参数不能消费原操作。"""
    prepared = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 1, CANCEL))
    with pytest.raises(BusinessError) as caught:
        backend.commit(
            "u1", prepared["operation_id"], {**CANCEL_ARGS, "reason": "ordered by mistake"}
        )
    assert caught.value.code == "parameter_mismatch"
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "prepared"
    )
    assert backend.export_snapshot() == seed_snapshot


def test_expired_operation_releases_target(backend: RetailBackend) -> None:
    """到期操作保存过期事件，释放订单占用供新修订请求准备。"""
    now = [100.0]

    def clock() -> float:
        """返回测试指定的时间。"""
        return now[0]

    backend.clock = clock
    backend.ttl_seconds = 10
    old = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 1, CANCEL))
    now[0] = 110.0
    with pytest.raises(BusinessError) as caught:
        backend.commit("u1", old["operation_id"], CANCEL_ARGS)
    assert caught.value.code == "expired"
    renewed = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 2, CANCEL))
    assert renewed["operation_id"] != old["operation_id"]
    assert backend.get_operation("u1", operation_id=old["operation_id"])["status"] == "expired"
    assert [entry["status"] for entry in backend.operation_events("u1", old["operation_id"])] == [
        "prepared",
        "expired",
    ]


def test_pending_target_blocks_other_action_but_not_other_order(backend: RetailBackend) -> None:
    """同一订单从准备开始串行占用，其他订单可以独立准备。"""
    backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 1, CANCEL))
    with pytest.raises(BusinessError) as caught:
        backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t2", 1, MODIFY))
    assert caught.value.code == "target_busy"
    other = backend.prepare(
        "u1", CANCEL, {**CANCEL_ARGS, "order_id": "#4"}, request_key("t3", 1, CANCEL)
    )
    assert other["status"] == "prepared"


def test_reject_preserves_reason_and_releases_target(backend: RetailBackend) -> None:
    """拒绝操作后保留事实与原因，新请求可以继续占用同一订单。"""
    old = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 1, CANCEL))
    rejected = backend.reject("u1", old["operation_id"], "用户改为换规格")
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "用户改为换规格"
    new = backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t2", 1, MODIFY))
    assert new["status"] == "prepared"
    with pytest.raises(BusinessError) as caught:
        backend.commit("u1", old["operation_id"], CANCEL_ARGS)
    assert caught.value.code == "rejected"


def test_reject_cannot_erase_committed_fact(backend: RetailBackend) -> None:
    """已提交操作不能通过拒绝入口撤销或擦除。"""
    prepared = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 1, CANCEL))
    backend.commit("u1", prepared["operation_id"], CANCEL_ARGS)
    with pytest.raises(BusinessError) as caught:
        backend.reject("u1", prepared["operation_id"], "用户改口")
    assert caught.value.code == "already_committed"
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "committed"
    )


def test_commit_revalidates_status_and_records_failure(backend: RetailBackend) -> None:
    """准备后订单状态改变会导致提交失败，失败事件与释放占用一并保存。"""
    prepared = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 1, CANCEL))
    changed = backend.query("u1", "get_order_details", {"order_id": "#1"})
    changed["status"] = "processed"
    with backend.engine.begin() as connection:
        connection.execute(update(orders).where(orders.c.id == "#1").values(data=changed))
    with pytest.raises(BusinessError) as caught:
        backend.commit("u1", prepared["operation_id"], CANCEL_ARGS)
    assert caught.value.code == "invalid_status"
    operation = backend.get_operation("u1", operation_id=prepared["operation_id"])
    assert operation["status"] == "failed"
    assert operation["reason"]
    assert [
        entry["status"] for entry in backend.operation_events("u1", prepared["operation_id"])
    ] == ["prepared", "failed"]
    assert backend.query("u1", "get_order_details", {"order_id": "#1"}) == changed
    changed["status"] = "pending"
    with backend.engine.begin() as connection:
        connection.execute(update(orders).where(orders.c.id == "#1").values(data=changed))
    renewed = backend.prepare("u1", CANCEL, CANCEL_ARGS, request_key("t1", 2, CANCEL))
    assert renewed["status"] == "prepared"


def test_commit_rejects_price_change_since_preview(backend: RetailBackend) -> None:
    """准备后商品价格变动时不能按新价格悄悄执行原确认。"""
    prepared = backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t1", 1, MODIFY))
    product = backend.query("u1", "get_product_details", {"product_id": "p1"})
    product["variants"]["b"]["price"] = 45.0
    with backend.engine.begin() as connection:
        connection.execute(update(products).where(products.c.id == "p1").values(data=product))
    before = backend.export_snapshot()
    with pytest.raises(BusinessError) as caught:
        backend.commit("u1", prepared["operation_id"], MODIFY_ARGS)
    assert caught.value.code == "stale_operation"
    assert backend.export_snapshot() == before
    assert backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "failed"


def test_commit_rechecks_gift_card_balance(backend: RetailBackend) -> None:
    """准备后被其他订单使用的礼品卡余额需在提交时重新校验。"""
    prepared = backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t1", 1, MODIFY))
    user = backend.query("u1", "get_user_details", {"user_id": "u1"})
    user["payment_methods"]["gift"]["balance"] = 0.0
    with backend.engine.begin() as connection:
        connection.execute(update(users).where(users.c.id == "u1").values(data=user))
    with pytest.raises(BusinessError) as caught:
        backend.commit("u1", prepared["operation_id"], MODIFY_ARGS)
    assert caught.value.code == "insufficient_balance"
    assert backend.query("u1", "get_order_details", {"order_id": "#1"})["status"] == "pending"


def test_event_write_failure_rolls_back_business_and_commit_marker(
    backend: RetailBackend, seed_snapshot: dict
) -> None:
    """事件入库失败时回滚业务写入和消费标志，原准备仍能重新提交。"""
    prepared = backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t1", 1, MODIFY))

    def fail_event_write(connection, cursor, statement, parameters, context, executemany) -> None:
        """模拟业务结果更新后，追加事件遇到存储异常。"""
        if statement.startswith("INSERT INTO events"):
            raise RuntimeError("模拟事件存储失败")

    event.listen(backend.engine, "before_cursor_execute", fail_event_write)
    try:
        with pytest.raises(RuntimeError, match="模拟事件存储失败"):
            backend.commit("u1", prepared["operation_id"], MODIFY_ARGS)
    finally:
        event.remove(backend.engine, "before_cursor_execute", fail_event_write)
    assert backend.export_snapshot() == seed_snapshot
    assert (
        backend.get_operation("u1", operation_id=prepared["operation_id"])["status"] == "prepared"
    )
    assert len(backend.operation_events("u1", prepared["operation_id"])) == 1
    committed = backend.commit("u1", prepared["operation_id"], MODIFY_ARGS)
    assert committed["status"] == "committed"
    assert (
        backend.query("u1", "get_user_details", {"user_id": "u1"})["payment_methods"]["gift"][
            "balance"
        ]
        == 290.0
    )


def test_prepare_event_failure_leaves_no_operation_or_target_lock(backend: RetailBackend) -> None:
    """准备事件写入失败会回滚操作及占用，相同请求随后可以正常准备。"""
    key = request_key("t1", 1, CANCEL)

    def fail_event_write(connection, cursor, statement, parameters, context, executemany) -> None:
        """模拟准备记录插入后的事件写入失败。"""
        if statement.startswith("INSERT INTO events"):
            raise RuntimeError("模拟准备事件失败")

    event.listen(backend.engine, "before_cursor_execute", fail_event_write)
    try:
        with pytest.raises(RuntimeError, match="模拟准备事件失败"):
            backend.prepare("u1", CANCEL, CANCEL_ARGS, key)
    finally:
        event.remove(backend.engine, "before_cursor_execute", fail_event_write)
    with pytest.raises(BusinessError) as caught:
        backend.get_operation("u1", request_key=key)
    assert caught.value.code == "not_found"
    assert backend.prepare("u1", CANCEL, CANCEL_ARGS, key)["status"] == "prepared"


def test_concurrent_same_request_gets_one_operation(
    backend: RetailBackend, peer: RetailBackend
) -> None:
    """两个实例同时重放同一请求时只建立一个操作。"""
    barrier = Barrier(2)
    key = request_key("t1", 1, CANCEL)

    def prepare(instance: RetailBackend) -> dict:
        """同步启动各实例的准备调用。"""
        barrier.wait(timeout=5)
        return instance.prepare("u1", CANCEL, CANCEL_ARGS, key)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(prepare, (backend, peer)))
    assert first["operation_id"] == second["operation_id"]
    assert len(backend.operation_events("u1", first["operation_id"])) == 1


def test_concurrent_different_requests_share_target_lock(
    backend: RetailBackend, peer: RetailBackend
) -> None:
    """不同实例同时准备同一订单时只有一个成功，另一个得到明确占用错误。"""
    barrier = Barrier(2)

    def prepare(index: int) -> str:
        """以不同请求标识竞争同一订单，并返回执行状态。"""
        instance = (backend, peer)[index]
        barrier.wait(timeout=5)
        try:
            operation = instance.prepare(
                "u1", CANCEL, CANCEL_ARGS, request_key(f"t{index}", 1, CANCEL)
            )
        except BusinessError as exc:
            return exc.code
        return operation["status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(prepare, (0, 1)))
    assert sorted(statuses) == ["prepared", "target_busy"]


def test_concurrent_commit_changes_business_once(
    backend: RetailBackend, peer: RetailBackend
) -> None:
    """同一操作被并发提交时只修改一次业务数据和礼品卡余额。"""
    operation = backend.prepare("u1", MODIFY, MODIFY_ARGS, request_key("t1", 1, MODIFY))
    barrier = Barrier(2)

    def commit(instance: RetailBackend) -> dict:
        """同步提交同一个已准备操作。"""
        barrier.wait(timeout=5)
        return instance.commit("u1", operation["operation_id"], deepcopy(MODIFY_ARGS))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(commit, (backend, peer)))
    assert first == second
    assert first["status"] == "committed"
    assert (
        backend.query("u1", "get_user_details", {"user_id": "u1"})["payment_methods"]["gift"][
            "balance"
        ]
        == 290.0
    )
    assert len(backend.operation_events("u1", operation["operation_id"])) == 2
