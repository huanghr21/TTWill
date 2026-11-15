"""验证并发业务提交期间的多表快照保持同一事务视图。"""

from pathlib import Path

from sqlalchemy import event

from ttwill.backend import RetailBackend, request_key


def test_export_snapshot_uses_one_view_during_concurrent_commit(
    tmp_path: Path, seed_snapshot: dict
) -> None:
    """读取用户后提交礼品卡退款，当前快照仍同时保留旧余额和旧订单。"""
    seed_snapshot["orders"]["#1"]["payment_history"][0]["payment_method_id"] = "gift"
    path = tmp_path / "snapshot.sqlite"
    reader = RetailBackend(path)
    writer = RetailBackend(path)
    try:
        reader.initialize(seed_snapshot)
        with reader.engine.connect() as connection:
            # WAL 允许在读取快照时提交另一事务，使读写交错可以在同一线程稳定复现。
            assert connection.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one() == "wal"
        args = {"order_id": "#1", "reason": "no longer needed"}
        operation = writer.prepare(
            "u1", "cancel_pending_order", args, request_key("snapshot", 1, "cancel_pending_order")
        )
        committed = False

        def commit_between_tables(
            connection, cursor, statement, parameters, context, executemany
        ) -> None:
            """在用户表读取结束、商品表开始读取时提交另一连接的取消订单事务。"""
            nonlocal committed
            if statement.startswith("SELECT products.") and not committed:
                result = writer.commit("u1", operation["operation_id"], args)
                assert result["status"] == "committed"
                committed = True

        event.listen(reader.engine, "before_cursor_execute", commit_between_tables)
        snapshot = reader.export_snapshot()
        assert committed
        assert snapshot == seed_snapshot

        current = writer.export_snapshot()
        assert current["users"]["u1"]["payment_methods"]["gift"]["balance"] == 360.0
        assert current["orders"]["#1"]["status"] == "cancelled"
    finally:
        writer.close()
        reader.close()
