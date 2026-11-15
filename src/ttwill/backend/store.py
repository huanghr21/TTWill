"""用 SQLite 保存业务实体、操作记录和事务内事件。"""

import hashlib
import json
import math
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Connection

from ttwill.backend.arguments import validate_arguments
from ttwill.backend.rules import WRITE_ACTIONS, BusinessError, apply_action

metadata = MetaData()
users = Table("users", metadata, Column("id", String, primary_key=True), Column("data", JSON))
products = Table("products", metadata, Column("id", String, primary_key=True), Column("data", JSON))
orders = Table(
    "orders",
    metadata,
    Column("id", String, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False, index=True),
    Column("data", JSON),
)
operations = Table(
    "operations",
    metadata,
    Column("operation_id", String, primary_key=True),
    Column("request_key", String, unique=True, nullable=False),
    Column("user_id", ForeignKey("users.id"), nullable=False),
    Column("target", String, nullable=False),
    Column("action", String, nullable=False),
    Column("args", JSON, nullable=False),
    Column("status", String, nullable=False),
    Column("prepared_at", Float),
    Column("expires_at", Float),
    Column("preview", JSON),
    Column("result", JSON),
    Column("reason", String),
)
Index(
    "one_prepared_per_target",
    operations.c.target,
    unique=True,
    sqlite_where=operations.c.status == "prepared",
)
events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("operation_id", ForeignKey("operations.operation_id"), nullable=False),
    Column("at", Float),
    Column("status", String),
    Column("reason", String),
)


def request_key(task_id: str, revision: int, action: str) -> str:
    """按任务、修订号和动作类型生成可在恢复时重算的请求标识。

    Args:
        task_id: 任务的稳定标识。
        revision: 任务修订号，须为正整数；请求内容修改后递增。
        action: 写操作名称，须属于 WRITE_ACTIONS。

    Returns:
        三个关联字段的 SHA-256 十六进制摘要；相同字段生成相同标识。

    Raises:
        BusinessError: 任务标识为空、修订号无效或动作不是受支持的写操作。
    """
    if not task_id or type(revision) is not int or revision < 1 or action not in WRITE_ACTIONS:
        raise BusinessError("请求关联字段无效", code="invalid_request_key")
    payload = json.dumps([task_id, revision, action], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _fold(value: str) -> str:
    """匹配身份时折叠大小写、重音和首尾空白，保留内部格式。"""
    return "".join(
        c for c in unicodedata.normalize("NFD", value.strip()) if not unicodedata.combining(c)
    ).casefold()


class RetailBackend:
    """提供身份查询、业务读取和有期限的准备/提交事务。"""

    def __init__(
        self, path: Path, *, ttl_seconds: float = 900, clock: Callable[[], float] = time.time
    ) -> None:
        """打开业务库并创建所需的表。

        Args:
            path: SQLite 文件路径；父目录不存在时一并创建。
            ttl_seconds: 准备操作的有效秒数，须为有限正数。
            clock: 返回当前 Unix 时间戳的函数，用于计算到期时间和记录事件。

        Raises:
            ValueError: ttl_seconds 不是有限正数。
        """
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("准备操作有效期必须是正数")
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 10})
        self.ttl_seconds = ttl_seconds
        self.clock = clock

        @event.listens_for(self.engine, "connect")
        def configure_sqlite(connection, record) -> None:
            """为每条 SQLite 连接开启外键约束。"""
            connection.execute("PRAGMA foreign_keys=ON")

        metadata.create_all(self.engine)

    def close(self) -> None:
        """释放连接池中的数据库连接。"""
        self.engine.dispose()

    @contextmanager
    def _transaction(self) -> Iterator[Connection]:
        """在读取可写状态前取得 SQLite 写锁，退出时提交或回滚事务。

        Yields:
            已开启事务的数据库连接；上下文内发生异常时由 SQLAlchemy 回滚。
        """
        with self.engine.begin() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            yield connection

    def initialize(self, snapshot: dict) -> None:
        """在同一事务中把业务快照导入空库。

        Args:
            snapshot: 包含 users、products、orders 的字典；各项以实体 ID 为键，
                值为实体数据，订单的 user_id 须指向快照内的用户。

        Raises:
            BusinessError: 快照顶层字段不符、实体键与 ID 不一致、订单用户不存在，
                或业务库已经包含实体。
        """
        if set(snapshot) != {"users", "products", "orders"}:
            raise BusinessError("业务快照须包含 users、products、orders", code="invalid_snapshot")
        with self._transaction() as connection:
            if any(
                connection.execute(select(table.c.id).limit(1)).first()
                for table in (users, products, orders)
            ):
                raise BusinessError("业务库已初始化", code="already_initialized")
            for name, table, id_field in (
                ("users", users, "user_id"),
                ("products", products, "product_id"),
                ("orders", orders, "order_id"),
            ):
                rows = []
                for key, value in snapshot[name].items():
                    if value[id_field] != key:
                        raise BusinessError("实体键与 ID 不一致", code="invalid_snapshot")
                    row = {"id": key, "data": value}
                    if name == "orders":
                        if value["user_id"] not in snapshot["users"]:
                            raise BusinessError("订单用户不存在", code="invalid_snapshot")
                        row["user_id"] = value["user_id"]
                    rows.append(row)
                if rows:
                    connection.execute(insert(table), rows)

    def export_snapshot(self) -> dict:
        """从同一数据库快照导出用户、商品和订单。

        Returns:
            包含 users、products、orders 的字典，各项按实体 ID 映射到实体数据；
            不包含操作记录和内部事件。
        """
        with self.engine.begin() as connection:
            connection.exec_driver_sql("BEGIN")
            return {
                name: {row.id: row.data for row in connection.execute(select(table))}
                for name, table in (("users", users), ("products", products), ("orders", orders))
            }

    def _entity(self, connection: Connection, table: Table, entity_id: str) -> dict:
        """按实体 ID 读取一条业务记录。

        Args:
            connection: 本次读取使用的数据库连接。
            table: 待读取的用户、商品或订单表。
            entity_id: 实体在该表中的 ID。

        Returns:
            记录的 data 字段，即业务实体字典。

        Raises:
            BusinessError: 指定实体不存在。
        """
        data = connection.execute(
            select(table.c.data).where(table.c.id == entity_id)
        ).scalar_one_or_none()
        if data is None:
            raise BusinessError("业务记录不存在", code="not_found")
        return data

    def find_user_id_by_email(self, email: str) -> str:
        """按邮箱匹配唯一用户。

        Args:
            email: 待匹配的邮箱；匹配时忽略大小写、重音和首尾空白。

        Returns:
            唯一匹配用户的 ID。

        Raises:
            BusinessError: 没有匹配用户或匹配到多个用户。
        """
        return self._find_user(lambda user: _fold(user["email"]) == _fold(email))

    def find_user_id_by_name_zip(self, first_name: str, last_name: str, zip: str) -> str:
        """按姓名和邮编匹配唯一用户。

        Args:
            first_name: 用户的名；匹配时忽略大小写、重音和首尾空白。
            last_name: 用户的姓，使用与 first_name 相同的匹配规则。
            zip: 用户地址的邮政编码，须完全一致。

        Returns:
            唯一匹配用户的 ID。

        Raises:
            BusinessError: 没有匹配用户或匹配到多个用户。
        """
        return self._find_user(
            lambda user: (
                _fold(user["name"]["first_name"]) == _fold(first_name)
                and _fold(user["name"]["last_name"]) == _fold(last_name)
                and user["address"]["zip"] == zip
            )
        )

    def _find_user(self, matches: Callable[[dict], bool]) -> str:
        """执行身份匹配并校验结果唯一性。

        Args:
            matches: 接收用户实体的匹配函数，返回 True 表示命中该用户。

        Returns:
            唯一匹配用户的 ID。

        Raises:
            BusinessError: 匹配用户数不是一个。
        """
        with self.engine.connect() as connection:
            found = [row.id for row in connection.execute(select(users)) if matches(row.data)]
        if len(found) != 1:
            raise BusinessError("用户不存在或匹配不唯一", code="identity_not_unique")
        return found[0]

    def query(self, user_id: str, action: str, args: dict) -> dict:
        """查询已验证用户可访问的业务实体或商品目录。

        Args:
            user_id: 调用方已验证的用户 ID，用于校验业务数据归属。
            action: 读取用户、订单、商品、规格或商品目录的工具名称。
            args: 与 action 对应的工具参数，字段和类型由参数模型校验。

        Returns:
            用户、订单、商品或规格的实体字典；查询目录时返回商品名称到商品 ID
            的映射。

        Raises:
            ValueError: 工具名称未知或参数结构无效。
            BusinessError: 调用用户或目标记录不存在、归属不符，或动作不属于查询。
        """
        args = validate_arguments(action, args)
        with self.engine.connect() as connection:
            user = self._entity(connection, users, user_id)
            if action == "get_user_details":
                if args["user_id"] != user_id:
                    raise BusinessError("用户归属不匹配", code="forbidden")
                return user
            if action == "get_order_details":
                order = self._entity(connection, orders, args["order_id"])
                self._check_owner(user_id, order["user_id"])
                return order
            if action == "get_product_details":
                return self._entity(connection, products, args["product_id"])
            if action == "get_item_details":
                for product in connection.execute(select(products.c.data)).scalars():
                    if args["item_id"] in product["variants"]:
                        return product["variants"][args["item_id"]]
                raise BusinessError("商品规格不存在", code="not_found")
            if action == "list_all_product_types":
                return {
                    product["name"]: product["product_id"]
                    for product in connection.execute(select(products.c.data)).scalars()
                }
        raise BusinessError("不支持该查询动作", code="unknown_action")

    def _check_owner(self, caller: str, owner: str) -> None:
        """校验调用身份与业务归属。"""
        if caller != owner:
            raise BusinessError("业务数据归属不匹配", code="forbidden")

    def _working_data(self, connection: Connection, user_id: str, args: dict) -> tuple:
        """取得本次动作涉及的实体副本，供规则预演或提交。

        Args:
            connection: 本次操作所在事务的数据库连接。
            user_id: 调用方已验证的用户 ID。
            args: 已通过结构校验的操作参数，用于定位用户和订单。

        Returns:
            按顺序包含以下对象的三元组：
                user: 用户实体的深拷贝。
                order: 订单实体的深拷贝；参数不含 order_id 时为 None。
                catalog: 以商品 ID 为键的相关商品字典；没有订单时为空。

        Raises:
            BusinessError: 相关实体不存在，或参数中的用户、订单不属于调用方。
        """
        user = self._entity(connection, users, user_id)
        order = None
        if "user_id" in args:
            self._check_owner(user_id, args["user_id"])
        if "order_id" in args:
            order = self._entity(connection, orders, args["order_id"])
            self._check_owner(user_id, order["user_id"])
        product_ids = {item["product_id"] for item in order["items"]} if order else set()
        catalog = {
            product_id: self._entity(connection, products, product_id) for product_id in product_ids
        }
        return deepcopy(user), deepcopy(order), catalog

    def _transition(
        self,
        connection: Connection,
        operation_id: str,
        status: str,
        *,
        reason: str | None = None,
        result: dict | None = None,
    ) -> None:
        """在同一事务更新操作状态并记录业务事件。

        Args:
            connection: 本次状态变化所在事务的数据库连接。
            operation_id: 待更新操作的 ID。
            status: 更新后的操作状态。
            reason: 失败、拒绝或过期的原因；None 表示没有原因说明。
            result: 提交产生的业务结果；None 表示当前状态没有提交结果。
        """
        connection.execute(
            update(operations)
            .where(operations.c.operation_id == operation_id)
            .values(status=status, reason=reason, result=result)
        )
        connection.execute(
            insert(events).values(
                operation_id=operation_id, at=self.clock(), status=status, reason=reason
            )
        )

    def _expire(self, connection: Connection) -> None:
        """将到期的待提交操作记为过期，释放其订单占用。"""
        expired = (
            connection.execute(
                select(operations.c.operation_id).where(
                    operations.c.status == "prepared", operations.c.expires_at <= self.clock()
                )
            )
            .scalars()
            .all()
        )
        for operation_id in expired:
            self._transition(connection, operation_id, "expired", reason="准备操作已过期")

    def prepare(self, user_id: str, action: str, args: dict, request_key: str) -> dict:
        """预演写操作，保存不可变参数、预期结果和业务对象占用。

        此方法保存操作记录，不修改业务实体。

        Args:
            user_id: 调用方已验证的用户 ID。
            action: 待准备的写操作名称。
            args: 与 action 对应的完整操作参数。
            request_key: 按任务 ID、修订号和动作类型生成的稳定请求标识。

        Returns:
            包含 operation_id、request_key、action、args、status、preview 等字段的
            操作记录；新操作状态为 prepared，result 为 None。同一请求重试时
            返回已有记录及其当前状态。

        Raises:
            ValueError: 操作参数结构无效。
            BusinessError: 写操作或请求标识无效、关联实体不存在、归属不符、
                请求标识已绑定其他参数、业务对象被占用，或业务规则不满足。
        """
        if action not in WRITE_ACTIONS or not request_key:
            raise BusinessError("准备请求无效", code="invalid_request")
        args = validate_arguments(action, args)
        with self._transaction() as connection:
            self._expire(connection)
            existing = (
                connection.execute(
                    select(operations).where(operations.c.request_key == request_key)
                )
                .mappings()
                .first()
            )
            if existing:
                self._check_owner(user_id, existing["user_id"])
                if existing["action"] != action or existing["args"] != args:
                    raise BusinessError("请求标识已用于其他参数", code="request_conflict")
                return dict(existing)
            user, order, catalog = self._working_data(connection, user_id, args)
            target = f"order:{order['order_id']}" if order else f"user:{user_id}"
            if connection.execute(
                select(operations.c.operation_id).where(
                    operations.c.target == target, operations.c.status == "prepared"
                )
            ).first():
                raise BusinessError("该业务对象还有待确认操作", code="target_busy")
            preview = apply_action(action, args, user, order, catalog)
            now = self.clock()
            record = dict(
                operation_id=uuid4().hex,
                request_key=request_key,
                user_id=user_id,
                target=target,
                action=action,
                args=args,
                status="prepared",
                prepared_at=now,
                expires_at=now + self.ttl_seconds,
                preview=preview,
                result=None,
                reason=None,
            )
            connection.execute(insert(operations), record)
            connection.execute(
                insert(events).values(
                    operation_id=record["operation_id"], at=now, status="prepared", reason=None
                )
            )
            return record

    def _operation(self, connection: Connection, user_id: str, **identifier: str) -> dict:
        """按唯一标识取得操作并校验身份归属。

        Args:
            connection: 本次读取使用的数据库连接。
            user_id: 调用方已验证的用户 ID。
            **identifier: 以 operation_id 或 request_key 为键的单项查询条件。

        Returns:
            包含操作身份、参数、状态和结果的完整操作记录。

        Raises:
            BusinessError: 操作不存在或不属于调用方。
        """
        column, value = next(iter(identifier.items()))
        record = (
            connection.execute(select(operations).where(operations.c[column] == value))
            .mappings()
            .first()
        )
        if record is None:
            raise BusinessError("操作不存在", code="not_found")
        self._check_owner(user_id, record["user_id"])
        return dict(record)

    def get_operation(
        self, user_id: str, *, operation_id: str | None = None, request_key: str | None = None
    ) -> dict:
        """按操作 ID 或可重算的请求标识对账，并更新到期状态。

        Args:
            user_id: 调用方已验证的用户 ID。
            operation_id: 后端生成的操作 ID，与 request_key 二选一。
            request_key: 按任务状态重算的请求标识，与 operation_id 二选一。

        Returns:
            操作的完整记录，包含当前 status、准备参数 args、预期结果 preview
            和提交结果 result；尚无提交结果时 result 为 None。

        Raises:
            ValueError: 同时提供或均未提供两个操作标识。
            BusinessError: 操作不存在或不属于调用方。
        """
        if (operation_id is None) == (request_key is None):
            raise ValueError("必须提供且只提供一个操作标识")
        with self._transaction() as connection:
            self._expire(connection)
            identifier = (
                {"operation_id": operation_id} if operation_id else {"request_key": request_key}
            )
            return self._operation(connection, user_id, **identifier)

    def commit(self, user_id: str, operation_id: str, expected_args: dict) -> dict:
        """重新校验准备参数和业务规则，原子写入业务、操作状态与事件。

        业务校验失败时持久化失败原因并释放对象占用。对话授权由调用方的
        确认代码检查。

        Args:
            user_id: 调用方已验证的用户 ID。
            operation_id: 待提交操作的 ID。
            expected_args: 已展示给用户的准备参数，须与操作保存的 args 完全一致。

        Returns:
            状态为 committed 的完整操作记录，result 为提交后的订单或用户数据；
            重复提交已提交操作时返回原记录。

        Raises:
            BusinessError: 操作不存在、归属不符、参数不一致、操作已失效、
                业务规则不满足，或重新预演的结果与准备结果不同。
        """
        failure = None
        with self._transaction() as connection:
            self._expire(connection)
            operation = self._operation(connection, user_id, operation_id=operation_id)
            if operation["args"] != expected_args:
                raise BusinessError("提交参数与准备参数不一致", code="parameter_mismatch")
            if operation["status"] == "committed":
                return operation
            if operation["status"] != "prepared":
                failure = BusinessError("操作已失效", code=operation["status"])
            else:
                try:
                    user, order, catalog = self._working_data(
                        connection, user_id, operation["args"]
                    )
                    result = apply_action(
                        operation["action"], operation["args"], user, order, catalog
                    )
                    if result != operation["preview"]:
                        raise BusinessError("业务数据已变化，需要重新准备", code="stale_operation")
                except BusinessError as exc:
                    failure = exc
                    self._transition(connection, operation_id, "failed", reason=str(exc))
                else:
                    connection.execute(update(users).where(users.c.id == user_id).values(data=user))
                    if order is not None:
                        connection.execute(
                            update(orders)
                            .where(orders.c.id == order["order_id"])
                            .values(data=order)
                        )
                    self._transition(connection, operation_id, "committed", result=result)
            record = self._operation(connection, user_id, operation_id=operation_id)
        if failure is not None:
            raise failure
        return record

    def reject(self, user_id: str, operation_id: str, reason: str) -> dict:
        """拒绝或撤销待提交操作，保留原因并释放业务对象占用。

        Args:
            user_id: 调用方已验证的用户 ID。
            operation_id: 待拒绝或撤销操作的 ID。
            reason: 拒绝或撤销原因，须包含非空白字符。

        Returns:
            当前操作记录；prepared 操作更新为 rejected，已经失效的操作保持
            原状态。

        Raises:
            ValueError: reason 为空或仅含空白。
            BusinessError: 操作不存在、不属于调用方，或已经提交。
        """
        if not reason.strip():
            raise ValueError("须提供拒绝或撤销原因")
        with self._transaction() as connection:
            self._expire(connection)
            operation = self._operation(connection, user_id, operation_id=operation_id)
            if operation["status"] == "committed":
                raise BusinessError("操作已提交", code="already_committed")
            if operation["status"] == "prepared":
                self._transition(connection, operation_id, "rejected", reason=reason)
            return self._operation(connection, user_id, operation_id=operation_id)

    def operation_events(self, user_id: str, operation_id: str) -> list[dict]:
        """读取当前用户一项操作的状态事件。

        Args:
            user_id: 调用方已验证的用户 ID。
            operation_id: 待读取事件的操作 ID。

        Returns:
            按事件 ID 升序排列的记录列表，每项包含操作 ID、时间戳、状态与原因。

        Raises:
            BusinessError: 操作不存在或不属于调用方。
        """
        with self.engine.connect() as connection:
            self._operation(connection, user_id, operation_id=operation_id)
            return [
                dict(row)
                for row in connection.execute(
                    select(events)
                    .where(events.c.operation_id == operation_id)
                    .order_by(events.c.id)
                ).mappings()
            ]
