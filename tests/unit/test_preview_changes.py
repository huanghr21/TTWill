"""验证商品修改确认只展示真实改动，保留完整业务预览与收付款金额。"""

import json
import re
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest

from ttwill.backend import BusinessError, RetailBackend, request_key
from ttwill.gateway import CallContext, SessionContext, ToolGateway
from ttwill.gateway.presentation import operation_content
from ttwill.gateway.skills import SkillCatalog
from ttwill.tracing import Trace


@pytest.fixture
def retail_60_gateway(tmp_path: Path, skill_catalog: SkillCatalog) -> Iterator[ToolGateway]:
    """以中文开发题 60 的真实订单和商品构造小型业务库。

    Args:
        tmp_path: 本项测试独占的数据库及轨迹目录。
        skill_catalog: 覆盖全部写动作的测试政策目录。

    Yields:
        包含四件商品订单及所属用户的网关，结束时关闭业务库。
    """
    source = Path(__file__).resolve().parents[2] / "data/source/tau2_zh_retail.json"
    database = json.loads(source.read_text())["database"]
    order = database["orders"]["#W5061109"]
    user = deepcopy(database["users"][order["user_id"]])
    user["orders"] = [order["order_id"]]
    snapshot = {
        "users": {user["user_id"]: user},
        "orders": {order["order_id"]: order},
        "products": {
            item["product_id"]: database["products"][item["product_id"]] for item in order["items"]
        },
    }
    backend = RetailBackend(tmp_path / "business.sqlite")
    backend.initialize(snapshot)
    try:
        yield ToolGateway(backend, Trace(tmp_path / "run", "preview"), skill_catalog)
    finally:
        backend.close()


def shown_preview(content: str) -> str:
    """取得用户确认正文中的预期结果文本。

    Args:
        content: 展示层生成的完整中文确认正文。

    Returns:
        包含中文字段和缩进列表的预期结果正文。
    """
    return content.split("\n\n确认后预期结果：\n", 1)[1]


def prepare_earbuds(gateway: ToolGateway) -> dict:
    """按开发题 60 的参考动作准备耳机规格修改。

    Args:
        gateway: 已导入该题用户及订单的网关。

    Returns:
        后端准备记录，仍包含修改后全部四件商品的完整预览。
    """
    action = "modify_pending_order_items"
    return gateway.backend.prepare(
        "ming_zhu_4204",
        action,
        {
            "order_id": "#W5061109",
            "item_ids": ["3694871183"],
            "new_item_ids": ["6077640618"],
            "payment_method_id": "paypal_3742148",
        },
        request_key("earbuds", 1, action),
    )


def test_retail_60_shows_only_earbuds_and_preserves_full_authoritative_preview(
    retail_60_gateway: ToolGateway,
) -> None:
    """真实 60 题只改耳机，展示原规格到新规格并保留全部付款退款记录。"""
    gateway = retail_60_gateway
    operation = prepare_earbuds(gateway)
    before = deepcopy(operation)
    initial = gateway.backend.export_snapshot()
    context = CallContext(SessionContext("preview", "ming_zhu_4204"), "coordinator", "main")

    content = gateway.operation_content(context, operation)

    preview = shown_preview(content)
    assert [line for line in preview.splitlines() if line.startswith("· ")] == [
        "· 订单状态：待发货（商品修改后）",
        "· 本次商品变更：",
        "· 支付与退款明细：",
    ]
    assert preview.count("· 原商品：") == preview.count("· 新规格：") == 1
    original, replacement = preview.split("    · 原商品：\n", 1)[1].split("    · 新规格：\n", 1)
    assert "      · 商品名称：无线耳机" in original
    assert "      · 商品规格编号：3694871183" in original
    assert "      · 价格：1796.69" in original
    assert "        · color：白色" in original
    assert "      · 商品规格编号：6077640618" in replacement
    assert "      · 价格：1700.44" in replacement
    assert (
        "      · 规格：\n"
        "        · color：蓝色\n"
        "        · battery life：8小时\n"
        "        · water resistance：不防水"
    ) in replacement
    assert preview.endswith(
        "· 支付与退款明细：\n"
        "  · 第 1 项：\n"
        "    · 类型：付款\n"
        "    · 金额：9236.01\n"
        "    · 支付方式：paypal_3742148\n"
        "  · 第 2 项：\n"
        "    · 类型：退款\n"
        "    · 金额：96.25\n"
        "    · 支付方式：paypal_3742148"
    )
    assert all(
        value not in content
        for value in (
            "彩妆套装",
            "办公椅",
            "蓝牙音箱",
            "ming_zhu_4204",
            operation["operation_id"],
            operation["request_key"],
        )
    )
    assert operation == before
    assert len(operation["preview"]["items"]) == 4
    assert gateway.get_operation(context, operation["operation_id"])["preview"] == before["preview"]
    assert gateway.backend.export_snapshot() == initial


def test_duplicate_items_keep_original_purchase_prices_and_pair_by_order_position(
    tmp_path: Path,
    seed_snapshot: dict,
    skill_catalog: SkillCatalog,
) -> None:
    """相同规格重复出现时按原条目配对，不混入未修改的既有目标规格。"""
    original = deepcopy(seed_snapshot["orders"]["#W0000001"]["items"][0])
    existing = {**deepcopy(original), "item_id": "b", "price": 17.0, "options": {"颜色": "红"}}
    discounted = {**deepcopy(original), "price": 19.0}
    seed_snapshot["orders"]["#W0000001"]["items"] = [existing, discounted, original]
    backend = RetailBackend(tmp_path / "duplicates.sqlite")
    backend.initialize(seed_snapshot)
    gateway = ToolGateway(backend, Trace(tmp_path / "run", "duplicates"), skill_catalog)
    context = CallContext(SessionContext("duplicates", "u1"), "coordinator", "main")
    action = "modify_pending_order_items"
    try:
        operation = backend.prepare(
            "u1",
            action,
            {
                "order_id": "#W0000001",
                "item_ids": ["a", "a"],
                "new_item_ids": ["b", "b"],
                "payment_method_id": "card",
            },
            request_key("duplicate-items", 1, action),
        )
        preview = shown_preview(gateway.operation_content(context, operation))
        prices = re.findall(
            r"· 原商品：.*?· 价格：(\S+).*?· 新规格：.*?· 价格：(\S+)",
            preview,
            re.DOTALL,
        )
        assert prices == [("19.0", "30.0"), ("20.0", "30.0")]
        assert "17.0" not in preview
        assert len(operation["preview"]["items"]) == 3
    finally:
        backend.close()


def test_item_modify_requires_original_items_in_presentation(
    retail_60_gateway: ToolGateway,
) -> None:
    """缺少原始订单商品时不能用修改后预览猜测原规格和成交价。"""
    with pytest.raises(ValueError, match="原始订单商品"):
        operation_content(prepare_earbuds(retail_60_gateway))


def test_cancel_preview_preserves_existing_scope(retail_60_gateway: ToolGateway) -> None:
    """取消订单保持原有状态、原因和全量收付款明细，不套用商品修改裁剪。"""
    gateway = retail_60_gateway
    action = "cancel_pending_order"
    operation = gateway.backend.prepare(
        "ming_zhu_4204",
        action,
        {"order_id": "#W5061109", "reason": "no longer needed"},
        request_key("cancel", 1, action),
    )
    context = CallContext(SessionContext("preview", "ming_zhu_4204"), "coordinator", "main")
    content = gateway.operation_content(context, operation)
    assert content == operation_content(operation)
    preview = shown_preview(content)
    assert [line for line in preview.splitlines() if line.startswith("· ")] == [
        "· 订单状态：取消",
        "· 取消原因：不再需要（no longer needed）",
        "· 支付与退款明细：",
    ]
    assert re.findall(r"· 金额：(\S+)", preview) == ["9236.01", "9236.01"]


@pytest.mark.parametrize("user_id", [None, "other-user"])
def test_operation_presentation_requires_operation_owner(
    retail_60_gateway: ToolGateway,
    user_id: str | None,
) -> None:
    """展示入口仍校验身份和操作归属，外部操作不能泄露订单原条目。"""
    context = CallContext(SessionContext("preview", user_id), "coordinator", "main")
    with pytest.raises(BusinessError):
        retail_60_gateway.operation_content(context, prepare_earbuds(retail_60_gateway))


def test_operation_presentation_rejects_inactive_operation(retail_60_gateway: ToolGateway) -> None:
    """操作作废后不再把当前订单包装成可确认的旧修改。"""
    gateway = retail_60_gateway
    context = CallContext(SessionContext("preview", "ming_zhu_4204"), "coordinator", "main")
    operation = prepare_earbuds(gateway)
    gateway.reject(context, operation["operation_id"], "用户撤销")
    with pytest.raises(BusinessError, match="待确认"):
        gateway.operation_content(context, operation)
