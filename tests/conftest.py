"""为业务、网关和命令行测试提供独立的小型 SQLite 数据库。"""

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest

from ttwill.backend import RetailBackend
from ttwill.gateway.skills import SkillCatalog


@pytest.fixture
def skill_catalog() -> SkillCatalog:
    """用项目 Skill 声明和普通测试政策构造静态目录。

    Returns:
        两种业务能力共用的静态目录；实例加载状态由各测试中的网关保存。
    """
    sections = {
        name: f"# {name}\n测试政策：核对信息后准备操作。\n\n"
        for name in (
            "General policy",
            "Cancel pending order",
            "Modify pending order",
            "Modify payment",
            "Modify items",
            "Return delivered order",
            "Exchange delivered order",
        )
    }
    return SkillCatalog(Path(__file__).resolve().parents[1] / "skills", sections, [])


@pytest.fixture
def seed_snapshot() -> dict:
    """构造两个用户、两类商品及待发货和已送达订单。

    Returns:
        包含 users、products、orders 三张实体映射的独立业务快照。
    """
    address = dict(
        address1="原地址", address2="", city="上海", state="上海", country="中国", zip="200000"
    )
    products = {
        "p1": {
            "name": "衬衫",
            "product_id": "p1",
            "variants": {
                "a": {"item_id": "a", "price": 20.0, "available": True, "options": {"颜色": "蓝"}},
                "b": {"item_id": "b", "price": 30.0, "available": True, "options": {"颜色": "红"}},
            },
        },
        "p2": {
            "name": "鞋",
            "product_id": "p2",
            "variants": {
                "x": {"item_id": "x", "price": 40.0, "available": True, "options": {"尺码": "小"}},
                "y": {"item_id": "y", "price": 60.0, "available": True, "options": {"尺码": "大"}},
            },
        },
    }
    users = {
        user_id: {
            "user_id": user_id,
            "name": {"first_name": "小明" if user_id == "u1" else "小红", "last_name": "王"},
            "email": f"{user_id}@example.com",
            "address": deepcopy(address),
            "payment_methods": {
                "card": {
                    "id": "card",
                    "source": "credit_card",
                    "brand": "visa",
                    "last_four": "1234",
                },
                "gift": {"id": "gift", "source": "gift_card", "balance": 300.0},
            },
            "orders": ["#W0000001", "#W0000002", "#W0000004"] if user_id == "u1" else ["#W0000003"],
        }
        for user_id in ("u1", "u2")
    }
    orders = {}
    for order_id, user_id, status in (
        ("#W0000001", "u1", "pending"),
        ("#W0000002", "u1", "delivered"),
        ("#W0000003", "u2", "pending"),
        ("#W0000004", "u1", "pending"),
    ):
        orders[order_id] = {
            "order_id": order_id,
            "user_id": user_id,
            "status": status,
            "address": deepcopy(address),
            "items": [
                {
                    "name": product["name"],
                    "product_id": product_id,
                    "item_id": item_id,
                    "price": product["variants"][item_id]["price"],
                    "options": deepcopy(product["variants"][item_id]["options"]),
                }
                for product_id, item_id in (("p1", "a"), ("p2", "x"))
                for product in (products[product_id],)
            ],
            "fulfillments": [],
            "payment_history": [
                {"transaction_type": "payment", "amount": 60.0, "payment_method_id": "card"}
            ],
        }
    return {"users": users, "products": products, "orders": orders}


@pytest.fixture
def backend(tmp_path: Path, seed_snapshot: dict) -> Iterator[RetailBackend]:
    """创建每项测试专用的业务库，并在测试结束后关闭连接池。

    Args:
        tmp_path: 本项测试的临时目录。
        seed_snapshot: 导入新库的初始业务快照。

    Yields:
        已初始化的本地业务后端。
    """
    instance = RetailBackend(tmp_path / "business.sqlite")
    instance.initialize(seed_snapshot)
    yield instance
    instance.close()
