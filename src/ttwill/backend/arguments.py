"""业务工具的参数结构，供后端校验和网关生成模型工具定义。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Arguments(BaseModel):
    """按原字段类型接收一次业务请求。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class EmailArguments(Arguments):
    """按邮箱查找唯一用户，并验证当前会话的身份。"""

    email: str = Field(description="用户邮箱")


class NameZipArguments(Arguments):
    """按姓名和邮编查找唯一用户，并验证当前会话的身份。"""

    first_name: str = Field(description="名")
    last_name: str = Field(description="姓")
    zip: str = Field(description="邮政编码")


class OrderArguments(Arguments):
    """读取订单详情。"""

    order_id: str = Field(description="订单 ID，包含开头的 #")


class ProductArguments(Arguments):
    """读取商品及其规格详情。"""

    product_id: str = Field(description="商品 ID")


class ItemArguments(Arguments):
    """读取单个商品规格详情。"""

    item_id: str = Field(description="商品规格 ID")


class UserArguments(Arguments):
    """读取当前已验证用户的账户详情。"""

    user_id: str = Field(description="已验证用户的 ID")


class CatalogArguments(Arguments):
    """列出商品名称与商品 ID 的对应关系。"""


class CancelArguments(OrderArguments):
    """准备取消待发货订单，返回待确认详情。"""

    reason: Literal["no longer needed", "ordered by mistake"] = Field(
        description="取消原因：不再需要或误下单，使用对应英文值"
    )


class AddressArguments(Arguments):
    """地址的各组成字段。"""

    address1: str = Field(description="地址第一行")
    address2: str = Field(description="地址第二行，无内容时传空字符串")
    city: str = Field(description="城市")
    state: str = Field(description="省或州")
    country: str = Field(description="国家")
    zip: str = Field(description="邮政编码")


class OrderAddressArguments(AddressArguments, OrderArguments):
    """准备修改待发货订单的收货地址，返回待确认详情。"""


class UserAddressArguments(AddressArguments, UserArguments):
    """准备修改用户默认地址，返回待确认详情。"""


class PaymentArguments(OrderArguments):
    """准备修改待发货订单的支付方式，返回待确认详情。"""

    payment_method_id: str = Field(description="当前用户已有的支付方式 ID")


class ReturnArguments(PaymentArguments):
    """准备已送达订单的退货申请，返回待确认详情。"""

    item_ids: list[str] = Field(
        min_length=1, description="本次办理的完整商品规格清单，重复商品重复列出"
    )


class ExchangeArguments(ReturnArguments):
    """准备已送达订单的换货申请，返回待确认详情。"""

    new_item_ids: list[str] = Field(
        min_length=1, description="新规格清单，与 item_ids 按位置逐项对应"
    )


class ModifyItemsArguments(ExchangeArguments):
    """准备一次性修改待发货订单的商品规格，返回待确认详情。"""


ARGUMENT_MODELS: dict[str, type[Arguments]] = {
    "find_user_id_by_email": EmailArguments,
    "find_user_id_by_name_zip": NameZipArguments,
    "get_order_details": OrderArguments,
    "get_product_details": ProductArguments,
    "get_item_details": ItemArguments,
    "get_user_details": UserArguments,
    "list_all_product_types": CatalogArguments,
    "cancel_pending_order": CancelArguments,
    "modify_pending_order_address": OrderAddressArguments,
    "modify_pending_order_items": ModifyItemsArguments,
    "modify_pending_order_payment": PaymentArguments,
    "modify_user_address": UserAddressArguments,
    "exchange_delivered_order_items": ExchangeArguments,
    "return_delivered_order_items": ReturnArguments,
}


def validate_arguments(name: str, args: dict) -> dict:
    """按工具对应的参数模型校验请求。

    Args:
        name: ARGUMENT_MODELS 中登记的业务工具名称。
        args: 本次请求的参数字典，须包含模型要求的字段且不得添加额外字段。

    Returns:
        校验后导出的普通字典，保留参数模型定义的字段和类型。

    Raises:
        ValueError: name 未登记，或参数的字段、类型、取值不满足模型约束；
            参数校验错误的具体类型为 pydantic.ValidationError。
    """
    if name not in ARGUMENT_MODELS:
        raise ValueError(f"未知业务工具：{name}")
    return ARGUMENT_MODELS[name].model_validate(args).model_dump()
