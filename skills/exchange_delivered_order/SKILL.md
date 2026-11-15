+++
name = "exchange_delivered_order"
description = "一次性准备已送达订单的全部换货商品，核对规格、差价与支付方式。"
capability = "after_sales"
tools = ["exchange_delivered_order_items"]
policy_sections = ["Exchange delivered order"]
+++

## 操作要点

1. 查询订单和商品，确认订单为 `delivered`；目标商品必须是同一产品下不同且可用的规格。
2. 按“订单 × 换货”汇总完整清单，原商品与目标商品逐项对应；提醒协调者确认用户已提供所有要换的商品，再一次性准备，不能拆成多次调用。
3. 收集用于支付或接收差价的支付方式，使用礼品卡时核对余额；将完整换货清单和差价要点交给协调者展示。
4. 写工具只准备换货申请，等待协调者获得确认后提交；不得宣称申请已提交或换货已完成，也不要求用户另下新单。
