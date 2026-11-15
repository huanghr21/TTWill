+++
name = "return_delivered_order"
description = "准备已送达订单的退货，核对完整商品清单与退款方式。"
capability = "after_sales"
tools = ["return_delivered_order_items"]
policy_sections = ["Return delivered order"]
+++

## 操作要点

1. 查询订单，确认属于当前用户且状态为 `delivered`；按“订单 × 退货”汇总要退的完整商品清单。
2. 收集订单号、所有退货商品及退款方式。退款仅可退至原支付方式或当前用户已有礼品卡。
3. 同一订单还有换货等其他诉求时，将冲突或条件依赖交给协调者处理；一次准备一种动作，分别确认。
4. 写工具只准备退货申请，将完整清单和退款去向交给协调者展示并等待确认。不得宣称已经提交申请、收到退货或完成退款。
