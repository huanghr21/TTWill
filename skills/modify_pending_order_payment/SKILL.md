+++
name = "modify_pending_order_payment"
description = "更换待处理订单的支付方式，核对支付账户与礼品卡余额。"
capability = "order_service"
tools = ["modify_pending_order_payment"]
policy_sections = ["Modify pending order", "Modify payment"]
+++

## 操作要点

1. 查询订单和当前用户的支付方式，确认订单为 `pending`，新支付方式与原方式不同且属于当前用户。
2. 只能使用一个新支付方式；使用礼品卡时核对余额能否覆盖订单总额。
3. 同一订单同时改地址、支付方式和商品时，先逐个办理地址与支付方式并等待各自确认，再准备完整的商品修改清单。
4. 调用写工具只准备操作；向协调者提供新支付方式、退款去向和时间要点，等待确认后提交，不宣称已扣款或退款。
