+++
name = "cancel_pending_order"
description = "取消待处理订单，核对订单状态、取消原因和退款去向。"
capability = "order_service"
tools = ["cancel_pending_order"]
policy_sections = ["Cancel pending order"]
+++

## 操作要点

1. 查询订单，确认属于当前用户且状态为 `pending`；收集订单号与允许的取消原因。
2. 对同一订单的取消诉求汇总成一个任务。取消后不能继续修改该订单，冲突诉求交给协调者澄清。
3. 调用写工具只会准备操作。将取消内容、退款去向与时间要点交给协调者展示，等待用户逐项确认。
4. 根据工具实际返回报告准备状态，不把准备成功描述为已经取消或退款；提交由协调者确认护栏执行。
