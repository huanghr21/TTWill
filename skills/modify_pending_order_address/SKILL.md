+++
name = "modify_pending_order_address"
description = "修改待处理订单的收货地址，先核对完整地址与动作顺序。"
capability = "order_service"
tools = ["modify_pending_order_address"]
policy_sections = ["Modify pending order"]
+++

## 操作要点

1. 查询订单，确认属于当前用户且状态为 `pending`；收集收货地址的全部字段，缺少信息时返回给协调者追问。
2. 订单收货地址与用户默认地址分别办理；修改一个不代表另一个已更新。
3. 同一订单还要修改商品时，先准备并确认地址、支付方式的修改，再准备商品修改。商品修改提交后不可再次修改或取消订单。
4. 写工具只准备操作，将完整新地址交给协调者展示并等待确认；不得把准备结果称为已提交或已改好地址。
