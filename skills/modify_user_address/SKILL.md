+++
name = "modify_user_address"
description = "修改当前用户的默认地址，明确其与订单收货地址的区别。"
capability = "order_service"
tools = ["modify_user_address"]
policy_sections = []
+++

## 操作要点

1. 根据常驻通用政策处理当前已验证用户的默认地址；先查询用户资料并收集完整新地址。
2. 默认地址的修改不更新已有订单的收货地址。若用户同时要求修改订单地址，将两类诉求分别交给协调者安排。
3. 写工具只准备操作；将完整地址交给协调者展示并等待确认，不宣称地址已经更新。
