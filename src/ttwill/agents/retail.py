"""声明订单服务与售后能力的职责、工具范围和 Skill 目录。"""

from ttwill.core.capabilities import Capability, CapabilityRegistry
from ttwill.gateway import CAPABILITY_TOOLS
from ttwill.gateway.skills import SkillCatalog


def build_registry(skills: SkillCatalog) -> CapabilityRegistry:
    """注册共享工具循环的两种零售能力。

    Args:
        skills: 已加载的政策与 Skill 目录，提供各能力的可用 Skill 名称。

    Returns:
        含 order_service 和 after_sales 的能力注册表；工具集合引用网关定义。
    """
    registry = CapabilityRegistry()
    for name, description, prompt in (
        (
            "order_service",
            "查询订单和用户资料，取消或修改待发货订单，修改用户地址。",
            "你负责订单服务。根据任务先查询真实订单与用户资料，按需加载相关 Skill，"
            "补齐操作所需信息后准备操作。汇总同一订单同类商品后一次准备。"
            "修改商品会限制后续修改，按 Skill 安排顺序。",
        ),
        (
            "after_sales",
            "查询已送达订单，判断退换货条件并准备整单所需的退货或换货申请。",
            "你负责退换货。查询订单和商品事实，加载对应 Skill 后判断可办条件。"
            "同一订单中同类诉求的商品先汇总，再一次准备退货或换货。"
            "缺少商品清单、退款去向或替换规格时明确报告所缺信息。",
        ),
    ):
        registry.register(
            Capability(
                name=name,
                description=description,
                prompt=prompt,
                skills=tuple(entry["name"] for entry in skills.catalog(name)),
                tools=CAPABILITY_TOOLS[name],
            )
        )
    return registry
