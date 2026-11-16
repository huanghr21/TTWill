"""保存预设能力的提示词、工具和 Skill 范围。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    """描述一种可分派的子 Agent 能力。

    Attributes:
        name: 能力名称，同时对应网关的能力白名单。
        description: 给协调者选择能力时使用的职责说明。
        prompt: 每个新实例使用的固定系统提示词。
        skills: 当前能力可以按需加载的 Skill 名称。
        tools: 当前能力允许调用的网关工具名称集合，包括 load_skill。
        model_tier: 默认模型档位；派发时可以覆盖模型档位。
    """

    name: str
    description: str
    prompt: str
    skills: tuple[str, ...]
    tools: frozenset[str]
    model_tier: str = "flash"


class CapabilityRegistry:
    """按名称注册固定能力，供每次派发创建独立实例。"""

    def __init__(self) -> None:
        """创建空的能力注册表。"""
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        """添加一种能力，保留已有能力定义。

        Args:
            capability: 具有唯一名称的能力定义。

        Raises:
            ValueError: 能力名称为空，或名称已被注册。
        """
        if not capability.name.strip():
            raise ValueError("能力名称不能为空")
        if capability.name in self._capabilities:
            raise ValueError(f"能力已注册：{capability.name}")
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Capability:
        """按名称读取能力定义。

        Args:
            name: 已注册的能力名称。

        Returns:
            对应的不可变能力定义。

        Raises:
            ValueError: 能力尚未注册。
        """
        if name not in self._capabilities:
            raise ValueError(f"未知能力：{name}")
        return self._capabilities[name]

    def catalog(self) -> list[dict]:
        """列出协调者可选择的能力及业务工具。

        Returns:
            按注册顺序排列的能力名称、职责说明与工具名称，不含实例提示词。
        """
        return [
            {
                "name": capability.name,
                "description": capability.description,
                "tools": sorted(capability.tools - {"load_skill"}),
            }
            for capability in self._capabilities.values()
        ]
