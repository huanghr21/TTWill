"""定义任务分派所需的能力与子 Agent 运行循环。"""

from ttwill.core.capabilities import Capability, CapabilityRegistry
from ttwill.core.subagent import DispatchTask, SubAgentRunner

__all__ = ["Capability", "CapabilityRegistry", "DispatchTask", "SubAgentRunner"]
