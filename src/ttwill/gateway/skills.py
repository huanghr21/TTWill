"""读取业务 Skill 声明，并按能力提供目录、政策原文和操作要点。"""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Skill:
    """保存一个已解析的 Skill 声明及其自有操作要点。"""

    name: str
    description: str
    capability: str
    tools: tuple[str, ...]
    policy_sections: tuple[str, ...]
    body: str


class SkillCatalog:
    """提供静态 Skill 资源；实例加载记录由工具网关管理。

    Attributes:
        common_policy: 未被任何 Skill 引用的适用政策原文，按源文件顺序拼接。
    """

    def __init__(
        self, directory: Path, policy_sections: dict[str, str], exclusions: list[dict]
    ) -> None:
        """解析 Skill 文件，核对章节引用并排除显式列出的协议原文。

        Args:
            directory: 由启动入口解析的 Skill 根目录，每个子目录包含 SKILL.md。
            policy_sections: 按原文顺序保存的章节索引，键为标题文本，值含原始标题。
            exclusions: 待排除条款列表，每项须含唯一 id、精确 text 和非空 reason。

        Raises:
            FileNotFoundError: Skill 根目录不存在。
            ValueError: 元数据或排除清单无效、引用章节缺失，或写工具被重复绑定。
        """
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        self._sections = dict(policy_sections)
        self._skills: dict[str, _Skill] = {}
        self._tool_skills: dict[str, str] = {}
        seen_exclusions = set()
        for exclusion in exclusions:
            if not all(
                isinstance(exclusion.get(key), str) and exclusion[key].strip()
                for key in ("id", "text", "reason")
            ):
                raise ValueError("排除条款须包含非空 id、text 和 reason")
            if exclusion["id"] in seen_exclusions:
                raise ValueError(f"排除条款 ID 重复：{exclusion['id']}")
            seen_exclusions.add(exclusion["id"])
            text = exclusion["text"]
            if sum(section.count(text) for section in self._sections.values()) != 1:
                raise ValueError(f"排除条款必须在政策中恰好出现一次：{exclusion['id']}")
            self._sections = {
                name: section.replace(text, "") for name, section in self._sections.items()
            }
        for path in sorted(directory.glob("*/SKILL.md")):
            skill = self._read(path)
            if skill.name in self._skills:
                raise ValueError(f"Skill 名称重复：{skill.name}")
            for name in skill.policy_sections:
                if name not in self._sections:
                    raise ValueError(f"Skill {skill.name} 引用了未知政策章节：{name}")
            for tool in skill.tools:
                if tool in self._tool_skills:
                    raise ValueError(f"写工具被重复绑定：{tool}")
                self._tool_skills[tool] = skill.name
            self._skills[skill.name] = skill
        if not self._skills:
            raise ValueError("Skill 目录中没有可加载的资源")
        referenced = {
            section for skill in self._skills.values() for section in skill.policy_sections
        }
        self.common_policy = "".join(
            content for name, content in self._sections.items() if name not in referenced
        )

    @staticmethod
    def _read(path: Path) -> _Skill:
        """解析一个 Skill 文件的 TOML 元数据及正文。

        Args:
            path: 形如 <skill_name>/SKILL.md 的资源文件。

        Returns:
            已核对名称、能力、工具列表和章节列表的 Skill 记录。

        Raises:
            ValueError: 文件没有 +++ 前导分隔、元数据字段无效，或名称与目录不符。
        """
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines or lines[0].strip() != "+++":
            raise ValueError(f"Skill 缺少 TOML 前导元数据：{path}")
        end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "+++"), None)
        if end is None:
            raise ValueError(f"Skill 前导元数据未结束：{path}")
        metadata = tomllib.loads("".join(lines[1:end]))
        fields = {"name", "description", "capability", "tools", "policy_sections"}
        if set(metadata) != fields:
            raise ValueError(f"Skill 元数据字段须为 {sorted(fields)}：{path}")
        if not all(
            isinstance(metadata[key], str) and metadata[key].strip()
            for key in ("name", "description", "capability")
        ):
            raise ValueError(f"Skill 名称、描述和能力须为非空文本：{path}")
        if metadata["name"] != path.parent.name:
            raise ValueError(f"Skill 名称须与目录名一致：{path}")
        if metadata["capability"] not in {"order_service", "after_sales"}:
            raise ValueError(f"未知 Skill 能力：{metadata['capability']}")
        for key in ("tools", "policy_sections"):
            values = metadata[key]
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"Skill {key} 须为文本列表：{path}")
            if len(values) != len(set(values)):
                raise ValueError(f"Skill {key} 存在重复项：{path}")
        if not metadata["tools"]:
            raise ValueError(f"Skill 必须声明关联写工具：{path}")
        body = "".join(lines[end + 1 :]).lstrip("\r\n")
        if not body.strip():
            raise ValueError(f"Skill 缺少操作要点：{path}")
        return _Skill(
            name=metadata["name"],
            description=metadata["description"],
            capability=metadata["capability"],
            tools=tuple(metadata["tools"]),
            policy_sections=tuple(metadata["policy_sections"]),
            body=body,
        )

    def names(self, capability: str) -> tuple[str, ...]:
        """列出当前能力允许加载的 Skill 名称。

        Args:
            capability: order_service、after_sales 或没有业务 Skill 的 coordinator。

        Returns:
            按名称排序的 Skill 名称元组；协调者返回空元组。

        Raises:
            ValueError: 能力名称未知。
        """
        if capability not in {"order_service", "after_sales", "coordinator"}:
            raise ValueError(f"未知能力：{capability}")
        return tuple(name for name, skill in self._skills.items() if skill.capability == capability)

    def catalog(self, capability: str) -> list[dict]:
        """提供能力提示词使用的 Skill 名称与一句话描述。

        Args:
            capability: 要列出目录的能力名称。

        Returns:
            目录列表，每项仅包含 name 和 description，不包含政策正文。

        Raises:
            ValueError: 能力名称未知。
        """
        return [
            {"name": name, "description": self._skills[name].description}
            for name in self.names(capability)
        ]

    def load(self, capability: str, name: str) -> dict:
        """加载本能力已注册的 Skill，拼接原文章节和项目操作要点。

        Args:
            capability: 发起加载的能力名称。
            name: 目录中的 Skill 名称；不接受文件路径。

        Returns:
            含 name、description、tools、policy_sections 和 content 的新字典；
            content 依次包含声明引用的政策原文及独立的中文操作要点。

        Raises:
            ValueError: 能力或 Skill 未知，或 Skill 不属于当前能力。
        """
        allowed = self.names(capability)
        if name not in self._skills:
            raise ValueError(f"未知 Skill：{name}")
        if name not in allowed:
            raise ValueError(f"当前能力不能加载 Skill：{name}")
        skill = self._skills[name]
        policy = "".join(self._sections[section] for section in skill.policy_sections)
        separator = "\n\n" if policy and not policy.endswith("\n\n") else ""
        return {
            "name": skill.name,
            "description": skill.description,
            "tools": list(skill.tools),
            "policy_sections": list(skill.policy_sections),
            "content": policy + separator + skill.body,
        }

    def required_skill(self, tool: str) -> str:
        """查找写工具所要求的唯一 Skill。

        Args:
            tool: Skill 声明绑定的业务写工具名称。

        Returns:
            调用工具前需要加载的 Skill 名称。

        Raises:
            ValueError: 该工具没有绑定的 Skill。
        """
        if tool not in self._tool_skills:
            raise ValueError(f"写工具未绑定 Skill：{tool}")
        return self._tool_skills[tool]
