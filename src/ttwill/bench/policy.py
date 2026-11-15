"""提取固定零售政策的原文章节及本项目不适用的 τ² 协议条款。"""

import re

PROTOCOL_EXCLUSIONS = (
    {
        "id": "tau2.single_tool_call",
        "text": "You should at most make one tool call at a time, ",
        "reason": "该限制属于 τ² 工具协议；本项目内部由网关执行工具调用。",
    },
    {
        "id": "tau2.no_simultaneous_response",
        "text": (
            "and if you take a tool call, you should not respond to the user at the same time. "
            "If you respond to the user, you should not make a tool call at the same time."
        ),
        "reason": "本项目内部工具调用不进入 τ² 对话轨迹，适配层每轮仅返回文本回复。",
    },
)


def extract_policy(policy: str) -> tuple[dict[str, str], list[dict]]:
    """按 Markdown 标题索引政策原文，并附上两个协议条款的排除清单。

    Args:
        policy: 固定中文数据包中的完整政策正文，不预先翻译或删改。

    Returns:
        按顺序返回两个值：
        - sections：按标题文本索引的原文片段，含标题及其后空白，止于下一个
          任意级标题；标题前的内容以空字符串为键，拼接所有值可还原原文。
        - exclusions：两项排除记录，每项包含 id、精确原文 text 和中文 reason。
          此函数不删除原文，由通用 Skill 加载器应用清单。

    Raises:
        ValueError: 政策没有标题、标题重复，或固定协议条款没有各出现一次。
    """
    headings = list(re.finditer(r"^#{1,6}[ \t]+([^\r\n]+?)[ \t]*\r?$", policy, re.MULTILINE))
    if not headings:
        raise ValueError("政策正文缺少 Markdown 标题")
    sections = {}
    if headings[0].start():
        sections[""] = policy[: headings[0].start()]
    for index, heading in enumerate(headings):
        name = re.sub(r"[ \t]+#+$", "", heading.group(1))
        if name in sections:
            raise ValueError(f"政策标题重复：{name}")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(policy)
        sections[name] = policy[heading.start() : end]
    for exclusion in PROTOCOL_EXCLUSIONS:
        if policy.count(exclusion["text"]) != 1:
            raise ValueError(f"政策协议条款与固定来源不一致：{exclusion['id']}")
    return sections, [dict(item) for item in PROTOCOL_EXCLUSIONS]
