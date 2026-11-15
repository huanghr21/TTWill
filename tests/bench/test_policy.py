"""验证固定零售政策的原文拆分、两条协议排除及 Skill 分配。"""

from pathlib import Path

import pytest

from ttwill.bench.policy import extract_policy
from ttwill.bench.retail import load_retail_bundle
from ttwill.gateway.skills import SkillCatalog

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy() -> str:
    """读取已校验哈希的真实中文包政策，不导入 τ² 运行依赖。"""
    return load_retail_bundle(ROOT / "data" / "source").policy


def test_every_heading_starts_an_exact_nonoverlapping_section(policy: str) -> None:
    """任意级标题均单独索引，按顺序拼接可逐字节还原原政策。"""
    sections, _ = extract_policy(policy)
    assert list(sections) == [
        "Retail agent policy",
        "Domain basic",
        "User",
        "Product",
        "Order",
        "Generic action rules",
        "Cancel pending order",
        "Modify pending order",
        "Modify payment",
        "Modify items",
        "Return delivered order",
        "Exchange delivered order",
    ]
    assert "".join(sections.values()).encode() == policy.encode()
    assert sections["Modify pending order"].startswith("## Modify pending order\n")
    assert "### Modify payment" not in sections["Modify pending order"]
    assert sections["Modify payment"].startswith("### Modify payment\n")


def test_exclusion_covers_both_protocol_rules_and_the_reverse_wording(policy: str) -> None:
    """两项清单覆盖完整协议段，包括回复时不可同时调用工具的反向表述。"""
    sections, exclusions = extract_policy(policy)
    assert len(exclusions) == 2
    assert len({item["id"] for item in exclusions}) == 2
    paragraph = next(paragraph for paragraph in policy.split("\n\n") if "at most" in paragraph)
    assert "".join(item["text"] for item in exclusions) == paragraph
    assert all(set(item) == {"id", "text", "reason"} for item in exclusions)
    catalog = SkillCatalog(ROOT / "skills", sections, exclusions)
    original_general = sections["Retail agent policy"]
    assert original_general.replace(paragraph, "") in catalog.common_policy
    assert "If you respond to the user" not in catalog.common_policy
    assert "explicit user confirmation" in catalog.common_policy
    assert "only help one user per conversation" in catalog.common_policy


def test_common_and_loaded_policy_preserve_every_applicable_character(policy: str) -> None:
    """常驻与按需内容共同保留全部业务原文，仅移除明确排除的两条协议。"""
    sections, exclusions = extract_policy(policy)
    catalog = SkillCatalog(ROOT / "skills", sections, exclusions)
    referenced = set()
    for capability in ("order_service", "after_sales"):
        for name in catalog.names(capability):
            loaded = catalog.load(capability, name)
            expected = "".join(sections[section] for section in loaded["policy_sections"])
            assert loaded["content"].startswith(expected)
            assert loaded["content"][len(expected) :].lstrip("\n").startswith("## 操作要点")
            referenced.update(loaded["policy_sections"])
    expected_common = "".join(text for name, text in sections.items() if name not in referenced)
    expected_filtered = policy
    for exclusion in exclusions:
        expected_common = expected_common.replace(exclusion["text"], "")
        expected_filtered = expected_filtered.replace(exclusion["text"], "")
    assert catalog.common_policy.encode() == expected_common.encode()
    assert len(expected_filtered) == len(expected_common) + sum(
        len(sections[name]) for name in referenced
    )
    assert sections["Generic action rules"] in catalog.common_policy
    assert sections["User"] in catalog.common_policy
    assert sections["Modify items"] not in catalog.common_policy


def test_policy_extraction_preserves_prefix_and_line_endings(policy: str) -> None:
    """标题前导内容与 CRLF 均保留，索引不标准化政策原文。"""
    original = "前导说明\r\n" + policy.replace("\n", "\r\n")
    sections, _ = extract_policy(original)
    assert sections[""] == "前导说明\r\n"
    assert "".join(sections.values()) == original


def test_changed_protocol_text_is_reported(policy: str) -> None:
    """协议条款变化必须更新来源专属清单，不能静默略过排除。"""
    with pytest.raises(ValueError, match="固定来源不一致"):
        extract_policy(policy.replace("at most make one tool call", "make many tool calls"))


def test_duplicate_section_name_is_reported(policy: str) -> None:
    """标题重复时显式报错，避免字典覆盖导致业务政策丢失。"""
    with pytest.raises(ValueError, match="政策标题重复"):
        extract_policy(policy + "\n## Modify items\n新增文字。\n")
