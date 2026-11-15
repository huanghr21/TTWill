"""验证通用 Skill 目录的能力边界、声明校验和原文组装。"""

import shutil
from pathlib import Path

import pytest

from ttwill.gateway.skills import SkillCatalog

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills"
SKILL_TOOLS = {
    "cancel_pending_order": "cancel_pending_order",
    "modify_pending_order_address": "modify_pending_order_address",
    "modify_pending_order_items": "modify_pending_order_items",
    "modify_pending_order_payment": "modify_pending_order_payment",
    "modify_user_address": "modify_user_address",
    "return_delivered_order": "return_delivered_order_items",
    "exchange_delivered_order": "exchange_delivered_order_items",
}


@pytest.fixture
def policy_sections() -> dict[str, str]:
    """提供普通原文章节索引，覆盖七个 Skill 所引用的全部标题。"""
    names = (
        "General",
        "Cancel pending order",
        "Modify pending order",
        "Modify payment",
        "Modify items",
        "Return delivered order",
        "Exchange delivered order",
    )
    return {name: f"## {name}\n\n原文：{name}。\n\n" for name in names}


@pytest.fixture
def skill_catalog(policy_sections: dict[str, str]) -> SkillCatalog:
    """用真实 Skill 资源和小型普通文本政策构建可加载目录。"""
    return SkillCatalog(SKILL_DIR, policy_sections, [])


def test_capability_catalog_exposes_only_names_and_descriptions(
    skill_catalog: SkillCatalog,
) -> None:
    """目录仅展示本能力名称与描述，不提前注入政策正文。"""
    order = skill_catalog.catalog("order_service")
    after_sales = skill_catalog.catalog("after_sales")
    assert len(order) == 5
    assert len(after_sales) == 2
    assert {item["name"] for item in order + after_sales} == set(SKILL_TOOLS)
    assert all(set(item) == {"name", "description"} for item in order + after_sales)
    assert skill_catalog.names("order_service") == tuple(sorted(item["name"] for item in order))
    assert skill_catalog.catalog("coordinator") == []


@pytest.mark.parametrize(("name", "tool"), SKILL_TOOLS.items())
def test_each_write_tool_has_one_skill(skill_catalog: SkillCatalog, name: str, tool: str) -> None:
    """七个业务写工具各自绑定一个 Skill，不依赖名称与工具名相同。"""
    capability = "after_sales" if name in skill_catalog.names("after_sales") else "order_service"
    loaded = skill_catalog.load(capability, name)
    assert skill_catalog.required_skill(tool) == name
    assert loaded["tools"] == [tool]
    assert "## 操作要点" in loaded["content"]
    assert "准备" in loaded["content"]
    assert "确认" in loaded["content"]


def test_load_retains_referenced_sections_and_separate_guidance(
    skill_catalog: SkillCatalog, policy_sections: dict[str, str]
) -> None:
    """商品修改同时加载父节和子节原文，通用政策保持常驻。"""
    loaded = skill_catalog.load("order_service", "modify_pending_order_items")
    policy = policy_sections["Modify pending order"] + policy_sections["Modify items"]
    assert loaded["policy_sections"] == ["Modify pending order", "Modify items"]
    assert loaded["content"].startswith(policy + "## 操作要点")
    assert "地址、支付方式" in loaded["content"]
    assert skill_catalog.common_policy == policy_sections["General"]
    assert policy_sections["General"] not in loaded["content"]


def test_default_address_does_not_invent_a_policy_section(skill_catalog: SkillCatalog) -> None:
    """默认地址使用通用授权与自有要点，不虚构官方专门章节。"""
    loaded = skill_catalog.load("order_service", "modify_user_address")
    assert loaded["policy_sections"] == []
    assert loaded["content"].startswith("## 操作要点")


@pytest.mark.parametrize(
    ("capability", "name", "message"),
    [
        ("after_sales", "cancel_pending_order", "当前能力不能加载"),
        ("order_service", "return_delivered_order", "当前能力不能加载"),
        ("coordinator", "cancel_pending_order", "当前能力不能加载"),
        ("unknown", "cancel_pending_order", "未知能力"),
        ("order_service", "unknown", "未知 Skill"),
        ("order_service", "../cancel_pending_order", "未知 Skill"),
        ("order_service", "cancel_pending_order/SKILL.md", "未知 Skill"),
        ("order_service", str(SKILL_DIR / "cancel_pending_order" / "SKILL.md"), "未知 Skill"),
    ],
)
def test_load_rejects_cross_capability_and_paths(
    skill_catalog: SkillCatalog, capability: str, name: str, message: str
) -> None:
    """调用方只能加载当前能力白名单中的名称，不能用路径读取文件。"""
    with pytest.raises(ValueError, match=message):
        skill_catalog.load(capability, name)


def test_unknown_tool_has_no_implicit_skill(skill_catalog: SkillCatalog) -> None:
    """未声明的工具不会通过名称猜测获得 Skill 绑定。"""
    with pytest.raises(ValueError, match="未绑定 Skill"):
        skill_catalog.required_skill("commit")


def test_loaded_metadata_is_not_shared(skill_catalog: SkillCatalog) -> None:
    """修改一次加载结果不会影响其他实例后续取得的元数据。"""
    first = skill_catalog.load("order_service", "cancel_pending_order")
    first["tools"].clear()
    first["policy_sections"].append("invented")
    second = skill_catalog.load("order_service", "cancel_pending_order")
    assert second["tools"] == ["cancel_pending_order"]
    assert second["policy_sections"] == ["Cancel pending order"]


def test_exclusion_removes_only_the_exact_text(policy_sections: dict[str, str]) -> None:
    """排除项只删除给定原文片段，不改写周边文字或输入字典。"""
    policy_sections["General"] = "# 通用\n保留前文。协议限制。保留后文。\n\n"
    exclusions = [{"id": "protocol", "text": "协议限制。", "reason": "本工具协议不适用"}]
    catalog = SkillCatalog(SKILL_DIR, policy_sections, exclusions)
    assert catalog.common_policy == "# 通用\n保留前文。保留后文。\n\n"
    assert "协议限制。" in policy_sections["General"]


@pytest.mark.parametrize("text", ["不存在的条款", "原文"])
def test_exclusion_must_match_once(policy_sections: dict[str, str], text: str) -> None:
    """排除文本缺失或多次出现时阻止加载，显式暴露来源变化。"""
    with pytest.raises(ValueError, match="恰好出现一次"):
        SkillCatalog(SKILL_DIR, policy_sections, [{"id": "test", "text": text, "reason": "说明"}])


def test_missing_referenced_section_is_reported(policy_sections: dict[str, str]) -> None:
    """专有章节缺失时不能返回没有该操作政策的 Skill。"""
    del policy_sections["Modify items"]
    with pytest.raises(ValueError, match="未知政策章节：Modify items"):
        SkillCatalog(SKILL_DIR, policy_sections, [])


def test_duplicate_write_binding_is_reported(
    tmp_path: Path, policy_sections: dict[str, str]
) -> None:
    """两个 Skill 不可同时声明为同一写工具的前置政策。"""
    directory = tmp_path / "skills"
    shutil.copytree(SKILL_DIR, directory)
    path = directory / "modify_user_address" / "SKILL.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'tools = ["modify_user_address"]', 'tools = ["cancel_pending_order"]'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="写工具被重复绑定"):
        SkillCatalog(directory, policy_sections, [])


def test_directory_name_must_match_declared_name(
    tmp_path: Path, policy_sections: dict[str, str]
) -> None:
    """目录名称和元数据名称一致，避免加载名称与文件定位混淆。"""
    directory = tmp_path / "skills"
    shutil.copytree(SKILL_DIR, directory)
    (directory / "cancel_pending_order").rename(directory / "different_name")
    with pytest.raises(ValueError, match="名称须与目录名一致"):
        SkillCatalog(directory, policy_sections, [])
