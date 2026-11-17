"""用固定官方数据离线验证转换、错误身份语义与不可覆盖的开发集产物。"""

import hashlib
import json
import re
import shutil
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from ttwill.bench import conversion

pytest.importorskip("tau2")

SOURCE = Path(__file__).resolve().parents[2] / "data/source"


@pytest.fixture(scope="module")
def compiled() -> dict:
    """编译同一份来源，让本模块共享经过真实官方编译器验证的开发题。

    Returns:
        完整开发题、逐字段校验报告及文本对照。
    """
    return conversion.compile_development_tasks(SOURCE)


def test_same_converter_matches_all_official_tasks_and_builds_exact_complement(
    compiled: dict,
) -> None:
    """同一转换循环须逐字段重建五十题，再得到互斥且完整的六十四题。"""
    report = compiled["report"]
    assert report["source_compilation_50_matches"] == 50
    assert report["official_50"]["matched"] == report["official_50"]["total"] == 50
    assert all(not row["differences"] for row in report["official_50"]["tasks"])
    assert report["database_differences"] == report["policy_differences"] == []
    assert report["unresolved"] == []
    source_tasks = json.loads((SOURCE / "tau2_retail_tasks.json").read_text())
    subset = set(json.loads((SOURCE / "retail_50.json").read_text())["task_ids"])
    expected = {task["id"] for task in source_tasks} - subset
    actual = {row["canonical_task_id"] for row in compiled["dataset"]["metadata"].values()}
    assert actual == expected
    assert len(actual) == 64


def test_conversion_preserves_reference_order_identifiers_and_reward_axes(compiled: dict) -> None:
    """本地化可改实体值，必须保留参考动作顺序、机器标识与奖励依据。"""
    original = {
        task["id"]: task for task in json.loads((SOURCE / "tau2_retail_tasks.json").read_text())
    }
    for task in compiled["dataset"]["tasks"]:
        source = original[task["id"].removesuffix("_zh_identity")]
        assert task["initial_state"] == source["initial_state"]
        assert (
            task["evaluation_criteria"]["reward_basis"]
            == source["evaluation_criteria"]["reward_basis"]
        )
        actions = task["evaluation_criteria"]["actions"]
        before = source["evaluation_criteria"]["actions"]
        assert [(row["action_id"], row["name"]) for row in actions] == [
            (row["action_id"], row["name"]) for row in before
        ]
        for left, right in zip(before, actions, strict=True):
            for field in (
                "order_id",
                "product_id",
                "item_ids",
                "new_item_ids",
                "payment_method_id",
            ):
                assert left["arguments"].get(field) == right["arguments"].get(field)


@pytest.mark.parametrize("task_id", ["35", "54", "55", "67"])
def test_wrong_identity_attempts_preserve_success_and_failure(task_id: str, compiled: dict) -> None:
    """错误邮箱与错误邮编仍失败，后续正确身份仍能找到同一个本地化用户。"""
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools

    source = json.loads((SOURCE / "tau2_retail_database.json").read_text())
    localized = json.loads((SOURCE / "tau2_zh_retail.json").read_text())["database"]
    before = next(
        task
        for task in json.loads((SOURCE / "tau2_retail_tasks.json").read_text())
        if task["id"] == task_id
    )
    after = next(
        task for task in compiled["dataset"]["tasks"] if task["id"] == task_id + "_zh_identity"
    )
    aliases = {
        row["canonical"]: row["localized"]
        for row in json.loads((SOURCE / "tau2_zh_retail_localization.json").read_text())[
            "user_id_revision"
        ]["aliases"]
    }
    outcomes = []
    for old, new in zip(
        before["evaluation_criteria"]["actions"],
        after["evaluation_criteria"]["actions"],
        strict=True,
    ):
        if old["name"] not in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
            continue
        values = []
        for database, action in ((source, old), (localized, new)):
            tools = RetailTools(RetailDB.model_validate(database))
            try:
                values.append(getattr(tools, action["name"])(**action["arguments"]))
            except ValueError:
                values.append(None)
        assert values[1] == (aliases[values[0]] if values[0] is not None else None)
        outcomes.append(values[0] is not None)
    assert outcomes[0] is False
    assert outcomes[-1] is True


def test_money_counts_tracking_numbers_and_units_remain_distinct(compiled: dict) -> None:
    """金额统一乘七，同时保留件数、物流号和容量等非金额数字。"""
    tasks = {task["id"]: task for task in compiled["dataset"]["tasks"]}
    assert tasks["2_zh_identity"]["evaluation_criteria"]["communicate_info"] == ["10"]
    assert tasks["29_zh_identity"]["evaluation_criteria"]["communicate_info"] == [
        "1260.7",
        "1326.99",
        "1460.2",
    ]
    assert "8050" in tasks["37_zh_identity"]["user_scenario"]["instructions"]["reason_for_call"]
    assert "21000" in tasks["90_zh_identity"]["user_scenario"]["instructions"]["reason_for_call"]
    text = json.dumps(tasks, ensure_ascii=False)
    assert "840887978435" in text
    assert "64GB" in text


def test_entity_replacement_does_not_corrupt_email_or_ordinary_prepositions() -> None:
    """姓名替换不能改错邮箱的局部，州码也不能命中普通介词。"""
    text = "Sofia lives in IN. sofia.thomas3019@example.com, sofia.thomas3069@example.com"
    actual = conversion._replace_text(
        text,
        {
            "Sofia": "雪",
            "IN": "河南省",
            "sofia.thomas3069@example.com": "xue.shi3069@example.com",
        },
    )
    assert actual == "雪 lives in 河南省. sofia.thomas3019@example.com, xue.shi3069@example.com"


def test_product_option_words_do_not_rewrite_unrelated_meanings(compiled: dict) -> None:
    """相机和全额退款不能被解释成包内隔层或全拉链选项。"""
    tasks = {task["id"]: task for task in compiled["dataset"]["tasks"]}
    for task_id in ("51_zh_identity", "98_zh_identity", "99_zh_identity"):
        assert "相机仓" not in tasks[task_id]["user_scenario"]["instructions"]["reason_for_call"]
    assert (
        "全拉链 refund"
        not in tasks["69_zh_identity"]["user_scenario"]["instructions"]["reason_for_call"]
    )


def test_currency_replacement_preserves_following_punctuation() -> None:
    """金额后的句法逗号必须保留，千位分隔符只属于完整分组。"""
    actual = re.sub(
        conversion.MONEY_PATTERN,
        lambda match: "CNY " + conversion._scaled(match[1], Decimal(7)),
        "If greater than $300, compare with $1,126.04.",
    )
    assert actual == "If greater than CNY 2100, compare with CNY 7882.28."


def test_ambiguous_caller_name_uses_explicit_postal_code(compiled: dict) -> None:
    """无参考动作的同名用户题按题目明示邮编确定身份。"""
    task = next(task for task in compiled["dataset"]["tasks"] if task["id"] == "24_zh_identity")
    known = task["user_scenario"]["instructions"]["known_info"]
    assert "98193" not in known
    assert "Name details:" in known


def test_gender_metadata_only_reuses_official_identity_labels(compiled: dict) -> None:
    """已有官方标签按用户复用，缺少标签的题显式保留未知。"""
    metadata = compiled["dataset"]["metadata"]
    unknown = compiled["report"]["unknown_caller_gender"]
    assert len(unknown) == 28
    assert set(unknown) == {
        task_id for task_id, row in metadata.items() if row["caller_gender"] is None
    }
    assert metadata["0_zh_identity"]["caller_gender"] is None
    assert metadata["7_zh_identity"]["caller_gender"] == "female"


@pytest.mark.parametrize("filename", list(conversion.CONVERSION_FILES))
def test_conversion_refuses_changed_original_source(filename: str, tmp_path: Path) -> None:
    """每份新增官方来源均须按原始字节哈希校验后才参与转换。"""
    shutil.copytree(SOURCE, tmp_path / "source")
    with (tmp_path / "source" / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="来源文件校验失败"):
        conversion.compile_development_tasks(tmp_path / "source")


def test_conversion_refuses_installed_upstream_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编译器代码也须来自数据清单固定的官方提交。"""
    monkeypatch.setattr(
        conversion, "distribution", lambda _: SimpleNamespace(read_text=lambda _: "{}")
    )
    with pytest.raises(ValueError, match="已安装 τ² 的提交与数据来源不一致"):
        conversion.compile_development_tasks(SOURCE)


def test_field_comparison_distinguishes_missing_null_and_scalar_types() -> None:
    """缺失字段、显式空值和布尔数值类型均不能被宽松相等吞掉。"""
    rows = conversion.field_differences({"flag": True}, {"flag": 1, "missing": None})
    assert {(tuple(row["path"]), row["kind"]) for row in rows} == {
        (("flag",), "type"),
        (("missing",), "missing_field"),
    }


def test_build_writes_checked_dataset_once_and_preserves_existing_artifacts(
    compiled: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """成功构建记录来源与审阅哈希，重复构建不能覆盖既有固定题集。"""
    monkeypatch.setattr(conversion, "compile_development_tasks", lambda *args: deepcopy(compiled))
    report = conversion.build_development_tasks(SOURCE, tmp_path)
    assert report["official_50"]["matched"] == 50
    assert json.loads((tmp_path / "dev_64.json").read_text()) == compiled["dataset"]
    before = hashlib.sha256((tmp_path / "dev_64.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        conversion.build_development_tasks(SOURCE, tmp_path)
    assert hashlib.sha256((tmp_path / "dev_64.json").read_bytes()).hexdigest() == before


def test_failed_conversion_only_writes_explicit_difference_report(
    compiled: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未解决的转换语义必须阻止开发题发布，同时留下可审阅的问题报告。"""
    failed = deepcopy(compiled)
    failed["report"]["unresolved"] = [{"canonical_task_id": "35", "reason": "错误邮箱待审阅"}]
    monkeypatch.setattr(conversion, "compile_development_tasks", lambda *args: failed)
    with pytest.raises(ValueError, match="仍有差异或未审阅字段"):
        conversion.build_development_tasks(SOURCE, tmp_path)
    assert not (tmp_path / "dev_64.json").exists()
    assert json.loads((tmp_path / "conversion_report.json").read_text())["unresolved"]


def test_review_cannot_change_reference_action_identifiers(tmp_path: Path) -> None:
    """本地化审阅只能补实体与文字，不得改动待执行的动作或订单标识。"""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "version": conversion.REVIEW_VERSION,
                "entries": [
                    {
                        "canonical_task_id": "0",
                        "path": ["evaluation_criteria", "actions", 4, "arguments", "order_id"],
                        "source": "#W2378156",
                        "native": "#OTHER",
                        "reason": "不可改动机器标识",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="不能修改机器标识"):
        conversion._review_entries(path)


def test_review_rejects_stale_source_before_replacing_text(
    compiled: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """字段路径相同而原文变化时不能继续套用旧审阅值。"""
    published = json.loads((SOURCE / "tau2_zh_retail.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    monkeypatch.setattr(
        conversion,
        "_official_compile",
        lambda _: (
            deepcopy(compiled["report"]),
            manifest,
            {"database": published["database"]},
        ),
    )
    review = json.loads((SOURCE.parent / "tasks/localization_overrides.json").read_text())
    review["entries"][0]["source"] += " 原文已变化"
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="本地化审阅原文已变化"):
        conversion.compile_development_tasks(SOURCE, path)


def test_reviewed_calculations_use_the_same_currency_as_prices_and_assertions(
    compiled: dict,
) -> None:
    """六条参考金额表达式须与商品价格和判分断言使用同一人民币比例。"""
    expected = {
        "28_zh_identity": [Decimal("6429.01")],
        "44_zh_identity": [Decimal("-125.93")],
        "45_zh_identity": [Decimal("69.23")],
        "46_zh_identity": [Decimal("7882.28"), Decimal("10483.55")],
        "61_zh_identity": [Decimal("211.26")],
    }
    actual = {}
    for task in compiled["dataset"]["tasks"]:
        expressions = [
            action["arguments"]["expression"]
            for action in task["evaluation_criteria"]["actions"]
            if action["name"] == "calculate"
        ]
        if expressions:
            actual[task["id"]] = [
                sum(
                    Decimal(term.replace(" ", ""))
                    for term in re.findall(r"[+-]?\s*\d+(?:\.\d+)?", expression)
                )
                for expression in expressions
            ]
    assert actual == expected


def test_expression_review_cannot_modify_another_tool(
    compiled: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """即使存在同名 expression 字段，也不能把 calculate 的审阅权限用于其他工具。"""
    source_dir = tmp_path / "source"
    shutil.copytree(SOURCE, source_dir)
    source_path = source_dir / "tau2_retail_tasks.json"
    tasks = json.loads(source_path.read_text())
    task = next(task for task in tasks if task["id"] == "28")
    action = next(
        action for action in task["evaluation_criteria"]["actions"] if action["name"] == "calculate"
    )
    action["name"] = "get_order_details"
    source_path.write_text(json.dumps(tasks))
    published = json.loads((SOURCE / "tau2_zh_retail.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    monkeypatch.setattr(
        conversion,
        "_official_compile",
        lambda _: (
            deepcopy(compiled["report"]),
            manifest,
            {"database": published["database"]},
        ),
    )
    with pytest.raises(ValueError, match="仅允许修改 calculate"):
        conversion.compile_development_tasks(
            source_dir, SOURCE.parent / "tasks/localization_overrides.json"
        )
