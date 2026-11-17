"""校验开发任务隔离、来源摘要和固定快速集的分层抽样。"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ttwill.bench.conversion import CONVERSION_VERSION
from ttwill.bench.retail import TAU2_COMMIT, RetailBundle
from ttwill.bench.tasks import QUICK_ALGORITHM, generate_quick_set, load_task_bundle


def _write_json(path: Path, value: Any) -> bytes:
    """保存 fixture JSON 并返回实际字节，供来源摘要验证使用。"""
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _read_development(tasks_dir: Path) -> dict[str, Any]:
    """读取 fixture 开发集，供各测试独立修改边界条件。"""
    return json.loads((tasks_dir / "dev_64.json").read_bytes())


@pytest.fixture
def task_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, RetailBundle]:
    """构造隔离的 114 题来源、官方 50 题及覆盖 14 层的开发 64 题。

    Args:
        tmp_path: 本例独占的临时目录。
        monkeypatch: 仅替换官方中文包读取，开发来源的校验仍执行真实文件读取。

    Returns:
        依次包含来源目录、开发任务目录与模拟官方中文包；加载器仍逐字节核对来源。
    """
    source_dir, tasks_dir = tmp_path / "source", tmp_path / "tasks"
    source_dir.mkdir()
    tasks_dir.mkdir()
    cancel = ["cancel_pending_order"]
    exchange = ["exchange_delivered_order_items"]
    address = ["modify_pending_order_address"]
    items = ["modify_pending_order_items"]
    user = ["modify_user_address"]
    returns = ["return_delivered_order_items"]
    populations = [
        (cancel, 3),
        (cancel * 2, 3),
        (exchange, 11),
        (exchange * 2, 3),
        (address + items, 6),
        (address + items + user, 10),
        (address, 1),
        (items, 7),
        (items * 2, 1),
        (user, 2),
        ([], 6),
        (returns, 9),
        (returns * 2, 1),
        (returns * 3, 1),
    ]
    tasks = []
    for names, count in populations:
        for _ in range(count):
            canonical_id = str(50 + len(tasks))
            tasks.append(
                {
                    "id": f"{canonical_id}_zh_identity",
                    "evaluation_criteria": {"actions": [{"name": name} for name in names]},
                }
            )
    upstream = _write_json(
        source_dir / "tau2_retail_tasks.json", [{"id": str(index)} for index in range(114)]
    )
    files = {
        "tau2_retail_tasks.json": {
            "sha256": hashlib.sha256(upstream).hexdigest(),
            "bytes": len(upstream),
        }
    }
    bundle = RetailBundle(
        database={"users": {}, "orders": {}, "products": {}},
        tasks=[{"id": f"{index}_zh_identity"} for index in range(50)],
        policy="中文政策",
        metadata={f"{index}_zh_identity": {"canonical_task_id": str(index)} for index in range(50)},
        settings={"language": "zh"},
        manifest={"commit": TAU2_COMMIT, "files": files, "subset": "retail_50"},
        language_pack="language: zh",
    )

    def load_official(source: Path) -> RetailBundle:
        """返回官方部分 fixture，来源目录仍由用例显式传入。"""
        assert source == source_dir
        return bundle

    monkeypatch.setattr("ttwill.bench.tasks.load_retail_bundle", load_official)
    _write_json(
        tasks_dir / "dev_64.json",
        {
            "conversion_version": CONVERSION_VERSION,
            "source": {
                "commit": TAU2_COMMIT,
                "files": files,
                "localization_overrides_sha256": None,
            },
            "tasks": tasks,
            "metadata": {
                task["id"]: {"canonical_task_id": task["id"].split("_")[0]} for task in tasks
            },
        },
    )
    return source_dir, tasks_dir, bundle


def test_retail_subset_does_not_require_development_files(task_sets: tuple) -> None:
    """官方 50 题继续使用原加载路径，无须生成开发题文件。"""
    source, tasks, official = task_sets
    assert load_task_bundle(source, tasks / "absent", "retail_50") is official


def test_development_bundle_records_source_and_excludes_evaluation(task_sets: tuple) -> None:
    """开发集保持相同业务数据，任务与元信息只包含官方评测集的差集。"""
    source, tasks, official = task_sets
    development = load_task_bundle(source, tasks, "dev_64")
    assert len(development.tasks) == len(development.metadata) == 64
    assert set(development.metadata).isdisjoint(official.metadata)
    assert development.database is official.database
    assert development.policy == official.policy
    assert development.manifest["subset"] == "dev_64"
    assert development.manifest["task_ids"] == [task["id"] for task in development.tasks]
    assert (
        development.manifest["task_files"]["dev_64.json"]["sha256"]
        == hashlib.sha256((tasks / "dev_64.json").read_bytes()).hexdigest()
    )
    assert official.manifest["subset"] == "retail_50"


def test_quick_set_covers_available_tools_and_every_stratum(task_sets: tuple) -> None:
    """20 题覆盖 14 层及已有六种写工具，同时明确记录支付修改缺口。"""
    source, tasks, _ = task_sets
    quick = generate_quick_set(tasks)
    assert quick["algorithm"] == QUICK_ALGORITHM
    assert quick["seed"] == 0
    assert len(set(quick["task_ids"])) == quick["size"] == 20
    assert len(quick["strata"]) == 14
    assert all(1 <= row["selected"] <= row["population"] for row in quick["strata"])
    assert sum(row["selected"] for row in quick["strata"]) == 20
    assert quick["population_coverage"]["write_tools"] == quick["selected_coverage"]["write_tools"]
    assert quick["selected_coverage"]["missing_write_tools"] == ["modify_pending_order_payment"]
    assert quick["selected_coverage"]["read_only_tasks"] > 0
    assert quick["selected_coverage"]["multi_write_tasks"] > 0
    assert quick["task_ids"] == [
        f"{task_id}_zh_identity"
        for task_id in [
            52,
            53,
            63,
            57,
            67,
            73,
            70,
            83,
            82,
            86,
            88,
            93,
            94,
            95,
            102,
            101,
            111,
            107,
            112,
            113,
        ]
    ]
    bundle = load_task_bundle(source, tasks, "dev_quick")
    assert [task["id"] for task in bundle.tasks] == quick["task_ids"]
    assert set(bundle.metadata) == set(quick["task_ids"])
    assert set(bundle.manifest["task_files"]) == {"dev_64.json", "dev_quick.json"}


def test_quick_set_selection_is_independent_of_source_list_order(
    task_sets: tuple, tmp_path: Path
) -> None:
    """源任务和元信息排列变化不改变抽中题目，原始字节摘要仍跟随文件变化。"""
    _, tasks, _ = task_sets
    first = generate_quick_set(tasks)
    second_dir = tmp_path / "reordered"
    second_dir.mkdir()
    development = _read_development(tasks)
    development["tasks"].reverse()
    development["metadata"] = dict(reversed(development["metadata"].items()))
    _write_json(second_dir / "dev_64.json", development)
    second = generate_quick_set(second_dir)
    assert first["task_ids"] == second["task_ids"]
    assert first["strata"] == second["strata"]
    assert first["source_sha256"] != second["source_sha256"]


def test_frozen_quick_set_cannot_be_overwritten(task_sets: tuple) -> None:
    """重新抽样不得覆盖已有清单，拒绝后保留原始文件字节。"""
    _, tasks, _ = task_sets
    generate_quick_set(tasks)
    original = (tasks / "dev_quick.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate_quick_set(tasks, seed=1)
    assert (tasks / "dev_quick.json").read_bytes() == original


@pytest.mark.parametrize("size", [15, 16, 17, 18, 19, 20])
def test_quick_set_allows_documented_sizes(task_sets: tuple, size: int) -> None:
    """文档规定的每个规模均分配足额且不重复的题目。"""
    _, tasks, _ = task_sets
    quick = generate_quick_set(tasks, size=size)
    assert len(set(quick["task_ids"])) == size


@pytest.mark.parametrize("arguments", [{"size": 14}, {"size": 21}, {"size": True}, {"seed": False}])
def test_quick_set_rejects_invalid_sampling_arguments(task_sets: tuple, arguments: dict) -> None:
    """越界题数和布尔种子在写清单前被拒绝。"""
    _, tasks, _ = task_sets
    with pytest.raises(ValueError, match="整数种子且题数"):
        generate_quick_set(tasks, **arguments)
    assert not (tasks / "dev_quick.json").exists()


@pytest.mark.parametrize("mutation", ["version", "duplicate", "metadata", "canonical_duplicate"])
def test_development_rejects_invalid_identity_structure(task_sets: tuple, mutation: str) -> None:
    """旧转换版本、重复任务及元信息错配都不能进入运行器。"""
    source, tasks, _ = task_sets
    development = _read_development(tasks)
    first, second = [task["id"] for task in development["tasks"][:2]]
    if mutation == "version":
        development["conversion_version"] = "unknown"
    elif mutation == "duplicate":
        development["tasks"][1]["id"] = first
    elif mutation == "metadata":
        del development["metadata"][first]
    else:
        development["metadata"][second]["canonical_task_id"] = development["metadata"][first][
            "canonical_task_id"
        ]
    _write_json(tasks / "dev_64.json", development)
    with pytest.raises(ValueError):
        load_task_bundle(source, tasks, "dev_64")


def test_development_rejects_evaluation_task_leakage(task_sets: tuple) -> None:
    """即使总题数和 ID 唯一性正确，也不能把 retail_50 题混入开发集。"""
    source, tasks, _ = task_sets
    development = _read_development(tasks)
    first = development["tasks"][0]["id"]
    development["metadata"][first]["canonical_task_id"] = "0"
    _write_json(tasks / "dev_64.json", development)
    with pytest.raises(ValueError, match="差集"):
        load_task_bundle(source, tasks, "dev_64")


@pytest.mark.parametrize("mutation", ["commit", "missing_file", "recorded_hash", "source_bytes"])
def test_development_rejects_changed_source(task_sets: tuple, mutation: str) -> None:
    """转换记录和实际来源都必须匹配已固定的官方文件清单。"""
    source, tasks, _ = task_sets
    development = _read_development(tasks)
    if mutation == "commit":
        development["source"]["commit"] = "other"
    elif mutation == "missing_file":
        development["source"]["files"] = {}
    elif mutation == "recorded_hash":
        development["source"]["files"]["tau2_retail_tasks.json"]["sha256"] = "changed"
    else:
        with (source / "tau2_retail_tasks.json").open("ab") as stream:
            stream.write(b"\n")
    _write_json(tasks / "dev_64.json", development)
    with pytest.raises(ValueError, match="来源"):
        load_task_bundle(source, tasks, "dev_64")


def test_development_tracks_reviewed_localization_overrides(task_sets: tuple) -> None:
    """转换所用补充规则一旦变动，生成快速集与运行加载都拒绝使用旧产物。"""
    source, tasks, _ = task_sets
    overrides = _write_json(tasks / "localization_overrides.json", {"rules": []})
    development = _read_development(tasks)
    development["source"]["localization_overrides_sha256"] = hashlib.sha256(overrides).hexdigest()
    _write_json(tasks / "dev_64.json", development)
    assert len(load_task_bundle(source, tasks, "dev_64").tasks) == 64
    with (tasks / "localization_overrides.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="补充规则"):
        generate_quick_set(tasks)
    with pytest.raises(ValueError, match="补充规则"):
        load_task_bundle(source, tasks, "dev_64")


def test_quick_set_rejects_changed_development_file(task_sets: tuple) -> None:
    """任务文件内容即使只有空白变化，也要重新审阅冻结来源而非静默复用清单。"""
    source, tasks, _ = task_sets
    generate_quick_set(tasks)
    with (tasks / "dev_64.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="快速集来源摘要"):
        load_task_bundle(source, tasks, "dev_quick")


@pytest.mark.parametrize("mutation", ["task", "duplicate", "algorithm", "quota", "coverage"])
def test_quick_set_rejects_manifest_tampering(task_sets: tuple, mutation: str) -> None:
    """题目、配额、覆盖统计及算法均须与固定种子的完整重算一致。"""
    source, tasks, _ = task_sets
    quick = generate_quick_set(tasks)
    if mutation == "task":
        quick["task_ids"][0] = "0_zh_identity"
    elif mutation == "duplicate":
        quick["task_ids"][0] = quick["task_ids"][1]
    elif mutation == "algorithm":
        quick["algorithm"] = "unknown"
    elif mutation == "quota":
        quick["strata"][0]["selected"] += 1
    else:
        quick["selected_coverage"]["missing_write_tools"] = []
    _write_json(tasks / "dev_quick.json", quick)
    with pytest.raises(ValueError, match="固定抽样算法"):
        load_task_bundle(source, tasks, "dev_quick")


def test_unknown_dataset_is_rejected_before_loading(task_sets: tuple) -> None:
    """任务集拼写错误不能回退为默认评测集。"""
    source, tasks, _ = task_sets
    with pytest.raises(ValueError, match="未知任务集"):
        load_task_bundle(source, tasks, "development")
