"""验证固定中文人设能直接初始化官方文字用户模拟器，无需外部数据目录。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("tau2", reason="安装 bench 额外依赖后运行官方中文人设测试")

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.domains.retail.data_model import RetailDB  # noqa: E402
from tau2.domains.retail.environment import get_environment  # noqa: E402
from tau2.runner.build import build_user  # noqa: E402

from ttwill.bench.persona import build_persona  # noqa: E402
from ttwill.bench.retail import RetailBundle, load_retail_bundle  # noqa: E402

SOURCE_DIR = Path(__file__).resolve().parents[2] / "data" / "source"


@pytest.fixture(scope="module")
def bundle() -> RetailBundle:
    """读取通过来源校验的完整中文任务和官方语言包。

    Returns:
        本模块只读共用的固定中文数据包。
    """
    return load_retail_bundle(SOURCE_DIR)


def test_official_persona_initializes_mandarin_text_user(bundle: RetailBundle) -> None:
    """官方用户系统提示词包含中文强制要求和原任务，不依赖默认英文语言包。"""
    task = Task.model_validate(bundle.tasks[0])
    persona, greeting = build_persona(bundle, task, seed=42)
    env = get_environment(db=RetailDB.model_validate(bundle.database), policy=bundle.policy)
    user = build_user(
        "user_simulator",
        env,
        task,
        llm="openai/test-user",
        llm_args={"temperature": 0, "api_base": "https://example.invalid/v1", "timeout": 5},
        persona_config=persona,
    )
    state = user.get_init_state()
    prompt = state.system_messages[0].content
    assert "every message you write to the agent MUST be in Mandarin" in prompt
    assert "Every message you write to the agent is in Mandarin" in prompt
    assert str(task.user_scenario) in prompt
    assert "Scenario monetary amounts are in CNY" in prompt
    assert "###STOP###" in prompt
    assert persona.language == "zh" and persona.mode == "text" and persona.voice is None
    assert greeting == "您好！请问有什么可以帮您？"
    assert user.tools is None
    assert user.llm == "openai/test-user"
    assert user.llm_args == {
        "temperature": 0,
        "api_base": "https://example.invalid/v1",
        "timeout": 5,
    }


def test_persona_assignment_is_reproducible_and_matches_task_gender(bundle: RetailBundle) -> None:
    """同一任务和种子稳定选择官方人设，全部五十题保持对应性别。"""
    original = deepcopy(bundle.metadata)
    for raw_task in bundle.tasks:
        task = Task.model_validate(raw_task)
        persona, greeting = build_persona(bundle, task, seed=42)
        repeated, repeated_greeting = build_persona(bundle, task, seed=42)
        assert persona == repeated and greeting == repeated_greeting
        assert persona.gender == bundle.metadata[task.id]["caller_gender"]
        assert persona.user_guidelines.count("Scenario monetary amounts are in CNY") == 1
    assert bundle.metadata == original


def test_unknown_task_cannot_invent_persona_metadata(bundle: RetailBundle) -> None:
    """任务缺少固定元信息时明确失败，不猜测性别或原任务 ID。"""
    task = Task.model_validate(bundle.tasks[0]).model_copy(update={"id": "unknown-task"})
    with pytest.raises(ValueError, match="任务缺少官方中文人设元信息"):
        build_persona(bundle, task, seed=42)


def test_mismatched_bundle_language_is_rejected(bundle: RetailBundle) -> None:
    """语言包与业务数据语言不一致时不能启动模拟用户。"""
    changed = replace(bundle, settings={**bundle.settings, "language": "en"})
    with pytest.raises(ValueError, match="语言包与中文零售数据不匹配"):
        build_persona(changed, Task.model_validate(bundle.tasks[0]), seed=42)
