"""用固定官方语言包为中文零售评测构造用户人设与开场白。"""

import yaml
from tau2.data_model.language import LanguagePack, RuntimePersona, TaskMetadata
from tau2.data_model.tasks import Task
from tau2.multilingual.runtime import assign_persona, resolve_persona

from ttwill.bench.retail import RetailBundle


def build_persona(bundle: RetailBundle, task: Task, *, seed: int) -> tuple[RuntimePersona, str]:
    """按官方规则选择本题的中文人设，并返回官方文字对话开场白。

    语言包和任务元数据来自已校验的同一固定提交，不读取 τ² 默认数据目录。
    人设按官方任务 ID、性别和随机种子确定；金额说明复用官方区域任务约定。

    Args:
        bundle: load_retail_bundle 返回的中文数据库、任务元信息和语言包原文。
        task: 本次运行的官方任务，其 ID 必须出现在数据包元信息中。
        seed: 官方 assign_persona 使用的随机种子，用于可复现的人设选择。

    Returns:
        二元组，依次为传给 build_user 的 RuntimePersona 和中文开场白。

    Raises:
        ValueError: 语言包格式或语言不匹配，或者任务缺少中文元信息。
    """
    pack = LanguagePack.model_validate(yaml.safe_load(bundle.language_pack))
    if pack.language != bundle.settings["language"] or pack.language != "zh":
        raise ValueError("评测语言包与中文零售数据不匹配")
    if task.id not in bundle.metadata:
        raise ValueError(f"任务缺少官方中文人设元信息：{task.id}")
    metadata = TaskMetadata.model_validate(bundle.metadata[task.id])
    persona = resolve_persona(
        pack,
        assign_persona(pack, metadata, seed),
        mode="text",
        binding=pack.binding("retail"),
    )
    # 与固定版本 multilingual.runtime.resolve_task 的区域任务说明保持一致。
    currency = bundle.settings["currency_code"]
    persona.user_guidelines += (
        f"\nScenario monetary amounts are in {currency}. "
        "Express that currency in the call language; "
        "do not convert amounts or substitute another currency."
    )
    return persona, pack.agent_greeting
