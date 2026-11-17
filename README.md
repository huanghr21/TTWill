# TTWill

面向买家的多品类电商客服 Agent：理解意愿，完成所托。

第 0～6 步已通过离线验收；第 7 步已接入官方中文用户模拟器、单题运行入口与终态判分，并取得真实模型单题结果：评测完成，该题业务失败。LangGraph 检查点保存任务进度，协调者按用户确认提交，核验通过才完成任务，详见[实施计划](<docs/05 实施计划.md>)和[文档导航](<docs/00 项目定位.md#文档导航>)。`retail_50` 无模型参考重放实测 48/50 终态一致，另两题已确认为上游缺陷，原始差异单列保留。

## 安装与检查

需要 Python 3.12 和 uv。在仓库根目录执行：

```bash
uv python install 3.12
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`uv.lock` 固定依赖版本。运行依赖为 HTTPX（兼容接口请求）、python-dotenv（本地密钥读取）、SQLAlchemy（业务 SQLite 事务）、Pydantic（参数与结构定义）、LangGraph 与 `langgraph-checkpoint-sqlite`（原生异步 SQLite 检查点）；开发依赖为 pytest、Ruff，构建使用 Hatchling。普通业务运行不需要 τ²；离线参考校验单独安装 `bench` 可选依赖。

## 命令行业务入口

从固定中文包初始化一个独立业务库，再按邮箱验证身份并查询。以下数据均来自基准的合成数据：

```bash
uv run ttwill business init --db runs/manual/business.sqlite
uv run ttwill business query --db runs/manual/business.sqlite \
  --email xue.cao2645@example.com --tool get_order_details \
  --args '{"order_id":"#W5918442"}'
uv run ttwill business prepare --db runs/manual/business.sqlite \
  --email xue.cao2645@example.com --tool cancel_pending_order \
  --task-id manual-cancel-1 --revision 1 \
  --args '{"order_id":"#W5918442","reason":"no longer needed"}'
```

准备只记录操作，返回 `operation_id`、`request_key`、原参数、期限与 `preview`，业务数据不变。
检查输出后，在新命令中传入返回的操作 ID 和同一份参数：

```bash
uv run ttwill business commit --db runs/manual/business.sqlite \
  --email xue.cao2645@example.com --operation-id '<准备返回的 operation_id>' \
  --args '{"order_id":"#W5918442","reason":"no longer needed"}'
uv run ttwill business operation --db runs/manual/business.sqlite \
  --email xue.cao2645@example.com --request-key '<准备返回的 request_key>'
```

取消准备使用 `business reject --operation-id ... --reason ...`，同样提供 `--db`、`--email`。
每条命令重新验证身份并保存轨迹；可用 `--config` 指定配置文件。业务库与初始化的来源路径以当前目录为准，轨迹目录以及 `skills/`、`data/source/` 资源目录以配置文件所在目录为准。初始化不覆盖已有业务数据。诊断用的 `business prepare` 会显式加载对应 Skill；子 Agent 则需要自己调用 `load_skill`。

同一任务 ID、修订号和动作产生同一请求标识，跨进程重试返回原操作，重复提交不重复退款。
同一订单在准备至提交、拒绝、过期或确定失败之间只允许一个待提交操作。请求改动或失效后重新办理时递增修订号。
准备有效期由 `[backend].operation_ttl_seconds` 设置，当前开发配置为 900 秒；它不代表用户的对话授权。
这里的显式提交用于后端验收；协调者中的提交走“最近回复展示 → 用户确认 → 代码提交 → 重读核验”的对话流程。

## 离线参考动作校验

```bash
uv sync --locked --extra bench
uv run --extra bench pytest
uv run --extra bench ttwill bench --output runs/reference-001
```

重放目录必须是新目录。每题新建业务库，正常导入固定提交的官方工具执行同一参考动作，
使用官方 `RetailDB.get_hash()` 比较完整终态，同时检查动作返回值和七个读工具。
这一步不调用模型；τ² 导入使用本地成本表。来源、文件校验值和许可证见 [data/README.md](data/README.md)。

2026-11-15 实测：50 题均完成重放，48 题终态及动作返回值一致，七个读工具全部一致，七个写工具均被覆盖。
`21_zh_identity`、`36_zh_identity` 的差异来自上游批量修改商品时复用最后一个规格，
详见[上游差异记录](<docs/07 调研_tau2-bench.md#第-2-步实测差异>)。
完整报告保留所有差异；存在任一终态、动作或读取差异时，命令退出码仍为 1，表示没有达到完全一致。第 2 步验收已接受上述两项已知差异，不将 48/50 改写成 50/50。
这些数据是无模型的后端一致性结果，不是含自然语言断言的 Agent 任务成功率。

## 本体场景验收

```bash
uv run pytest -q tests/acceptance
```

七个场景使用固定中文开发种子、真实政策、SQLite 后端和持久运行时，模型输出按手写脚本提供。检查完整业务终态、逐轮确认、任务修订、跨会话恢复，以及 span、模型文件和账本的关联；不调用真实模型服务。场景清单及保留数据库、对话和轨迹的方法见[验收说明](tests/acceptance/README.md)。

本次验收修复了重复任务快照导致三轮对话提前超预算的问题：持久运行时每次注入一份最新权威快照，保留完整原话和工具往返，默认预算保持不变。提交、拒绝和核验轨迹带上对应任务 ID 与修订，派工记录包含任务说明。本次结果不代表真实模型的任务成功率。

## 单题模型评测

先安装 `bench` 可选依赖。`config.toml` 的 `flash` 档位已配置官方 `https://api.deepseek.com`、API 模型名 `deepseek-flash` 和 `reasoning_effort = "high"`；该模型名对应 DeepSeek-V4.1-Flash-1017，见 [官方接入说明](https://api-docs.deepseek.com/zh-cn/)。密钥只写入同目录 `.env` 的 `TTWILL_API_KEY`。本体、模拟用户和判分共用此档位时：

```bash
uv run --extra bench ttwill evaluate --task-id 6_zh_identity \
  --user-tier flash --judge-tier flash --seed 0 --repeat 1 \
  --output runs/step7-task6-attempt-001
```

`6_zh_identity` 是固定中文任务列表中的第一题，用于接入验收。也可在同一配置中增加 `models.user_simulator` 与 `models.evaluator` 档位，省略两个 tier 选项；本体仍用 `flash`，摘要用 `[context].summary_tier`。默认官方对话上限为 100 步（用户和助手各算一步），总时限为 600 秒、在步骤间检查；单次调用另受 `[llm].timeout_seconds` 限制。

入口为整个系统创建独立业务库、检查点与档案。`HalfDuplexAgent` 只转接用户原话和助手文本，不接收隐藏用户场景或参考答案。身份由模型从对话提取并通过真实网关验证；验证成功后，在下一次创建任务的决策前接入检查点，保留本轮已消耗步数。模拟用户使用固定提交的官方中文人设和开场白。当前仅接受 `retail_50`，明确拒绝非空 `initial_state`。

判分的预测侧直接加载实际业务库快照，参考侧在同源初始库执行官方参考动作。报告分别保留 `db_match`（数据库终态一致）与 `task_success`（含适用自然语言断言的任务成功）。自然语言判分使用显式配置的模型，并保持固定官方提示词、JSON 结构及聚合规则；官方对缺失断言结果的宽松行为也保留，限制见[接入记录](<docs/07 调研_tau2-bench.md#第-7-步接入记录>)。

每次尝试要求新目录，额外保存 `metadata.json`、`task.json`、`initial_database.json`、`final_database.json`、`user_prompt.json`、`simulation.json`、`messages.json` 和 `report.json`；中途失败时保存已产生的产物。配置快照不含密钥，代码及提示资源带内容哈希。全部模型调用进入统一轨迹、原文记录和 `usage.jsonl`，费用以此账本为准，不采用官方轨迹中可能缺失的 `agent_cost`。官方 LiteLLM 的隐式重试和缓存关闭，失败尝试保留；本体统一调用层仍按配置重试。重试耗尽的网络故障、超时或判分中断记为 `completed=false`，不填造任务失败分数。正常完成但业务不成功仍为已完成失败，命令退出码为 1。

2026-11-17 首次真实运行保存在 `runs/step7-deepseek-task6-20261117-attempt-001/`：`completed=true`、`termination_reason=user_stop`、`db_match=false`、`task_success=false`，判分为 0。53 次 API 调用均成功返回，输入 258260、输出 33126、其中缓存输入 169088 token，费用未知。协调者多次违反工具调用约定，随后触发轮数及上下文预算上限；没有提交写操作，最终用户退出。这是已完成判分的业务失败，保留原尝试，不重跑覆盖。

本题没有自然语言断言，因此没有实际调用判分模型；该链路仍只有离线模拟验证。`action_checks` 也不能用于本体网关动作覆盖统计，原因见[接入记录](<docs/07 调研_tau2-bench.md#第-7-步接入记录>)。第 7 步已取得要求的真实单题判分，不代表任务表现通过；基线与正式实验尚未开始，困难评测集仍须在 D 组表现调优前冻结。

首轮协议修复已通过 640 项离线测试：身份验证前可直接询问信息，派工注明已验证身份，内部停止原因与客服回复分离，纯文本回复只归档一次。DeepSeek 思考工具协议要求的 `reasoning_content` 保留；计划顺序及其余上下文压缩问题仍待处理，详见 [09 失败案例与修复记录](<docs/09 失败案例与修复记录.md>)。修复后未调用真实模型，实际改善留待开发快速集验证。

## 假模型演示

运行包含关闭重开、身份重验与重新展示的跨会话演示：

```bash
uv run ttwill runtime-demo
```

第一段聊天准备取消订单并暂停；关闭业务库、档案和检查点连接后，第二段聊天打开同一组文件并重新验身份。新聊天先重新展示原操作，下一条“确认”才提交并核验。三轮输出依次为 `pending_confirm → pending_confirm → done`，任务和操作 ID 保持不变，退款只执行一次；输出包含两段轨迹及业务库、检查点、档案的实际路径。

`PersistentRuntime.open_session()` 先用真实网关查找用户，再读取该线程检查点并校验用户与原业务库路径；`send_message()` 通过 LangGraph 的等待节点接收原话。准备成功但检查点尚无操作 ID 时，按任务 ID、修订号和动作找回操作；提交成功但检查点未更新时，恢复只核验，不再次提交。异常中断后新建运行时并重新打开会话，旧聊天的同意不会被重放。

`[context]` 的开发默认值是 `max_chars = 64000`、`recent_turns = 4`、`summary_tier = "flash"`。预算按消息和工具定义的 JSON 字符数计算，不是 token 数。超限先把旧工具结果替换为档案引用，仍超限才调用一次摘要模型并保存摘要供复用。当前用户原话、任务快照与近期轮次完整保留；无法容纳时明确说明原因并停止本轮。历史摘要不构成确认。

当前运行时支持单进程内同一业务任务串行推进，接续新聊天会使旧实例失去推进资格；多个进程不要同时推进同一任务。不同评测任务使用独立业务库和检查点范围。原有的 `coordinator-demo` 仍是直接调用内存协调器的轻量演示。

运行请求与确认的两轮完整对话：

```bash
uv run ttwill coordinator-demo
```

第一轮验证身份并准备取消订单，回复中展示中文办理参数和预期结果；第二轮接收真实入口传入的“确认”，由代码提交，再重新查询订单并核验，任务才变为 `done`。输出包含两轮回复、任务状态、逐操作确认结果、业务库和轨迹路径。演示使用独立业务库与假模型，不需要密钥。

协调者先处理本轮改口和撤销，再独立判定上一回复的完整展示内容。只有仍有效、修订号和参数一致的操作能提交；多个展示项收到范围不明的“好的”时不提交。同单同类诉求聚合，同单不同写操作逐个办理，不同订单可并行。回复中的任务状态和操作详情由代码生成，模型的完成措辞不能覆盖业务事实。

`[agent].max_rounds` 限制每条用户消息的协调决策次数，开发默认 8。身份查询、任务更新、结束计划、分派、回复以及无效决策均计一轮；每轮只调用一个协调工具，`dispatch_tasks` 可批量派出不同订单的任务，因此分派轮数也受该上限约束。超限将仍需推进的任务标为 `failed` 并说明原因；已准备且等待用户的操作保留。

单独派发订单服务或售后任务：

```bash
uv run ttwill agent-demo --capability order_service
uv run ttwill agent-demo --capability after_sales
```

每次演示创建独立业务库，用假模型依次查询订单、加载 Skill、准备取消或退货，再返回结构化结果；业务状态保持原样。输出包含 `result`、业务库和轨迹路径，状态应为 `need_confirm`。`actions`、业务事实和待确认参数由网关与后端记录填写；模型只提供 `missing`、`message_hint`，不能填写完成状态或调用提交工具。

子 Agent 的循环由 `[agent].max_steps` 限制，开发默认 12 步。每个工具调用计一步，无工具回复也计一步；超限会返回 `failed`，作废本实例尚未提交的准备项，并记录 guard。实例结束即清除 Skill 加载记录，下一实例重新加载。政策正文保留固定来源中的英文，中文操作要点单独放在 `skills/`。

完整对话轨迹包含 `session → turn → coordinator` 下的任务更新、分派、确认、提交、核验、回复和状态快照；分派下是子 Agent 的模型、工具与护栏记录。查看轨迹，或运行只包含一次模型调用的基础设施演示：

```bash
uv run ttwill demo
uv run ttwill trace runs/<run_id>/traces/<session_id>.jsonl
```

演示会打印实际文件路径和 `session → turn → llm` 树，不需要密钥，也不会访问模型服务。四个演示使用内存配置副本，将全部模型档位固定为 `fake-demo` 和零价格，保留用户配置的运行目录及预算；不修改真实配置文件。演示模型身份、用量和价格不代表真实服务的计量或定价。

运行产物保存在配置指定的目录中，默认是 `runs/`，不进入 Git：

```text
runs/<run_id>/
├── business.sqlite             本次任务的业务记录与准备 / 提交事实
├── checkpoints.sqlite          LangGraph 任务状态与节点进度，唯一任务恢复来源
├── archive.sqlite              只追加的完整消息、事件与历史摘要
├── traces/<session_id>.jsonl    每个 span 的开始和结束事件
├── models/<call_id>-<n>.json    每次尝试的完整请求、响应或错误类型
└── usage.jsonl                 所有会话、重试的用量、费用与延迟
```

相对路径以配置文件所在目录为准；从其他目录启动时传 `ttwill demo --config /绝对路径/config.toml`。轨迹中只保存模型记录的引用，原始请求和响应放在独立文件。查看器会标出尚未结束的 span。

前三个数据库文件由 `runtime-demo` 创建，基础设施演示只创建自身需要的记录。恢复实现使用原生检查点和中断机制，参考 [LangGraph 持久化](https://docs.langchain.com/oss/python/langgraph/persistence)与[中断文档](https://docs.langchain.com/oss/python/langgraph/interrupts)。

## 模型调用

业务代码通过 `LLMClient.complete("flash", messages, tools=...)` 指定档位；实际模型与参数来自 `config.toml`。`FakeModel` 接受响应或错误序列，用于离线测试；`CompatibleModel` 支持非流式 Chat Completions 接口，保留工具调用和服务返回的扩展字段。

切换真实服务时，在配置中填写实际 API `model`、`base_url` 与 `api_key_env`。密钥只写入同目录 `.env`，变量名与 `api_key_env` 一致（参见 `.env.example`）。参数写入 `[models.flash.parameters]`，不能覆盖调用层的模型、消息、工具和流式设置。当前固定官方默认思考强度 `high`，见 [思考模式说明](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)。使用统一调用层的示例：

```python
from pathlib import Path

from ttwill.config import load_config
from ttwill.llm import CompatibleModel, LLMClient
from ttwill.tracing import Trace

# 放在 async 函数内；session_id 每段新聊天使用唯一值。
config = load_config(Path("config.toml"))
trace = Trace(config.runs_dir / "manual-run", "session-001")
async with CompatibleModel(config.root / ".env") as backend:
    client = LLMClient(config, trace, backend)
    with trace.span("session", "模型调用"):
        response = await client.complete("flash", [{"role": "user", "content": "你好"}])
```

每次尝试分别记录耗时和结果；仅传输错误、超时、HTTP 408/429/5xx 重试，鉴权失败和无效响应直接报错。`max_retries` 是首次调用之外的重试次数，等待时间按 `retry_delay_seconds` 指数增加。超时既约束 HTTP 各阶段，也约束单次尝试的总用时。取消操作向上传播，不自动重试。

有用量但缺少价格，或服务未返回用量时，费用记为 `null`，不当作零。提供缓存用量时按缓存单价计算；未配置缓存单价则费用仍为未知。失败尝试也留在账本中，其未知用量不会伪造为零。价格均为配置估算，并非服务端结算金额。

DeepSeek 当前按高峰和低峰时段计费，固定单价不能覆盖两种时段，因此本次配置暂不填写价格，完整保存输入、缓存输入和输出 token，费用保留 `null`；费率和时段见 [官方价格表](https://api-docs.deepseek.com/quick_start/pricing/)。需要费用估算时应注明所采用的时段及单价，不把未知费用填成零。

真实接口已完成上述单题调用，自动测试继续使用 HTTPX 模拟传输或假模型。协议参考 [DeepSeek Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)，超时语义参考 [HTTPX 文档](https://www.python-httpx.org/advanced/timeouts/)。
