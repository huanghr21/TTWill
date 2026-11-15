# TTWill

面向买家的多品类电商客服 Agent：理解意愿，完成所托。

当前实现到第 3 步：业务后端、按实例加载 Skill、订单服务与售后子 Agent。协调者和对话确认按[实施计划](<docs/05 实施计划.md>)继续开发，设计见[文档导航](<docs/00 项目定位.md#文档导航>)。`retail_50` 实测 48/50 终态一致，另两题已确认为上游缺陷，原始差异单列保留。

## 安装与检查

需要 Python 3.12 和 uv。在仓库根目录执行：

```bash
uv python install 3.12
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`uv.lock` 固定依赖版本。运行依赖为 HTTPX（兼容接口请求）、python-dotenv（本地密钥读取）、SQLAlchemy（SQLite 事务）和 Pydantic（工具参数与结构定义）；开发依赖为 pytest、Ruff，构建使用 Hatchling。普通业务运行不需要 τ²；离线参考校验单独安装 `bench` 可选依赖。尚未引入 LangGraph。

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
这里的显式提交用于后端验收，第 4 步再实现“最近回复展示 → 用户确认 → 代码提交”的对话护栏。

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

## 假模型演示

单独派发订单服务或售后任务：

```bash
uv run ttwill agent-demo --capability order_service
uv run ttwill agent-demo --capability after_sales
```

每次演示创建独立业务库，用假模型依次查询订单、加载 Skill、准备取消或退货，再返回结构化结果；业务状态保持原样。输出包含 `result`、业务库和轨迹路径，状态应为 `need_confirm`。`actions`、业务事实和待确认参数由网关与后端记录填写；模型只提供 `missing`、`message_hint`，不能填写完成状态或调用提交工具。

子 Agent 的循环由 `[agent].max_steps` 限制，开发默认 12 步。每个工具调用计一步，无工具回复也计一步；超限会返回 `failed`，作废本实例尚未提交的准备项，并记录 guard。实例结束即清除 Skill 加载记录，下一实例重新加载。政策正文保留固定来源中的英文，中文操作要点单独放在 `skills/`。

轨迹包含 `session → turn → dispatch → subagent → llm / tool / guard / result`。查看轨迹，或运行只包含一次模型调用的基础设施演示：

```bash
uv run ttwill demo
uv run ttwill trace runs/<run_id>/traces/<session_id>.jsonl
```

演示会打印实际文件路径和 `session → turn → llm` 树，不需要密钥，也不会访问模型服务。默认 `config.toml` 中的 `fake-demo`、用量和零价格只用于演示，不代表真实服务的 API 标识、计量或定价。

运行产物保存在配置指定的目录中，默认是 `runs/`，不进入 Git：

```text
runs/<run_id>/
├── traces/<session_id>.jsonl    每个 span 的开始和结束事件
├── models/<call_id>-<n>.json    每次尝试的完整请求、响应或错误类型
└── usage.jsonl                 所有会话、重试的用量、费用与延迟
```

相对路径以配置文件所在目录为准；从其他目录启动时传 `ttwill demo --config /绝对路径/config.toml`。轨迹中只保存模型记录的引用，原始请求和响应放在独立文件。查看器会标出尚未结束的 span。

## 模型调用

业务代码通过 `LLMClient.complete("flash", messages, tools=...)` 指定档位；实际模型与参数来自 `config.toml`。`FakeModel` 接受响应或错误序列，用于离线测试；`CompatibleModel` 支持非流式 Chat Completions 接口，保留工具调用和服务返回的扩展字段。

真实服务接入时，在配置中将对应档位的 `model` 改为实际 API 标识，设置 `base_url` 与 `api_key_env`，删除演示用的零价格或替换为已核实的美元/百万 token 单价。密钥只写入同目录 `.env`，变量名与 `api_key_env` 一致（参见 `.env.example`）。参数写入 `[models.flash.parameters]`，不能覆盖调用层的模型、消息、工具和流式设置。示例：

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

真实接口目前通过 HTTPX 模拟传输验证，尚未发起付费调用。协议参考 [DeepSeek Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)，超时语义参考 [HTTPX 文档](https://www.python-httpx.org/advanced/timeouts/)。
