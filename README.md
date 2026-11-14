# TTWill

面向买家的多品类电商客服 Agent：理解意愿，完成所托。

当前实现第 0～1 步：最小 Python 包、配置读取、统一模型调用、假模型、轨迹及命令行查看。业务后端、子 Agent 和协调者按[实施计划](<docs/05 实施计划.md>)继续开发，项目设计见[文档导航](<docs/00 项目定位.md#文档导航>)。

## 安装与检查

需要 Python 3.12 和 uv。在仓库根目录执行：

```bash
uv python install 3.12
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`uv.lock` 固定依赖版本。运行依赖仅为 HTTPX（兼容接口请求）和 python-dotenv（本地密钥读取）；开发依赖为 pytest、Ruff，构建使用 Hatchling。本阶段不需要 τ²、LangGraph 或数据库依赖。

## 假模型演示

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
