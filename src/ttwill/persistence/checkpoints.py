"""初始化原生 LangGraph SQLite checkpointer，不包装另一层状态存取接口。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """打开本地检查点数据库，退出时释放异步连接。

    Args:
        path: 检查点 SQLite 文件；父目录须已存在，运行时与业务库分别保存。

    Yields:
        原生 AsyncSqliteSaver，直接交给 StateGraph.compile 使用。
    """
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        saver.serde = JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=())
        yield saver
