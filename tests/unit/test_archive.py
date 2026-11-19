"""验证完整档案、幂等重放、追加限制和用户作用域隔离。"""

import sqlite3

import pytest

from ttwill.persistence import Archive
from ttwill.persistence.archive import DIALOGUE_MARKER


def test_messages_survive_reopen_without_truncation(tmp_path):
    """关闭连接后仍能读回完整长消息与工具记录，修改读取结果不会改写原文。"""
    path = tmp_path / "archive.sqlite"
    message = {"role": "user", "content": "原话" * 5000}
    with Archive(path) as archive:
        first = archive.append_message("thread", "u1", "m1", message, turn=1)
        second = archive.append("thread", "u1", "tool1", "tool_result", {"value": [1, 2]}, turn=1)
        assert second > first
    with Archive(path) as archive:
        read = archive.messages("thread", "u1")
        assert read[0]["payload"] == message
        read[0]["payload"]["content"] = "外部改动"
        assert archive.get_event("thread", "u1", "m1")["payload"] == message
        assert archive.get_event("thread", "u1", "tool1")["payload"] == {"value": [1, 2]}


def test_identical_events_replay_idempotently(tmp_path):
    """相同事件的字典键顺序不影响幂等判断，也不会生成第二份记录。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        first = archive.append("t", "u", "e", "event", {"a": 1, "b": 2}, turn=1)
        replay = archive.append("t", "u", "e", "event", {"b": 2, "a": 1}, turn=1)
        assert replay == first
        assert len(archive.entries("t", "u")) == 1


@pytest.mark.parametrize(
    ("kind", "payload", "turn"),
    [("event", {"a": 2}, 1), ("message", {"a": 1}, 1), ("event", {"a": 1}, 2)],
)
def test_event_id_cannot_be_rebound(tmp_path, kind, payload, turn):
    """同一标识的正文、类型或轮次变化均被拒绝，原记录保持完整。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        archive.append("t", "u", "e", "event", {"a": 1}, turn=1)
        with pytest.raises(ValueError, match="不同内容"):
            archive.append("t", "u", "e", kind, payload, turn=turn)
        assert archive.get_event("t", "u", "e")["payload"] == {"a": 1}
        assert len(archive.entries("t", "u")) == 1


def test_queries_are_scoped_by_thread_and_owner(tmp_path):
    """消息、摘要与引用查询都不能读到另一用户或另一任务线程的内容。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        archive.append_message("t1", "u1", "same", {"role": "user", "content": "甲"}, turn=1)
        archive.append_message("t2", "u1", "same", {"role": "user", "content": "乙"}, turn=1)
        archive.append_message("t1", "u2", "same", {"role": "user", "content": "丙"}, turn=1)
        archive.append("t1", "u2", "summary", "summary", {"text": "另一用户摘要"})
        assert [item["payload"]["content"] for item in archive.messages("t1", "u1")] == ["甲"]
        assert archive.get_event("t1", "u1", "summary") is None
        assert archive.entries("t1", "u1", kind="summary") == []
        assert archive.entries("missing", "u1") == []


def test_database_enforces_append_only(tmp_path):
    """数据库层拒绝更新或删除档案原文，避免其他调用路径绕过追加约定。"""
    path = tmp_path / "archive.sqlite"
    with Archive(path) as archive:
        archive.append("t", "u", "e", "event", {"value": 1})
        with sqlite3.connect(path) as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute("UPDATE archive_entries SET payload = '{}'")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute("DELETE FROM archive_entries")
        assert archive.get_event("t", "u", "e")["payload"] == {"value": 1}


@pytest.mark.parametrize(
    ("thread", "owner", "event", "kind", "turn"),
    [
        ("", "u", "e", "event", 0),
        ("t", "", "e", "event", 0),
        ("t", "u", "", "event", 0),
        ("t", "u", "e", "task_state", 0),
        ("t", "u", "e", "event", -1),
        ("t", "u", "e", "event", True),
    ],
)
def test_invalid_scope_or_record_metadata_is_rejected(tmp_path, thread, owner, event, kind, turn):
    """无作用域记录、任务状态类型及非法轮次不进入档案。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        with pytest.raises(ValueError):
            archive.append(thread, owner, event, kind, {}, turn=turn)


def test_non_json_payload_is_rejected_before_writing(tmp_path):
    """无穷数值或非 JSON 对象不能产生部分写入。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        with pytest.raises(ValueError):
            archive.append("t", "u", "nan", "event", {"value": float("nan")})
        with pytest.raises(TypeError):
            archive.append("t", "u", "set", "event", {"value": {1, 2}})
        assert archive.entries("t", "u") == []


def tool_exchange() -> list[tuple[str, dict]]:
    """构造同一协调决策中必须原子保存的工具调用及结果。"""
    return [
        (
            "call",
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "query", "arguments": "{}"},
                    }
                ],
            },
        ),
        ("result", {"role": "tool", "tool_call_id": "call1", "content": "完整业务结果"}),
    ]


def test_message_batch_is_idempotent_and_keeps_tool_pairs(tmp_path):
    """同批工具消息重放不会重复写入，重开连接后仍是完整调用与结果。"""
    path = tmp_path / "archive.sqlite"
    messages = tool_exchange()
    with Archive(path) as archive:
        sequences = archive.append_messages("t", "u", messages, turn=1)
        assert archive.append_messages("t", "u", messages, turn=1) == sequences
    with Archive(path) as archive:
        assert [entry["payload"] for entry in archive.messages("t", "u")] == [
            message for _, message in messages
        ]


def test_message_batch_conflict_rolls_back_earlier_new_messages(tmp_path):
    """后续事件标识冲突时整批回滚，不能留下新插入的孤立工具调用。"""
    path = tmp_path / "archive.sqlite"
    baseline = {"role": "user", "content": "已存在的原文"}
    with Archive(path) as archive:
        archive.append_message("t", "u", "result", baseline, turn=1)
        with pytest.raises(ValueError, match="不同内容"):
            archive.append_messages("t", "u", tool_exchange(), turn=1)
    with Archive(path) as archive:
        assert [entry["payload"] for entry in archive.messages("t", "u")] == [baseline]
        assert archive.get_event("t", "u", "call") is None


def test_database_failure_rolls_back_entire_message_batch(tmp_path):
    """真实 SQLite 在第二条插入失败时回滚第一条，重开后不存在残缺往返。"""
    path = tmp_path / "archive.sqlite"
    with Archive(path) as archive:
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TRIGGER fail_tool_result BEFORE INSERT ON archive_entries "
                "WHEN NEW.event_id = 'result' "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END;"
            )
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            archive.append_messages("t", "u", tool_exchange(), turn=1)
    with Archive(path) as archive:
        assert archive.messages("t", "u") == []
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TRIGGER fail_tool_result")
        archive.append_messages("t", "u", tool_exchange(), turn=1)
        assert len(archive.messages("t", "u")) == 2


def test_invalid_message_rolls_back_entire_batch(tmp_path):
    """第二条消息无效时第一条也不保存，保留消息角色边界。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        messages = [tool_exchange()[0], ("invalid", {"role": "unknown", "content": "无效"})]
        with pytest.raises(ValueError, match="有效角色"):
            archive.append_messages("t", "u", messages, turn=1)
        assert archive.messages("t", "u") == []


def test_dialogue_marks_actual_utterances_without_changing_complete_archive(tmp_path):
    """显式标记跨轮对话，原始工具与内部消息仍完整保存，读取对话不泄漏元数据。"""
    path = tmp_path / "archive.sqlite"
    user = {"role": "user", "content": "帮我修改地址"}
    reply = {"role": "assistant", "content": "请确认以下地址卡片。"}
    messages = [
        ("user", user),
        *tool_exchange(),
        ("internal", {"role": "assistant", "content": "请确认以下地址卡片。"}),
        ("unmarked-user", {"role": "user", "content": "帮我修改地址"}),
        ("reply", reply),
    ]
    with Archive(path) as archive:
        archive.append_messages("t", "u", messages, turn=1, dialogue_ids={"user", "reply"})
        archive.append_message("other", "u", "user", user, turn=1, dialogue=True)
        archive.append_message("t", "other", "reply", reply, turn=1, dialogue=True)
        assert DIALOGUE_MARKER not in user
        assert DIALOGUE_MARKER not in reply
    with Archive(path) as archive:
        dialogue = archive.dialogue_messages("t", "u")
        assert [entry["event_id"] for entry in dialogue] == ["user", "reply"]
        assert [entry["payload"] for entry in dialogue] == [user, reply]
        assert len(archive.messages("t", "u")) == len(messages)
        assert archive.get_event("t", "u", "result")["payload"] == tool_exchange()[1][1]
        assert archive.get_event("t", "u", "user")["payload"][DIALOGUE_MARKER] is True
        dialogue[0]["payload"]["content"] = "外部改动"
        assert archive.dialogue_messages("t", "u")[0]["payload"] == user


def test_dialogue_metadata_is_part_of_idempotency_identity(tmp_path):
    """重放保留原标记，但不能把已存的内部消息重新定义为实际回复。"""
    message = {"role": "assistant", "content": "回复正文"}
    with Archive(tmp_path / "archive.sqlite") as archive:
        sequence = archive.append_message("t", "u", "reply", message, turn=1, dialogue=True)
        assert archive.append_message("t", "u", "reply", message, turn=1, dialogue=True) == sequence
        with pytest.raises(ValueError, match="不同内容"):
            archive.append_message("t", "u", "reply", message, turn=1)
        archive.append_message("t", "u", "internal", message, turn=1)
        with pytest.raises(ValueError, match="不同内容"):
            archive.append_message("t", "u", "internal", message, turn=1, dialogue=True)
        assert [entry["event_id"] for entry in archive.dialogue_messages("t", "u")] == ["reply"]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": "系统内容"},
        tool_exchange()[0][1],
        tool_exchange()[1][1],
    ],
)
def test_internal_messages_cannot_be_marked_as_dialogue(tmp_path, message):
    """误标工具调用、结果或系统内容时整批回滚，不能污染跨轮历史。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        with pytest.raises(ValueError, match="对话标记只接受"):
            archive.append_messages(
                "t",
                "u",
                [("user", {"role": "user", "content": "原话"}), ("internal", message)],
                turn=1,
                dialogue_ids={"user", "internal"},
            )
        assert archive.messages("t", "u") == []


def test_dialogue_marks_require_explicit_batch_ids(tmp_path):
    """保留标记不能从原始消息注入，批次外标识也不能被默默忽略。"""
    with Archive(tmp_path / "archive.sqlite") as archive:
        with pytest.raises(ValueError, match="本批消息"):
            archive.append_messages("t", "u", tool_exchange(), turn=1, dialogue_ids={"missing"})
        with pytest.raises(ValueError, match="档案接口设置"):
            archive.append_message(
                "t",
                "u",
                "injected",
                {"role": "user", "content": "原话", DIALOGUE_MARKER: True},
                turn=1,
            )
        assert archive.messages("t", "u") == []
