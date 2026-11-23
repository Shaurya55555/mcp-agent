"""Agent loop against the real MCP server over stdio, with a scripted LLM (no network)."""
import asyncio

import httpx
import pytest

from agent import agent as A
from agent import trace
from server import seed


def reply(content=None, calls=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                             for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": msg}], "usage": {"total_tokens": 10}}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path / "traces")
    db = seed.build(tmp_path / "t.db")

    def go(script, max_steps=8):
        it = iter(script)
        monkeypatch.setattr(A, "chat", lambda messages, tools: next(it))
        monkeypatch.setattr(A, "MAX_STEPS", max_steps)
        return asyncio.run(A.run_task("q", str(db)))

    return go


def test_tool_then_answer(run):
    out = run([reply(calls=[("get_schema", "{}")]), reply("done")])
    assert out["answer"] == "done" and out["tool_calls"] == 1 and out["tool_errors"] == 0


def test_malformed_tool_arguments_are_reported_not_fatal(run):
    out = run([reply(calls=[("run_sql", "{not json")]), reply("recovered")])
    assert out["answer"] == "recovered" and out["tool_errors"] == 1


def test_unknown_tool_is_reported_not_fatal(run):
    out = run([reply(calls=[("drop_everything", "{}")]), reply("ok")])
    assert out["answer"] == "ok" and out["tool_errors"] == 1


def test_server_side_rejection_reaches_the_model(run):
    out = run([reply(calls=[("run_sql", '{"query": "DELETE FROM customers"}')]), reply("refused")])
    assert out["tool_errors"] == 1 and out["answer"] == "refused"


def test_step_limit_stops_a_looping_model(run):
    out = run([reply(calls=[("get_schema", "{}")])] * 3, max_steps=3)
    assert out["answer"] == "(stopped: step limit reached)" and out["llm_calls"] == 3


def test_chat_retries_429_with_retry_after(monkeypatch):
    sleeps, seq = [], iter([429, 200])

    class R:
        def __init__(self, code):
            self.status_code, self.headers = code, {"retry-after": "2"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(A.httpx, "post", lambda *a, **k: R(next(seq)))
    monkeypatch.setattr(A.time, "sleep", sleeps.append)
    assert A.chat([], []) == {"ok": True} and sleeps == [2.0]


def test_chat_gives_up_after_repeated_transport_errors(monkeypatch):
    n = {"c": 0}

    def post(*a, **k):
        n["c"] += 1
        raise httpx.ConnectError("down")

    monkeypatch.setattr(A.httpx, "post", post)
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    with pytest.raises(httpx.ConnectError):
        A.chat([], [])
    assert n["c"] == 6
