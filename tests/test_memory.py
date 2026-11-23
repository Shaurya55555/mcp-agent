"""Long-term memory: the same tests run on SQLite always and on PostgreSQL when MEMORY_TEST_DSN is set (CI does)."""
import json
import os

import pytest

from agent import memory as M

PG_DSN = os.getenv("MEMORY_TEST_DSN", "")


@pytest.fixture(params=["sqlite", "postgres"])
def opener(request, tmp_path):
    """Returns a function that opens a NEW connection to the same database each time it is called."""
    if request.param == "sqlite":
        path = tmp_path / "mem.db"
        return lambda: M.SqliteMemory(path)
    if not PG_DSN:
        pytest.skip("MEMORY_TEST_DSN not set, PostgreSQL tests skipped")
    M.PostgresMemory(PG_DSN).clear()
    return lambda: M.PostgresMemory(PG_DSN)


def test_fact_stored_in_one_session_is_recalled_in_a_later_one(opener):
    store1 = opener()
    s1 = M.Session(store1)
    s1.record("Remember that I manage the Bengaluru region", "Noted: you manage the Bengaluru region.")
    store1.close()  # the first session is over, its connection is gone

    s2 = M.Session(opener())  # a later session: new Session object, new connection
    found = s2.recall("Which region do I manage?")
    assert found and "Bengaluru" in found[0].summary
    assert found[0].session_id == s1.session_id != s2.session_id


def test_a_session_does_not_recall_its_own_turns_from_long_term_memory(opener):
    s = M.Session(opener())
    s.record("Bengaluru has 7 customers", "yes")
    assert s.recall("Bengaluru customers") == []  # its own turns come back through short-term memory instead


def test_unrelated_query_recalls_nothing(opener):
    M.Session(opener()).record("Which city has the most customers?", "Mumbai with 9")
    assert M.Session(opener()).recall("what is the weather forecast") == []
    assert M.Session(opener()).recall("the of and") == []  # only stop words


def test_better_match_ranks_first_and_k_limits_results(opener):
    s1 = M.Session(opener())
    s1.record("Open tickets for Anaya Kapoor", "Anaya Kapoor has 5 open tickets")
    s1.record("Revenue of delivered orders", "Total revenue is 1,200,000")
    s1.record("Customers in Bengaluru", "7 customers in Bengaluru")
    s2 = M.Session(opener(), recall_k=2)
    found = s2.recall("how many open tickets does Anaya Kapoor have")
    assert len(found) <= 2 and "Anaya" in found[0].summary


def test_newer_memory_wins_a_tie(opener):
    s1 = M.Session(opener())
    s1.record("Which region do I manage", "Region: Pune")
    s2 = M.Session(opener())
    s2.record("Which region do I manage", "Region: Bengaluru")
    found = M.Session(opener(), recall_k=1).recall("which region do I manage")
    assert "Bengaluru" in found[0].summary


def test_clear_and_count(opener):
    store = opener()
    store.clear()
    M.Session(store).record("hello world", "hi")
    assert store.count() == 1
    store.clear()
    assert store.count() == 0


def test_summary_is_capped_and_whitespace_collapsed():
    s = M.make_summary("q   with\n\nspaces", "a" * 2000)
    assert len(s) == M.MAX_SUMMARY_CHARS and s.endswith("...") and s.startswith("Q: q with spaces A: ")


def test_terms_drops_stop_words_and_dedupes():
    assert M.terms("Which region do I manage? The region!") == ["region", "manage"]


def test_memory_prompt_is_labelled_as_possibly_stale(tmp_path):
    store = M.SqliteMemory(tmp_path / "m.db")
    M.Session(store).record("Customers in Bengaluru", "7 customers")
    text = M.Session(store).memory_prompt("how many customers in Bengaluru")
    assert "out of date" in text and "7 customers" in text
    assert M.Session(store).memory_prompt("completely unrelated zebra") == ""


def test_short_term_history_replays_only_the_last_turns():
    s = M.Session(None, max_turns=2)
    for i in range(4):
        s.record(f"q{i}", f"a{i}")
    msgs = s.history_messages()
    assert [m["content"] for m in msgs] == ["q2", "a2", "q3", "a3"]
    assert s.recall("anything") == [] and s.memory_prompt("anything") == ""


def test_open_memory_picks_the_backend_from_the_dsn(tmp_path, monkeypatch):
    assert isinstance(M.open_memory(str(tmp_path / "x.db")), M.SqliteMemory)
    made = {}
    monkeypatch.setattr(M, "PostgresMemory", lambda dsn: made.setdefault("dsn", dsn))
    M.open_memory("postgresql://u:p@localhost/db")
    assert made["dsn"].startswith("postgresql://")


# ====================================================================== wired into the agents (scripted LLM)
import asyncio  # noqa: E402

from agent import agent as A  # noqa: E402
from agent import multi as MU  # noqa: E402
from agent import trace  # noqa: E402
from server import seed  # noqa: E402


def reply(content=None, calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                           for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": m}], "usage": {"total_tokens": 5}}


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path / "traces")
    db = str(seed.build(tmp_path / "shop.db"))
    sent = []

    def script(replies):
        it = iter(replies)

        def fake(messages, tools):
            sent.append(json.loads(json.dumps(messages)))
            return next(it)

        monkeypatch.setattr(A, "chat", fake)

    return db, script, sent


FACT_TASK = "My name is Priya and I manage the Bengaluru region. Remember that."


def test_single_agent_recalls_a_fact_told_in_an_earlier_session(opener, agent_env):
    db, script, sent = agent_env
    store1 = opener()
    script([reply("Noted, Priya. You manage the Bengaluru region.")])
    asyncio.run(A.run_task(FACT_TASK, db, memory=M.Session(store1)))
    store1.close()

    script([reply("You manage the Bengaluru region.")])
    s2 = M.Session(opener())  # new session, new database connection
    out = asyncio.run(A.run_task("Which region do I manage?", db, memory=s2))
    system = sent[-1][0]["content"]
    assert "Notes from earlier sessions" in system and "Bengaluru" in system
    assert out["answer"] == "You manage the Bengaluru region." and s2.last_recalled == 1


def test_control_without_memory_the_prompt_has_no_notes(agent_env):
    db, script, sent = agent_env
    script([reply("I do not know.")])
    asyncio.run(A.run_task("Which region do I manage?", db))
    assert "Notes from earlier sessions" not in sent[-1][0]["content"]


def test_multi_agent_worker_also_receives_recalled_notes(opener, agent_env):
    db, script, sent = agent_env
    store1 = opener()
    script([reply('{"steps": ["answer"]}'), reply("Noted, Bengaluru."), reply('{"approved": true, "feedback": ""}')])
    asyncio.run(MU.run_task_multi(FACT_TASK, db, memory=M.Session(store1)))
    store1.close()

    script([reply('{"steps": ["recall"]}'), reply("Bengaluru."), reply('{"approved": true, "feedback": ""}')])
    asyncio.run(MU.run_task_multi("Which region do I manage?", db, memory=M.Session(opener())))
    worker_system = sent[-2][0]["content"]  # [planner, worker, critic] -> worker is second to last
    assert "Bengaluru" in worker_system and "Notes from earlier sessions" in worker_system


def test_short_term_memory_replays_earlier_turns_within_one_session(agent_env):
    db, script, sent = agent_env
    s = M.Session(None)
    script([reply("There are 7.")])
    asyncio.run(A.run_task("How many customers are in Bengaluru?", db, memory=s))
    script([reply("Seven, as I said.")])
    asyncio.run(A.run_task("And how many did you say?", db, memory=s))
    msgs = sent[-1]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "How many customers are in Bengaluru?" and msgs[2]["content"] == "There are 7."


def test_a_session_does_not_get_its_own_turns_twice(opener, agent_env):
    db, script, sent = agent_env
    s = M.Session(opener())
    script([reply("There are 7 customers in Bengaluru.")])
    asyncio.run(A.run_task("How many customers are in Bengaluru?", db, memory=s))
    script([reply("Same.")])
    asyncio.run(A.run_task("How many customers in Bengaluru again?", db, memory=s))
    assert "Notes from earlier sessions" not in sent[-1][0]["content"]  # it is in short-term history instead


def test_step_limit_stops_are_not_remembered(opener, agent_env, monkeypatch):
    db, script, sent = agent_env
    monkeypatch.setattr(A, "MAX_STEPS", 1)
    store = opener()
    store.clear()
    script([reply(calls=[("get_schema", "{}")])])
    out = asyncio.run(A.run_task("something long", db, memory=M.Session(store)))
    assert out["answer"].startswith("(stopped") and store.count() == 0


def test_memory_span_is_traced(agent_env):
    db, script, sent = agent_env
    script([reply("ok")])
    out = asyncio.run(A.run_task("hello world", db, memory=M.Session(None)))
    spans = [json.loads(line) for line in (trace.TRACE_DIR / f"{out['trace']}.jsonl").read_text(encoding="utf-8").splitlines()]
    mem = [s for s in spans if s.get("name") == "memory"][0]
    assert mem["input"]["recalled"] == 0 and mem["input"]["short_term_turns"] == 0
