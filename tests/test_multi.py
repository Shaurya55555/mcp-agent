"""Planner / worker / critic orchestration against the real MCP server, with a scripted LLM (no network)."""
import asyncio
import json

import pytest

from agent import agent as A
from agent import multi as M
from agent import trace
from server import seed


def msg(content=None, calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                           for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": m}], "usage": {"total_tokens": 10}}


PLAN = msg(json.dumps({"steps": ["read the schema", "count customers in Bengaluru"]}))
APPROVE = msg('{"approved": true, "feedback": ""}')


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path / "traces")
    db = seed.build(tmp_path / "t.db")
    seen = {}

    def go(script, **kw):
        it = iter(script)

        def fake(messages, tools):
            seen.setdefault("calls", []).append((json.loads(json.dumps(messages)), tools))
            return next(it)

        monkeypatch.setattr(A, "chat", fake)
        out = asyncio.run(M.run_task_multi("How many customers are in Bengaluru?", str(db), **kw))
        out["_calls"] = seen["calls"]
        return out

    return go


def spans(out):
    path = trace.TRACE_DIR / f"{out['trace']}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_happy_path_planner_worker_critic(run):
    out = run([PLAN, msg(calls=[("get_schema", "{}")]), msg("There are 7 customers."), APPROVE])
    assert out["answer"] == "There are 7 customers." and out["revisions"] == 0
    assert out["plan"] == ["read the schema", "count customers in Bengaluru"]
    turns = [(s["agent"], s["turn"]) for s in spans(out) if s["kind"] == "agent_turn"]
    assert turns == [("planner", 1), ("worker", 1), ("critic", 1)]
    assert out["llm_calls"] == 4 and out["tool_calls"] == 1 and out["agent_turns"] == 3


def test_every_llm_and_tool_span_is_labelled_with_its_agent(run):
    out = run([PLAN, msg(calls=[("get_schema", "{}")]), msg("a"), APPROVE])
    labelled = [s["agent"] for s in spans(out) if s["kind"] in ("llm", "tool")]
    assert labelled == ["planner", "worker", "worker", "worker", "critic"]


def test_planner_gets_no_tools_and_worker_gets_the_plan(run):
    out = run([PLAN, msg("done"), APPROVE])
    planner_msgs, planner_tools = out["_calls"][0]
    worker_msgs, worker_tools = out["_calls"][1]
    assert not planner_tools and worker_tools
    assert "1. read the schema" in worker_msgs[0]["content"]


def test_critic_sees_the_workers_tool_observations(run):
    out = run([PLAN, msg(calls=[("get_schema", "{}")]), msg("answer"), APPROVE])
    critic_user = out["_calls"][-1][0][-1]["content"]
    assert "get_schema" in critic_user and "customers" in critic_user and "DRAFT: answer" in critic_user


def test_rejected_draft_goes_back_to_the_worker_once(run):
    reject = msg('{"approved": false, "feedback": "count is not in the observations"}')
    out = run([PLAN, msg("guess 99"), reject, msg(calls=[("get_schema", "{}")]), msg("corrected 7"), APPROVE])
    assert out["answer"] == "corrected 7" and out["revisions"] == 1
    workers = [s for s in spans(out) if s["kind"] == "agent_turn" and s["agent"] == "worker"]
    assert [w["turn"] for w in workers] == [1, 2]
    worker2_msgs = out["_calls"][3][0]
    assert any("Reviewer feedback: count is not in the observations" in (m.get("content") or "")
               for m in worker2_msgs)


def test_revision_budget_is_capped(run):
    reject = msg('{"approved": false, "feedback": "no"}')
    out = run([PLAN, msg("a"), reject, msg("b"), reject], max_revisions=1)
    assert out["answer"] == "b" and out["revisions"] == 1


def test_unusable_planner_output_falls_back_to_one_step(run):
    out = run([msg("I think we should just look"), msg("ok"), APPROVE])
    assert out["plan"] == ["How many customers are in Bengaluru?"]
    assert any("fallback plan" in n for n in out["notes"])
    planner_turn = [s for s in spans(out) if s["kind"] == "agent_turn" and s["agent"] == "planner"][0]
    assert planner_turn["fallback"] is True


def test_unusable_critic_output_fails_open(run):
    out = run([PLAN, msg("answer"), msg("looks fine to me")])
    assert out["answer"] == "answer" and out["revisions"] == 0
    assert any("treated as approval" in n for n in out["notes"])


def test_no_critic_mode_skips_the_critic(run):
    out = run([PLAN, msg("answer")], use_critic=False)
    assert out["llm_calls"] == 2
    assert not [s for s in spans(out) if s.get("agent") == "critic"]


def test_worker_step_limit_applies(run, monkeypatch):
    monkeypatch.setattr(A, "MAX_STEPS", 2)
    out = run([PLAN, msg(calls=[("get_schema", "{}")]), msg(calls=[("get_schema", "{}")])])
    assert out["answer"] == "(stopped: step limit reached)"


def test_parse_json_object_tolerates_fences_and_prose():
    assert M.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert M.parse_json_object('sure: {"approved": true} thanks') == {"approved": True}
    assert M.parse_json_object("no json") is None and M.parse_json_object("[1,2]") is None
