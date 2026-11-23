"""Multi-agent mode: planner -> worker -> critic, sharing one state object.

    planner  (no tools)   turns the task into 1 to 4 ordered steps
    worker   (MCP tools)  executes the plan with the same tool loop the single agent uses
    critic   (no tools)   checks the draft answer against the tool observations; may send the worker back once

python -m agent.multi "Which customer has the most open tickets?"
python -m agent.multi --no-critic "How many customers are in Bengaluru?"

Failure policy: a planner that returns unusable output falls back to a one-step plan (the task itself); a critic that
returns unusable output is treated as approval (fail open) and the fact is recorded in the trace. The single-agent
mode in agent.py is untouched and remains the default.
"""
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field

from . import agent as base
from .trace import Tracer

PLANNER_SYSTEM = (
    "You are the planner for a support-operations assistant that can use these database tools: {tools}. "
    "Break the user's task into the fewest ordered steps (1 to 4) that a tool-using worker should carry out. "
    "Do not answer the task and do not invent data. Reply with JSON only: {{\"steps\": [\"...\", \"...\"]}}"
)

WORKER_SYSTEM = (
    base.SYSTEM + " You are the worker agent. A planner prepared these steps for you:\n{plan}\n"
    "Follow them, adapt if the data says otherwise, and finish with the final answer."
)

CRITIC_SYSTEM = (
    "You are a strict reviewer. You get the TASK, the tool OBSERVATIONS and the worker's DRAFT answer. "
    "Approve only if the draft answers the task and every concrete value it states appears in the observations "
    "(or the task asked for a write and the observations show it succeeded). "
    "Reply with JSON only: {\"approved\": true|false, \"feedback\": \"one short sentence\"}"
)


@dataclass
class SharedState:
    """State every agent can read; written by the orchestrator, not by the agents directly."""
    task: str
    plan: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    draft: str | None = None
    verdicts: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def plan_text(self) -> str:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(self.plan, 1)) or "1. " + self.task

    def observation_text(self, cap: int = 3000) -> str:
        lines = [f"{o['tool']}({json.dumps(o['args'], ensure_ascii=False)[:200]}) -> "
                 f"{'ERROR ' if o['error'] else ''}{o['output']}" for o in self.observations]
        return "\n".join(lines)[-cap:] or "(no tool calls were made)"


def parse_json_object(text: str) -> dict | None:
    """First JSON object in a model reply, tolerating code fences and prose around it."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


async def plan_step(tracer, state: SharedState, tool_names: list[str]) -> None:
    turn = tracer.turn("planner")
    msgs = [{"role": "system", "content": PLANNER_SYSTEM.format(tools=", ".join(tool_names))},
            {"role": "user", "content": state.task}]
    msg = await base.llm_turn(msgs, None, tracer, "planner", 0)
    obj = parse_json_object(msg.get("content") or "")
    steps = [str(s).strip() for s in (obj or {}).get("steps", []) if str(s).strip()][:4] if obj else []
    if steps:
        state.plan = steps
        turn.finish(plan=steps)
    else:
        state.plan = [state.task]
        state.notes.append("planner output unusable, used one-step fallback plan")
        turn.finish(plan=state.plan, fallback=True)


async def critic_step(tracer, state: SharedState) -> dict:
    turn = tracer.turn("critic")
    msgs = [{"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"TASK: {state.task}\n\nOBSERVATIONS:\n{state.observation_text()}\n\n"
                                        f"DRAFT: {state.draft}"}]
    msg = await base.llm_turn(msgs, None, tracer, "critic", 0)
    obj = parse_json_object(msg.get("content") or "")
    if obj is None or "approved" not in obj:
        verdict = {"approved": True, "feedback": "", "fail_open": True}
        state.notes.append("critic output unusable, treated as approval")
        turn.finish(verdict="approved (fail open)", fallback=True)
    else:
        verdict = {"approved": bool(obj["approved"]), "feedback": str(obj.get("feedback", ""))[:300]}
        turn.finish(verdict="approved" if verdict["approved"] else "rejected", feedback=verdict["feedback"])
    state.verdicts.append(verdict)
    return verdict


async def run_task_multi(task: str, db: str | None = None, use_critic: bool = True, max_revisions: int = 1,
                         memory=None) -> dict:
    """`memory` is an agent.memory.Session. Recalled notes go to the worker; short-term turns replay to the worker."""
    mode = "multi" if use_critic else "multi-nocritic"
    tracer = Tracer(task, mode=mode)
    state = SharedState(task=task)
    async with base.mcp_session(db) as session:
        listed = (await session.list_tools()).tools
        tools = base.tool_specs(listed)
        tracer.emit("run", name="tools_available", input=[t.name for t in listed])

        await plan_step(tracer, state, [t.name for t in listed])

        messages = base.start_messages(WORKER_SYSTEM.format(plan=state.plan_text()), task, memory, tracer)
        steps_left = base.MAX_STEPS
        revisions = 0
        while True:
            turn = tracer.turn("worker")
            used_before = len([s for s in tracer.spans if s["kind"] == "llm" and s.get("agent") == "worker"])
            draft = await base.tool_loop(session, messages, tools, tracer, "worker", steps_left, state.observations)
            used = len([s for s in tracer.spans if s["kind"] == "llm" and s.get("agent") == "worker"]) - used_before
            steps_left -= used
            turn.finish(revision=revisions)
            if draft is None:
                state.draft = "(stopped: step limit reached)"
                break
            state.draft = draft
            if not use_critic:
                break
            verdict = await critic_step(tracer, state)
            if verdict["approved"] or revisions >= max_revisions or steps_left <= 0:
                break
            revisions += 1
            messages.append({"role": "assistant", "content": draft})
            messages.append({"role": "user", "content": "Reviewer feedback: " + (verdict["feedback"] or "not approved")
                             + ". Fix this using the tools if needed and give the corrected final answer."})
        if memory is not None and state.draft and state.draft != "(stopped: step limit reached)":
            memory.record(task, state.draft)
        tracer.emit("run", name="answer", output=state.draft)
    return {"answer": state.draft, "plan": state.plan, "revisions": revisions, "notes": state.notes,
            **tracer.summary()}


if __name__ == "__main__":
    args = sys.argv[1:]
    critic = "--no-critic" not in args
    mem = None
    if "--memory" in args:
        from .memory import Session, open_memory
        mem = Session(open_memory())
    q = " ".join(a for a in args if not a.startswith("--")) or "How many customers are there?"
    print(json.dumps(asyncio.run(run_task_multi(q, use_critic=critic, memory=mem)), indent=2, ensure_ascii=False))
