"""Multi-step tool-using agent: connects to the MCP shop server over stdio, lets an LLM
plan and call tools until it answers. Every step is traced.

python -m agent.agent "Which customer has the most open tickets?"          # single agent
python -m agent.multi "Which customer has the most open tickets?"          # planner + worker + critic
"""
import asyncio
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .trace import Tracer

ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
BASE = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

SYSTEM = (
    "You are a support-operations assistant with access to a shop database through tools. "
    "Look at the schema before writing SQL. Use tools for every fact; never guess. "
    "Only call create_ticket when the user explicitly asks to open a ticket. "
    "Finish with a short, direct answer that states the concrete values."
)

_tl = threading.local()


def chat(messages, tools=None):
    """One chat-completions call with retries. `tools=None` makes a plain (tool-less) call."""
    payload = {"model": MODEL, "temperature": 0, "messages": messages}
    if tools:
        payload["tools"] = tools
    for attempt in range(6):
        t_req = time.time()
        try:
            r = httpx.post(f"{BASE}/chat/completions", headers={"Authorization": f"Bearer {KEY}"},
                           json=payload, timeout=120)
        except httpx.TransportError:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < 5:
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt * 2)))
            continue
        r.raise_for_status()
        _tl.api_ms = round((time.time() - t_req) * 1000)  # the successful request only, no retry waits
        return r.json()


def chat_timed(messages, tools=None):
    """Runs `chat` and returns (response, api_ms); api_ms is None when the call was faked."""
    _tl.api_ms = None
    resp = chat(messages, tools)
    return resp, getattr(_tl, "api_ms", None)


def tool_specs(listed) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.inputSchema}} for t in listed]


@contextlib.asynccontextmanager
async def mcp_session(db: str | None = None):
    env = {**os.environ, **({"SHOP_DB": db} if db else {})}
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server" / "shop_server.py")], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        yield session


async def llm_turn(messages, tools, tracer, agent: str, step: int) -> dict:
    """One LLM call, traced. Returns the assistant message."""
    t0 = time.time()
    resp, api_ms = await asyncio.to_thread(chat_timed, messages, tools)
    msg = resp["choices"][0]["message"]
    tracer.emit("llm", agent=agent, step=step, ms=round((time.time() - t0) * 1000), api_ms=api_ms, model=MODEL,
                tokens=resp.get("usage", {}).get("total_tokens", 0),
                tool_calls=[c["function"]["name"] for c in msg.get("tool_calls") or []])
    return msg


async def tool_loop(session, messages, tools, tracer, agent: str, max_steps: int, observations: list | None = None):
    """LLM <-> tool loop used by the single agent and by the multi-agent worker.
    Appends to `messages` in place. Returns the final answer text, or None if the step limit was hit."""
    for step in range(max_steps):
        msg = await llm_turn(messages, tools, tracer, agent, step)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content") or ""
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for c in calls:
            name, raw = c["function"]["name"], c["function"].get("arguments") or "{}"
            t1, err = time.time(), None
            try:
                args = json.loads(raw)
                res = await session.call_tool(name, args)
                out = "".join(getattr(x, "text", "") for x in res.content)
                err = out if res.isError else None
            except Exception as e:
                args, out, err = raw, f"{type(e).__name__}: {e}", f"{type(e).__name__}: {e}"
            tracer.emit("tool", agent=agent, name=name, args=args, ms=round((time.time() - t1) * 1000),
                        error=err, output=out[:500])
            if observations is not None:
                observations.append({"tool": name, "args": args, "output": out[:600], "error": err})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out[:6000]})
    return None


def start_messages(system: str, task: str, memory=None, tracer=None) -> list[dict]:
    """System prompt (+ long-term notes recalled for this task) + this session's earlier turns + the task."""
    notes = memory.memory_prompt(task) if memory else ""
    if tracer is not None and memory is not None:
        tracer.emit("run", name="memory", input={"session": memory.session_id,
                                                 "recalled": memory.last_recalled,
                                                 "short_term_turns": len(memory.turns)})
    history = memory.history_messages() if memory else []
    return [{"role": "system", "content": system + notes}, *history, {"role": "user", "content": task}]


async def run_task(task: str, db: str | None = None, memory=None) -> dict:
    """Single-agent mode. `memory` is an agent.memory.Session; without it the agent is stateless, as before."""
    tracer = Tracer(task, mode="single")
    async with mcp_session(db) as session:
        listed = (await session.list_tools()).tools
        tools = tool_specs(listed)
        tracer.emit("run", name="tools_available", input=[t.name for t in listed])
        messages = start_messages(SYSTEM, task, memory, tracer)
        turn = tracer.turn("single")
        answer = await tool_loop(session, messages, tools, tracer, "single", MAX_STEPS)
        turn.finish()
        if answer is None:
            answer = "(stopped: step limit reached)"
        elif memory is not None:
            memory.record(task, answer)  # only finished answers are remembered, not step-limit stops
        tracer.emit("run", name="answer", output=answer)
    return {"answer": answer, **tracer.summary()}


if __name__ == "__main__":
    args = sys.argv[1:]
    mem = None
    if "--memory" in args:  # each CLI run is a new session; long-term memory is shared through MEMORY_DSN
        from .memory import Session, open_memory
        mem = Session(open_memory())
    q = " ".join(a for a in args if a != "--memory") or "How many customers are there?"
    print(json.dumps(asyncio.run(run_task(q, memory=mem)), indent=2))
