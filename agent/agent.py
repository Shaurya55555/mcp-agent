"""Multi-step tool-using agent: connects to the MCP shop server over stdio, lets an LLM
plan and call tools until it answers. Every step is traced.

python -m agent.agent "Which customer has the most open tickets?"
"""
import asyncio
import json
import os
import sys
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


def chat(messages, tools):
    payload = {"model": MODEL, "temperature": 0, "messages": messages, "tools": tools}
    for attempt in range(6):
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
        return r.json()


async def run_task(task: str, db: str | None = None) -> dict:
    tracer = Tracer(task)
    env = {**os.environ, **({"SHOP_DB": db} if db else {})}
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "server" / "shop_server.py")], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        listed = (await session.list_tools()).tools
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.inputSchema}} for t in listed]
        tracer.emit("run", name="tools_available", input=[t.name for t in listed])
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        answer = None
        for step in range(MAX_STEPS):
            t0 = time.time()
            resp = await asyncio.to_thread(chat, messages, tools)
            msg = resp["choices"][0]["message"]
            tracer.emit("llm", step=step, ms=round((time.time() - t0) * 1000), model=MODEL,
                        tokens=resp.get("usage", {}).get("total_tokens", 0),
                        tool_calls=[c["function"]["name"] for c in msg.get("tool_calls") or []])
            calls = msg.get("tool_calls")
            if not calls:
                answer = msg.get("content") or ""
                break
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
                tracer.emit("tool", name=name, args=args, ms=round((time.time() - t1) * 1000),
                            error=err, output=out[:500])
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out[:6000]})
        if answer is None:
            answer = "(stopped: step limit reached)"
        tracer.emit("run", name="answer", output=answer)
    return {"answer": answer, **tracer.summary()}


if __name__ == "__main__":
    out = asyncio.run(run_task(" ".join(sys.argv[1:]) or "How many customers are there?"))
    print(json.dumps(out, indent=2))
