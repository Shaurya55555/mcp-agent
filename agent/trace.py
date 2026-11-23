"""Tiny tracer: one JSONL file per run, spans for every LLM call, tool call and agent turn."""
import json
import time
import uuid
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[1] / "traces"


class Turn:
    """One turn of one agent (planner, worker, critic, or the single agent). Emits an `agent_turn` span on finish."""

    def __init__(self, tracer: "Tracer", agent: str, idx: int):
        self.tracer, self.agent, self.idx = tracer, agent, idx
        self.start = len(tracer.spans)
        self.t0 = time.time()

    def finish(self, **fields):
        inner = self.tracer.spans[self.start:]
        llm = [s for s in inner if s["kind"] == "llm"]
        tools = [s for s in inner if s["kind"] == "tool"]
        self.tracer.emit("agent_turn", agent=self.agent, turn=self.idx, ms=round((time.time() - self.t0) * 1000),
                         llm_calls=len(llm), tool_calls=len(tools), tokens=sum(s.get("tokens", 0) for s in llm),
                         **fields)


class Tracer:
    def __init__(self, task: str, mode: str = "single"):
        self.trace_id = uuid.uuid4().hex[:8]
        self.path = TRACE_DIR / f"{self.trace_id}.jsonl"
        TRACE_DIR.mkdir(exist_ok=True)
        self.t0 = time.time()
        self.spans: list[dict] = []
        self.turns: dict[str, int] = {}
        self.emit("run", name="task", input=task, mode=mode)

    def emit(self, kind: str, **fields):
        span = {"trace": self.trace_id, "t": round(time.time() - self.t0, 3), "kind": kind, **fields}
        self.spans.append(span)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    def turn(self, agent: str) -> Turn:
        self.turns[agent] = self.turns.get(agent, 0) + 1
        return Turn(self, agent, self.turns[agent])

    def summary(self) -> dict:
        llm = [s for s in self.spans if s["kind"] == "llm"]
        tools = [s for s in self.spans if s["kind"] == "tool"]
        api_ms = sum(s.get("api_ms") or 0 for s in llm) + sum(s.get("ms") or 0 for s in tools)
        return {
            "trace": self.trace_id,
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for s in tools if s.get("error")),
            "tokens": sum(s.get("tokens", 0) for s in llm),
            "agent_turns": sum(1 for s in self.spans if s["kind"] == "agent_turn"),
            "seconds": round(time.time() - self.t0, 2),
            # time actually spent in provider calls and tools, excluding rate-limit retry waits
            "active_seconds": round(api_ms / 1000, 2),
        }
