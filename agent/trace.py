"""Tiny tracer: one JSONL file per run, spans for every LLM call and tool call."""
import json
import time
import uuid
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[1] / "traces"


class Tracer:
    def __init__(self, task: str):
        self.trace_id = uuid.uuid4().hex[:8]
        self.path = TRACE_DIR / f"{self.trace_id}.jsonl"
        TRACE_DIR.mkdir(exist_ok=True)
        self.t0 = time.time()
        self.spans: list[dict] = []
        self.emit("run", name="task", input=task)

    def emit(self, kind: str, **fields):
        span = {"trace": self.trace_id, "t": round(time.time() - self.t0, 3), "kind": kind, **fields}
        self.spans.append(span)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        llm = [s for s in self.spans if s["kind"] == "llm"]
        tools = [s for s in self.spans if s["kind"] == "tool"]
        return {
            "trace": self.trace_id,
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for s in tools if s.get("error")),
            "tokens": sum(s.get("tokens", 0) for s in llm),
            "seconds": round(time.time() - self.t0, 2),
        }
