"""Re-score saved eval answers with the current matcher; no model calls.

python -m agent.rescore
Reads results/eval.json (written by agent.eval_tasks), recomputes the text check for read-only tasks,
keeps the stored result for the write task (it was verified by inspecting the database), and adds
per-call latency taken from the trace files.
"""
import json
import statistics
from pathlib import Path

from server import seed
from .eval_tasks import EVAL_DB, any_form, tasks

ROOT = Path(__file__).resolve().parents[1]


def main():
    saved = json.loads((ROOT / "results" / "eval.json").read_text(encoding="utf-8"))
    seed.build(EVAL_DB)  # deterministic seed, so expected values are reproducible
    spec = tasks(EVAL_DB)
    rows, llm_ms, tool_ms = [], [], []
    for run, (task, expect, extra) in zip(saved["runs"], spec):
        assert run["task"] == task, "task order changed since the saved run"
        ok = run["ok"] if extra else any_form(expect, run["answer"])
        rows.append({"task": task, "ok_as_run": run["ok"], "ok_rescored": ok, "tool_calls": run["tool_calls"]})
        trace = ROOT / "traces" / f"{run['trace']}.jsonl"
        if trace.exists():
            for line in trace.read_text(encoding="utf-8").splitlines():
                s = json.loads(line)
                (llm_ms if s["kind"] == "llm" else tool_ms if s["kind"] == "tool" else []).append(s.get("ms", 0)) \
                    if s["kind"] in ("llm", "tool") else None
    n = len(rows)
    summary = {
        "tasks": n,
        "passed_as_run": sum(r["ok_as_run"] for r in rows),
        "passed_rescored": sum(r["ok_rescored"] for r in rows),
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
        "tool_error_rate": saved["summary"]["tool_error_rate"],
        "median_llm_call_ms": round(statistics.median(llm_ms)) if llm_ms else None,
        "median_tool_call_ms": round(statistics.median(tool_ms)) if tool_ms else None,
        "note": "per-task wall time in eval.json is dominated by API rate-limit retries; use per-call latency",
    }
    out = {"summary": summary, "rows": rows}
    (ROOT / "results" / "eval_rescored.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for r in rows:
        if r["ok_as_run"] != r["ok_rescored"]:
            print("changed:", r["task"])


if __name__ == "__main__":
    main()
