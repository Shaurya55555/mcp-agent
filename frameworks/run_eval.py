"""Score a framework implementation on the same 12 ground-truth tasks as agent/eval_tasks.py.

python -m frameworks.run_eval langgraph     # -> results/frameworks_langgraph.json
python -m frameworks.run_eval adk           # -> results/frameworks_adk.json
python -m frameworks.loc                    # lines of code per implementation
"""
import asyncio
import importlib
import json
import shutil
import statistics
import sys

from agent.eval_tasks import EVAL_DB, ROOT, any_form, q, tasks
from server import seed


async def main(name: str):
    mod = importlib.import_module(f"frameworks.{name}_flow")
    seed.build(EVAL_DB)
    work = ROOT / "data" / f"eval_work_{name}.db"
    shutil.copy(EVAL_DB, work)
    out = []
    for i, (task, expect, extra) in enumerate(tasks(EVAL_DB), 1):
        try:
            res = await mod.run_task(task, str(work))
        except Exception as e:  # a framework crash counts as a failed task, and is recorded
            res = {"answer": f"CRASH {type(e).__name__}: {e}"[:300], "llm_calls": 0, "tool_calls": 0, "tool_errors": 0,
                   "tokens": 0, "seconds": 0.0}
        ok = any_form(expect, res["answer"])
        if extra and extra[0] == "write":
            ok = ok and q(work, "SELECT customer_id, priority, subject FROM tickets WHERE id > ?", extra[1]) == [(3, "high", "Refund not received")]
        res.update(task=task, ok=ok)
        out.append(res)
        print(f"{'PASS' if ok else 'FAIL'} [{i:02d}] llm={res['llm_calls']} tools={res['tool_calls']} tokens={res['tokens']} {res['seconds']}s", flush=True)
        if not ok:
            print("      answer:", str(res["answer"])[:200])
    n = len(out)
    summary = {"framework": name, "tasks": n, "passed": sum(r["ok"] for r in out),
               "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in out), 2),
               "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in out), 2),
               "avg_tokens": round(statistics.mean(r["tokens"] for r in out)),
               "p50_wall_seconds": round(statistics.median(r["seconds"] for r in out), 2),
               "crashes": sum(str(r["answer"]).startswith("CRASH") for r in out)}
    print(json.dumps(summary, indent=2))
    (ROOT / "results" / f"frameworks_{name}.json").write_text(json.dumps({"summary": summary, "runs": out}, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
