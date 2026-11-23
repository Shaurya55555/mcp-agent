"""Side-by-side table of eval runs: python -m agent.compare [results/eval_single.json results/eval_multi.json ...]

With no arguments it reads every results/eval_*.json except eval_rescored.json (the legacy eval.json is not read).
Several files for the same mode (for example eval_single.json and eval_single_run2.json) are reported as separate rows,
so run-to-run variance stays visible instead of being averaged away.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def row(path: Path) -> str:
    d = json.loads(path.read_text(encoding="utf-8"))
    s, runs = d["summary"], d["runs"]
    failed = [i for i, r in enumerate(runs, 1) if not r["ok"]]
    return (f"| {path.stem.removeprefix('eval_')} | {sum(r['ok'] for r in runs)}/{s['tasks']} "
            f"| {s['avg_llm_calls']} | {s['avg_tool_calls']} | {s['avg_tokens']} "
            f"| {s['p50_active_seconds']} | {s['mean_active_seconds']} | {s['p50_seconds']} "
            f"| {', '.join(map(str, failed)) or '-'} |")


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted(
        p for p in (ROOT / "results").glob("eval_*.json") if p.stem != "eval_rescored")
    head = ("| run | tasks passed | LLM calls / task | tool calls / task | tokens / task | p50 active s | mean active s "
            "| p50 wall s | failed task #s |\n|---|---|---|---|---|---|---|---|---|")
    out = head + "\n" + "\n".join(row(p) for p in paths)
    print(out)
    (ROOT / "results" / "compare.md").write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
