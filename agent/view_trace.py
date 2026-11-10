"""Pretty-print a trace: python -m agent.view_trace traces/<id>.jsonl"""
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    s = json.loads(line)
    k = s["kind"]
    if k == "llm":
        print(f"{s['t']:6.2f}s  LLM  step={s['step']} {s['ms']}ms tokens={s['tokens']} -> {s['tool_calls'] or 'ANSWER'}")
    elif k == "tool":
        flag = "ERR " if s.get("error") else "ok  "
        print(f"{s['t']:6.2f}s  TOOL {flag}{s['name']} {json.dumps(s['args'])[:90]} ({s['ms']}ms)")
    else:
        print(f"{s['t']:6.2f}s  {s['name'].upper()}: {str(s.get('input') or s.get('output'))[:110]}")
