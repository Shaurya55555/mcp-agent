"""Pretty-print a trace: python -m agent.view_trace traces/<id>.jsonl"""
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    s = json.loads(line)
    k, who = s["kind"], s.get("agent", "")
    tag = f"[{who}] " if who else ""
    if k == "llm":
        print(f"{s['t']:6.2f}s  {tag}LLM  step={s['step']} {s['ms']}ms tokens={s['tokens']} -> {s['tool_calls'] or 'ANSWER'}")
    elif k == "tool":
        flag = "ERR " if s.get("error") else "ok  "
        print(f"{s['t']:6.2f}s  {tag}TOOL {flag}{s['name']} {json.dumps(s['args'])[:90]} ({s['ms']}ms)")
    elif k == "agent_turn":
        extra = {x: s[x] for x in ("plan", "verdict", "feedback", "fallback") if x in s}
        print(f"{s['t']:6.2f}s  == {s['agent'].upper()} turn {s['turn']}: {s['ms']}ms, {s['llm_calls']} llm, "
              f"{s['tool_calls']} tool, {s['tokens']} tokens {json.dumps(extra, ensure_ascii=False)[:160] if extra else ''}")
    else:
        print(f"{s['t']:6.2f}s  {s['name'].upper()}: {str(s.get('input') or s.get('output'))[:110]}")
