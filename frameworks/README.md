# The same planner / worker / critic flow in three ways

The flow from `agent/multi.py` (planner writes a plan, worker runs MCP tools, critic can send the worker back once)
rebuilt with **LangGraph** (`langgraph_flow.py`) and **Google ADK** (`adk_flow.py`). Same MCP shop server, same model
(`openai/gpt-oss-120b` on Groq's free tier), same 12 ground-truth tasks, same pass check.

    pip install -r frameworks/requirements.txt
    python -m frameworks.langgraph_flow "How many customers are in Bengaluru?"
    python -m frameworks.adk_flow "How many customers are in Bengaluru?"
    python -m frameworks.run_eval langgraph      # or adk; writes results/frameworks_<name>.json
    python -m frameworks.loc

## Lines of code (non-blank, non-comment, no docstrings)
| implementation | lines |
|---|---|
| native: `agent/multi.py` | 124 |
| native, shared pieces it relies on: tool loop, session, retries in `agent/agent.py` / tracer in `agent/trace.py` | 129 / 48 |
| LangGraph: `frameworks/langgraph_flow.py` | 92 |
| Google ADK: `frameworks/adk_flow.py` | 65 |

The LangGraph file imports the exact prompts and the JSON parser from `agent/multi.py`; the ADK file has its own prompt strings.
So the line counts are not a pure framework comparison, but they show the order of magnitude: ADK hides the loop, the tool plumbing and
the shared state (`output_key`), LangGraph makes the graph explicit (6 nodes, 3 conditional edges) and keeps routing in your code.

## Behavior (one run each, 12 tasks)
| | native multi (with critic) | LangGraph | Google ADK |
|---|---|---|---|
| tasks passed | 12/12 (twice) | 12/12 | 12/12 |
| LLM calls / task | 4.83 | 5.67 | 4.92 |
| tool calls / task | 1.83 | 2.33 | 1.92 |
| tokens / task | 2454 and 2470 | 3283 | 4559 |
| p50 wall seconds / task | 12.5 and 11.4 | 13.3 | 31.2 |

- All three reach the right answers. LangGraph made extra tool calls on a few tasks (5 tool calls on task 5 where the native flow used 2,
  4 on the write task); I did not investigate why, so treat it as an observation, not a diagnosis.
- ADK sends the most tokens: its agents see the whole conversation history, so the critic and later turns re-read earlier tool output. At
  Groq's free 8,000 tokens/minute limit that shows up as rate-limit waits, so the ADK wall time is **not** comparable (the native runs also report
  "active" seconds in `results/compare.md`; the framework runs only have wall time).
- Critic mechanics differ: native and LangGraph parse a JSON verdict; ADK has the critic call the built-in `exit_loop` tool when it approves. The ADK
  critic never rejected a draft in this run, as in the others.
- `langchain-mcp-adapters` opens a fresh MCP session per tool call (the server process is restarted each time); the native client keeps one session.
- ADK 2.x prints deprecation warnings for `SequentialAgent` and `LoopAgent` (it points to `Workflow`).
- Per-agent tracing exists only in the native version; the framework versions would use LangSmith or ADK's own event stream, which I did not wire up.
- These frameworks are not part of CI (heavy dependencies); the native flow is (tests in `tests/test_multi.py`).
