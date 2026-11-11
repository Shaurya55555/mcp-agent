# mcp-agent: an MCP server and a tracing tool-calling agent

An MCP server (stdio) exposing four tools over a seeded SQLite shop database, plus a multi-step agent that
connects as an MCP client, discovers the tools at runtime, and loops plan -> tool call -> observe until it can answer.
Every LLM call and tool call is traced, and the server keeps its own audit log.

    agent (LLM loop)  --MCP/stdio-->  shop_server.py  -->  SQLite
        |                                   |
   traces/<id>.jsonl                 logs/tool_calls.jsonl

## Tools (server/shop_server.py)
| tool | what it does | guard rails |
|---|---|---|
| `get_schema` | tables and columns | read-only connection |
| `run_sql` | one SELECT, max 50 rows | rejects non-SELECT and multi-statement input |
| `get_customer` | profile, order count, spend, open tickets | by id or email |
| `create_ticket` | opens a support ticket (the only write) | validates priority and customer |

The agent's system prompt only allows `create_ticket` when the user explicitly asks for one.

## Observability
- **Agent trace** (`agent/trace.py`): one JSONL file per run with a span per LLM call (latency, tokens,
  requested tools) and per tool call (args, latency, error, output). View with
  `python -m agent.view_trace traces/<id>.jsonl`. A real example is in `docs/sample_trace.jsonl`:

      0.00s  TASK: How many customers are in Bengaluru?
      7.36s  LLM  step=0 2946ms tokens=378 -> ['get_schema']
      8.18s  TOOL ok  get_schema {} (819ms)
      9.46s  LLM  step=1 1278ms tokens=556 -> ['run_sql']
      9.46s  TOOL ok  run_sql {"query": "SELECT COUNT(*) ... WHERE city = 'Bengaluru';"} (7ms)
     26.22s  ANSWER: There are **7** customers in Bengaluru.

- **Server audit log** (`logs/tool_calls.jsonl`): tool, args, latency and error for every call, written by the
  server regardless of which client called it.

## Evaluation
`agent/eval_tasks.py` runs 12 tasks (counts, aggregates, joins, lookups, schema listing, and one write).
Expected answers are computed from the database with SQL, so ground truth is correct by construction.
The write task is checked by inspecting the database afterwards, not by trusting the agent's words.

Results (model: `openai/gpt-oss-120b` on Groq): **pending, run in progress.**

| metric | value |
|---|---|
| task success rate | TBD |
| average tool calls per task | TBD |
| tool error rate | TBD |
| p50 seconds per task | TBD |

## Run
    pip install -r requirements.txt
    set LLM_API_KEY=...        # any OpenAI-compatible endpoint (see agent/agent.py)
    python -m agent.agent "Which customer has the most open tickets?"
    python -m agent.eval_tasks
    pytest

## Limits
Single agent, not a multi-agent system. SQLite demo data, one model, 12 tasks. Success is a substring check on
the final answer plus a database check for the write task, so it can miss correct answers phrased unusually.
