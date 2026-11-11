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

Results (model: `openai/gpt-oss-120b` on Groq, one run, 12 tasks):

| metric | value |
|---|---|
| tasks passed as run | **11 / 12** (91.7%) |
| tasks passed after fixing the answer matcher | **12 / 12** (see below) |
| average tool calls per task | 1.83 (schema, then one query; the write task used 1) |
| tool error rate | 0 of 22 tool calls |
| median tool-call latency (server side) | 30 ms |

The single miss was task 9 ("which customer has the most open tickets"). The agent's answer was correct
("Anaya Kapoor ... 5 open tickets") but it used a Unicode narrow no-break space between the two names, and my
original check looked for a plain space. I fixed the matcher (NFKC normalization, covered by tests) and re-scored
the saved answers without calling the model again (`python -m agent.rescore`, output in `results/eval_rescored.json`).
Both numbers are reported on purpose: 11/12 is what the original check said, 12/12 is after the check was corrected.
No agent answer was changed. The write task is verified by inspecting the database, and was kept as stored.

No end-to-end latency is reported: the free API tier rate-limits (8,000 tokens per minute), so per-task wall time in
`results/eval.json` is dominated by retry waits. A clean LLM call takes about 1 to 3 seconds (see `docs/sample_trace.jsonl`).

## Run
    pip install -r requirements.txt
    set LLM_API_KEY=...        # any OpenAI-compatible endpoint (see agent/agent.py)
    python -m agent.agent "Which customer has the most open tickets?"
    python -m agent.eval_tasks
    pytest

## Limits
Single agent, not a multi-agent system. SQLite demo data, one model, one run of 12 tasks (a pass rate on 12 tasks
is a smoke test, not a benchmark). Success is a normalized substring check on the final answer plus a database check
for the write task, so it can still miss correct answers phrased unusually. The tasks are simple (one or two
queries); multi-hop reasoning over the data is not tested.
