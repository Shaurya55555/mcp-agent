# Design decisions and likely questions: mcp-agent

**What is MCP and why use it here instead of plain function calling?**
The Model Context Protocol standardizes how a client discovers and calls tools over a transport (stdio here). The
agent never imports the tool code: it asks the server for its tool list and schemas at runtime and converts them to
the LLM's function-calling format. Any MCP client (this agent, an IDE, another agent) can use the same server unchanged.

**How is the agent loop built?**
Plan, call, observe: send messages and tool schemas to the LLM; if it returns tool calls, execute each through the MCP
session, append the results, and loop; stop on a plain answer or after a step limit (8). Tool errors are returned to
the model as text, so it can correct a bad query.

**How do you keep a database tool safe?**
`run_sql` accepts only a single SELECT and uses a read-only SQLite connection, so even a bypassed regex cannot write.
Only `create_ticket` writes, with validated arguments, and the system prompt allows it only on an explicit request.

**What is traced and why two logs?**
The agent trace records every LLM call (latency, tokens, requested tools) and every tool call (args, latency, error).
The server's audit log records tool calls from any client, so it still works when the caller is not my agent.

**How is the agent evaluated without grading by eye?**
12 tasks whose expected values are computed from the database with SQL, so ground truth is correct by construction.
Substring checks on the final answer, plus a database inspection for the write task.

**Is this multi-agent?**
No. It is a single tool-using agent with multi-step reasoning. Calling it multi-agent would be wrong; the next step
would be a planner and a SQL specialist sharing the same MCP server.

**A bug worth telling:** my audit decorator hid the real function signature, so the model was told every tool took
arguments `a` and `kw`. Fixed with `functools.wraps`. Lesson: check the schema the model actually receives.
