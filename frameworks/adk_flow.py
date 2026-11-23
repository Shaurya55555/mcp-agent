"""The planner -> worker -> critic flow rebuilt with Google ADK. Same MCP server and model (via LiteLLM -> Groq).

python -m frameworks.adk_flow "How many customers are in Bengaluru?"
Needs: pip install -r frameworks/requirements.txt

Structure: SequentialAgent[planner, LoopAgent[worker, critic]]. The plan and draft travel through session state
(`output_key`), the critic ends the loop by calling the built-in `exit_loop` tool when it approves.
"""
import asyncio
import json
import os
import sys
import time

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import exit_loop
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

from agent import agent as base

os.environ.setdefault("GROQ_API_KEY", base.KEY)

PLANNER = ("You are the planner for a support-operations assistant that can use database tools (get_schema, run_sql, "
           "get_customer, create_ticket). Break the user's task into the fewest ordered steps (1 to 4) a tool-using worker "
           "should carry out. Do not answer the task and do not invent data. Reply with a short numbered list only.")
WORKER = (base.SYSTEM + " You are the worker agent. A planner prepared these steps for you:\n{plan}\n"
          "Follow them, adapt if the data says otherwise, and finish with the final answer. "
          "If a reviewer left feedback, fix it first: {feedback?}")
CRITIC = ("You are a strict reviewer. Review the worker's DRAFT answer: {draft}\nFor the task in the conversation, approve only "
          "if it answers the task and every concrete value it states was returned by a tool call in this conversation "
          "(or the task asked for a write and the tool call succeeded). If it is acceptable, call the exit_loop tool and "
          "say nothing else. Otherwise reply with one short sentence of feedback and do not call any tool.")


def build(db: str | None = None, use_critic: bool = True):
    env = {**os.environ, **({"SHOP_DB": db} if db else {})}
    toolset = McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command=sys.executable, args=[str(base.ROOT / "server" / "shop_server.py")], env=env),
        timeout=30))
    model = LiteLlm(model=f"groq/{base.MODEL}", temperature=0, num_retries=6)
    planner = LlmAgent(name="planner", model=model, instruction=PLANNER, output_key="plan")
    worker = LlmAgent(name="worker", model=model, instruction=WORKER, tools=[toolset], output_key="draft")
    if not use_critic:
        return SequentialAgent(name="flow", sub_agents=[planner, worker]), toolset
    critic = LlmAgent(name="critic", model=model, instruction=CRITIC, tools=[exit_loop], output_key="feedback")
    loop = LoopAgent(name="review_loop", sub_agents=[worker, critic], max_iterations=2)
    return SequentialAgent(name="flow", sub_agents=[planner, loop]), toolset


async def run_task(task: str, db: str | None = None, use_critic: bool = True) -> dict:
    agent, toolset = build(db, use_critic)
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="shop", session_service=svc)
    await svc.create_session(app_name="shop", user_id="u", session_id="s")
    t0, llm_calls, tool_calls, tokens, errors, answer = time.time(), 0, 0, 0, 0, ""
    try:
        async for ev in runner.run_async(user_id="u", session_id="s",
                                         new_message=types.Content(role="user", parts=[types.Part(text=task)])):
            if ev.usage_metadata:
                llm_calls += 1
                tokens += ev.usage_metadata.total_token_count or 0
            for fc in ev.get_function_calls():
                if fc.name != "exit_loop":
                    tool_calls += 1
            for fr in ev.get_function_responses():
                if fr.name != "exit_loop" and isinstance((fr.response or {}).get("isError"), bool) and fr.response["isError"]:
                    errors += 1
        sess = await svc.get_session(app_name="shop", user_id="u", session_id="s")
        answer = sess.state.get("draft", "")
    finally:
        await toolset.close()
    return {"answer": answer, "plan": sess.state.get("plan", ""), "llm_calls": llm_calls, "tool_calls": tool_calls,
            "tool_errors": errors, "tokens": tokens, "seconds": round(time.time() - t0, 2), "active_seconds": 0.0,
            "agent_turns": 0, "revisions": 0}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_task(" ".join(sys.argv[1:]) or "How many customers are there?")), indent=2, default=str))
