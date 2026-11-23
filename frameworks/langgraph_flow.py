"""The planner -> worker -> critic flow rebuilt with LangGraph. Same prompts, same MCP server, same model as agent/multi.py.

python -m frameworks.langgraph_flow "How many customers are in Bengaluru?"
Needs: pip install -r frameworks/requirements.txt
"""
import asyncio
import json
import sys
import time
from typing import Annotated, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent import agent as base
from agent.multi import CRITIC_SYSTEM, PLANNER_SYSTEM, WORKER_SYSTEM, parse_json_object


class Counter(BaseCallbackHandler):
    """Counts every LLM call (planner, worker, critic) and its tokens."""

    def __init__(self):
        self.calls, self.tokens = 0, 0

    def on_llm_end(self, response, **kw):
        self.calls += 1
        msg = response.generations[0][0].message
        self.tokens += (getattr(msg, "usage_metadata", None) or {}).get("total_tokens", 0)


class State(TypedDict):
    task: str
    plan: list[str]
    messages: Annotated[list, add_messages]
    draft: str
    approved: bool
    feedback: str
    revisions: int


def build(tools, llm, use_critic=True, max_revisions=1):
    worker_llm = llm.bind_tools(tools)

    def planner(s: State):
        names = ", ".join(t.name for t in tools)
        out = llm.invoke([SystemMessage(PLANNER_SYSTEM.format(tools=names)), HumanMessage(s["task"])])
        obj = parse_json_object(out.content)
        steps = [str(x).strip() for x in (obj or {}).get("steps", []) if str(x).strip()][:4]
        plan = steps or [s["task"]]
        plan_text = "\n".join(f"{i}. {x}" for i, x in enumerate(plan, 1))
        return {"plan": plan, "messages": [SystemMessage(WORKER_SYSTEM.format(plan=plan_text)), HumanMessage(s["task"])]}

    def worker(s: State):
        return {"messages": [worker_llm.invoke(s["messages"])]}

    def after_worker(s: State):
        last = s["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "critic" if use_critic else "finish"

    def finish(s: State):
        return {"draft": s["messages"][-1].content}

    def critic(s: State):
        draft = s["messages"][-1].content
        obs = "\n".join(f"{m.name}(...) -> {str(m.content)[:600]}" for m in s["messages"] if m.type == "tool")[-3000:]
        out = llm.invoke([SystemMessage(CRITIC_SYSTEM),
                          HumanMessage(f"TASK: {s['task']}\n\nOBSERVATIONS:\n{obs or '(no tool calls were made)'}\n\nDRAFT: {draft}")])
        v = parse_json_object(out.content)
        approved = True if v is None or "approved" not in v else bool(v["approved"])  # unusable output fails open
        return {"draft": draft, "approved": approved, "feedback": str((v or {}).get("feedback", ""))[:300]}

    def after_critic(s: State):
        return END if s["approved"] or s["revisions"] >= max_revisions else "revise"

    def revise(s: State):
        return {"revisions": s["revisions"] + 1,
                "messages": [HumanMessage("Reviewer feedback: " + (s["feedback"] or "not approved")
                                          + ". Fix this using the tools if needed and give the corrected final answer.")]}

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("worker", worker), ("tools", ToolNode(tools)), ("critic", critic),
                     ("revise", revise), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "worker")
    g.add_conditional_edges("worker", after_worker, {"tools": "tools", "critic": "critic", "finish": "finish"})
    g.add_edge("tools", "worker")
    g.add_conditional_edges("critic", after_critic, {END: END, "revise": "revise"})
    g.add_edge("revise", "worker")
    g.add_edge("finish", END)
    return g.compile()


async def run_task(task: str, db: str | None = None, use_critic: bool = True) -> dict:
    env = {**__import__("os").environ, **({"SHOP_DB": db} if db else {})}
    client = MultiServerMCPClient({"shop": {"command": sys.executable, "args": [str(base.ROOT / "server" / "shop_server.py")],
                                            "transport": "stdio", "env": env}})
    tools = await client.get_tools()
    llm = ChatOpenAI(model=base.MODEL, base_url=base.BASE, api_key=base.KEY, temperature=0, max_retries=6)
    graph = build(tools, llm, use_critic)
    t0, counter = time.time(), Counter()
    final = await graph.ainvoke({"task": task, "messages": [], "revisions": 0},
                                {"recursion_limit": 6 * base.MAX_STEPS, "callbacks": [counter]})
    msgs = final["messages"]
    ai = [m for m in msgs if isinstance(m, AIMessage)]
    return {"answer": final.get("draft") or "", "plan": final["plan"], "revisions": final["revisions"],
            "llm_calls": counter.calls,
            "tool_calls": sum(len(m.tool_calls) for m in ai), "tool_errors": sum(1 for m in msgs if m.type == "tool" and m.status == "error"),
            "tokens": counter.tokens, "seconds": round(time.time() - t0, 2), "active_seconds": 0.0, "agent_turns": 0}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_task(" ".join(sys.argv[1:]) or "How many customers are there?")), indent=2))
