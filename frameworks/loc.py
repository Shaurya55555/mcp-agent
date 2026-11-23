"""Lines of code (non-blank, non-comment, docstrings excluded) of each implementation of the same flow."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def loc(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    doc_lines = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            d = node.body[0]
            doc_lines.update(range(d.lineno, d.end_lineno + 1))
    return sum(1 for i, l in enumerate(src.splitlines(), 1) if l.strip() and not l.strip().startswith("#") and i not in doc_lines)


if __name__ == "__main__":
    files = {"native (agent/multi.py)": "agent/multi.py", "native shared loop + tracing (agent/agent.py)": "agent/agent.py",
             "native tracer (agent/trace.py)": "agent/trace.py", "LangGraph (frameworks/langgraph_flow.py)": "frameworks/langgraph_flow.py",
             "Google ADK (frameworks/adk_flow.py)": "frameworks/adk_flow.py"}
    for label, f in files.items():
        print(f"{loc(ROOT / f):4d}  {label}")
