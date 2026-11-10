"""MCP server exposing 4 tools over the shop database.

Run (stdio):  python server/shop_server.py
Every tool call is appended to logs/tool_calls.jsonl (server-side audit log).
"""
import functools
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.getenv("SHOP_DB", seed.DB))
LOG = ROOT / "logs" / "tool_calls.jsonl"
MAX_ROWS = 50

mcp = FastMCP("shop")


def conn(readonly: bool = True) -> sqlite3.Connection:
    if not DB.exists():
        seed.build(DB)
    c = sqlite3.connect(f"file:{DB.as_posix()}?mode={'ro' if readonly else 'rw'}", uri=True)
    c.row_factory = sqlite3.Row
    return c


def audited(fn):
    """Log tool name, args, latency, and outcome for every call."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0, err, out = time.time(), None, None
        try:
            out = fn(*a, **kw)
            return out
        except Exception as e:  # surfaced to the model as tool error text
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            LOG.parent.mkdir(exist_ok=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": t0, "tool": fn.__name__, "args": kw or list(a),
                                    "ms": round((time.time() - t0) * 1000, 1), "error": err,
                                    "result_chars": len(json.dumps(out)) if out is not None else 0}) + "\n")
    return wrapper


@mcp.tool()
@audited
def get_schema() -> dict:
    """Return all tables and their columns in the shop database."""
    with conn() as c:
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {t: [r["name"] + " " + r["type"] for r in c.execute(f"PRAGMA table_info({t})")] for t in tables}


@mcp.tool()
@audited
def run_sql(query: str) -> dict:
    """Run ONE read-only SELECT query (max 50 rows returned). Use get_schema first."""
    q = query.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", q) or ";" in q:
        raise ValueError("only a single SELECT statement is allowed")
    with conn() as c:
        rows = c.execute(q).fetchmany(MAX_ROWS + 1)
    return {"rows": [dict(r) for r in rows[:MAX_ROWS]], "truncated": len(rows) > MAX_ROWS}


@mcp.tool()
@audited
def get_customer(customer_id: int | None = None, email: str | None = None) -> dict:
    """Look up a customer by id or email; includes order count, total spend and open tickets."""
    if customer_id is None and not email:
        raise ValueError("provide customer_id or email")
    with conn() as c:
        row = c.execute("SELECT * FROM customers WHERE id=? OR email=?", (customer_id, email)).fetchone()
        if not row:
            raise LookupError("customer not found")
        cid = row["id"]
        stats = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(o.qty*p.price),0) spend FROM orders o JOIN products p ON p.id=o.product_id "
            "WHERE o.customer_id=? AND o.status!='cancelled'", (cid,)).fetchone()
        open_t = c.execute("SELECT COUNT(*) FROM tickets WHERE customer_id=? AND status='open'", (cid,)).fetchone()[0]
    return {**dict(row), "orders": stats["n"], "total_spend": stats["spend"], "open_tickets": open_t}


@mcp.tool()
@audited
def create_ticket(customer_id: int, subject: str, priority: str = "medium") -> dict:
    """Open a new support ticket for a customer. priority is low, medium or high."""
    if priority not in {"low", "medium", "high"}:
        raise ValueError("priority must be low, medium or high")
    c = conn(readonly=False)
    try:
        if not c.execute("SELECT 1 FROM customers WHERE id=?", (customer_id,)).fetchone():
            raise LookupError("customer not found")
        cur = c.execute("INSERT INTO tickets(customer_id,subject,priority,status,created_on) "
                        "VALUES(?,?,?,'open',date('now'))", (customer_id, subject[:200], priority))
        c.commit()
        return {"ticket_id": cur.lastrowid, "status": "open"}
    finally:
        c.close()


if __name__ == "__main__":
    mcp.run()
