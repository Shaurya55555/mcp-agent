"""12 tasks whose ground truth is computed from the DB, so expected answers are correct by construction.

python -m agent.eval_tasks   # prints success rate, avg tool calls, p50 latency
"""
import asyncio
import json
import shutil
import sqlite3
import statistics
from pathlib import Path

from server import seed
from .agent import run_task

ROOT = Path(__file__).resolve().parents[1]
EVAL_DB = ROOT / "data" / "eval.db"


def q(db, sql, *a):
    c = sqlite3.connect(db)
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def tasks(db):
    top_c = q(db, "SELECT c.name, COUNT(*) n FROM tickets t JOIN customers c ON c.id=t.customer_id "
                  "WHERE t.status='open' GROUP BY c.id ORDER BY n DESC, c.id LIMIT 2")
    rev = q(db, "SELECT SUM(o.qty*p.price) FROM orders o JOIN products p ON p.id=o.product_id "
                "WHERE o.status='delivered'")[0][0]
    best_prod = q(db, "SELECT p.name, SUM(o.qty) s FROM orders o JOIN products p ON p.id=o.product_id "
                      "GROUP BY p.id ORDER BY s DESC LIMIT 1")[0]
    blr = q(db, "SELECT COUNT(*) FROM customers WHERE city='Bengaluru'")[0][0]
    c7 = q(db, "SELECT name, email, city FROM customers WHERE id=7")[0]
    n7 = q(db, "SELECT COUNT(*) FROM orders WHERE customer_id=7 AND status!='cancelled'")[0][0]
    hi = q(db, "SELECT COUNT(*) FROM tickets WHERE priority='high' AND status='open'")[0][0]
    canc = q(db, "SELECT COUNT(*) FROM orders WHERE status='cancelled'")[0][0]
    city_top = q(db, "SELECT city, COUNT(*) n FROM customers GROUP BY city ORDER BY n DESC, city LIMIT 1")[0]
    pricey = q(db, "SELECT name, price FROM products ORDER BY price DESC LIMIT 1")[0]
    n_before = q(db, "SELECT COUNT(*) FROM tickets")[0][0]
    return [
        ("How many customers are in Bengaluru?", [str(blr)], None),
        ("What is the total revenue (qty x price) of delivered orders?", [f"{rev:,}", str(rev)], None),
        ("Which product has the most units ordered across all orders, and how many units?",
         [best_prod[0], str(best_prod[1])], None),
        ("What is the email of customer 7, and how many non-cancelled orders have they placed?",
         [c7[1], str(n7)], None),
        ("How many high-priority tickets are currently open?", [str(hi)], None),
        ("How many orders were cancelled?", [str(canc)], None),
        ("Which city has the most customers, and how many?", [city_top[0], str(city_top[1])], None),
        ("What is the most expensive product and its price?", [pricey[0], str(pricey[1])], None),
        ("Which customer has the most open tickets? Give name and count.", [top_c[0][0], str(top_c[0][1])], None),
        ("Where does customer 7 live and what is their name?", [c7[0], c7[2]], None),
        ("List the tables in the database.", ["customers", "orders", "products", "tickets"], None),
        ("Open a high-priority ticket for customer 3 with subject 'Refund not received'. Tell me the new ticket id.",
         ["ticket"], ("write", n_before)),
    ]


def any_form(expect, answer):
    """Each expected item may be a plain string; compare ignoring thousands separators."""
    a = answer.lower().replace(",", "")
    return all(e.lower().replace(",", "") in a for e in expect)


async def main():
    seed.build(EVAL_DB)
    work = ROOT / "data" / "eval_work.db"
    shutil.copy(EVAL_DB, work)
    results = []
    for i, (task, expect, extra) in enumerate(tasks(EVAL_DB), 1):
        res = await run_task(task, str(work))
        ok = any_form(expect, res["answer"])
        if extra and extra[0] == "write":
            rows = q(work, "SELECT customer_id, priority, subject FROM tickets WHERE id > ?", extra[1])
            ok = ok and rows == [(3, "high", "Refund not received")]
        res.update(task=task, ok=ok)
        results.append(res)
        print(f"{'PASS' if ok else 'FAIL'} [{i:02d}] tools={res['tool_calls']} {res['seconds']}s  {task[:70]}", flush=True)
        if not ok:
            print("      answer:", res["answer"][:200])
    n = len(results)
    summary = {
        "tasks": n,
        "success_rate": round(sum(r["ok"] for r in results) / n, 3),
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in results), 2),
        "tool_error_rate": round(sum(r["tool_errors"] for r in results) / max(1, sum(r["tool_calls"] for r in results)), 3),
        "p50_seconds": round(statistics.median(r["seconds"] for r in results), 2),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in results)),
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
