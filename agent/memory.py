"""Agent memory: short-term state inside a session, long-term task summaries across sessions.

Short-term  `Session` keeps this session's earlier turns (task, answer) and replays them to the model.
Long-term   a `MemoryStore` keeps one summary per finished task and finds relevant ones for a new task.
            PostgreSQL (full-text search with ts_rank_cd) when MEMORY_DSN starts with postgres://, otherwise SQLite.
            Both backends implement the same interface and are covered by the same tests.

Recall is keyword-based (Postgres full-text search, or term overlap on SQLite), not semantic: a question that shares
no words with the stored summary will not find it. Recalled text is injected into the prompt as a hint labelled as
possibly stale, because facts about the shop database can change between sessions.

    store = open_memory()                       # MEMORY_DSN env var, default sqlite file data/memory.db
    s = Session(store)                          # new session id; many sessions can share one store
    s.record("My region is Bengaluru", "Noted.")
    Session(open_memory()).recall("which region do I manage")   # a later session finds it
"""
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE = ROOT / "data" / "memory.db"
MAX_SUMMARY_CHARS = 600

_STOP = set("""a an and are as at be but by can do does for from had has have how i if in into is it its me my of on or
our so than that the their them then there these they this to us was we were what when where which who whom why will
with would you your""".split())


def terms(text: str) -> list[str]:
    """Lower-case content words (3+ chars, not stop words), deduplicated, order kept."""
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) >= 3 and w not in _STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out


@dataclass
class Memory:
    id: int
    session_id: str
    task: str
    summary: str
    created_at: float
    score: float = 0.0


def make_summary(task: str, answer: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Deterministic task summary (no LLM call): the question and the answer, whitespace-collapsed and capped."""
    text = f"Q: {' '.join(task.split())} A: {' '.join((answer or '').split())}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


class SqliteMemory:
    def __init__(self, path: str | Path = DEFAULT_SQLITE):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
                        " task TEXT NOT NULL, summary TEXT NOT NULL, created_at REAL NOT NULL)")
        self.db.commit()

    def remember(self, session_id: str, task: str, summary: str) -> int:
        cur = self.db.execute("INSERT INTO memories(session_id, task, summary, created_at) VALUES(?,?,?,?)",
                              (session_id, task, summary, time.time()))
        self.db.commit()
        return cur.lastrowid

    def recall(self, query: str, k: int = 3, exclude_session: str | None = None) -> list[Memory]:
        q = set(terms(query))
        if not q:
            return []
        rows = self.db.execute("SELECT id, session_id, task, summary, created_at FROM memories ORDER BY id DESC LIMIT 500")
        scored = []
        for r in rows:
            if exclude_session and r[1] == exclude_session:
                continue
            overlap = len(q & set(terms(r[2] + " " + r[3])))
            if overlap:
                scored.append(Memory(*r, score=overlap / len(q)))
        scored.sort(key=lambda m: (-m.score, -m.created_at))
        return scored[:k]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def clear(self):
        self.db.execute("DELETE FROM memories")
        self.db.commit()

    def close(self):
        self.db.close()


class PostgresMemory:
    """Long-term memory in PostgreSQL. Needs `psycopg` (pip install "psycopg[binary]")."""

    def __init__(self, dsn: str):
        import psycopg  # imported lazily so SQLite-only users do not need it

        self.db = psycopg.connect(dsn, autocommit=True)
        self.db.execute("""CREATE TABLE IF NOT EXISTS agent_memories (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            task TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', task || ' ' || summary)) STORED)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS agent_memories_tsv ON agent_memories USING GIN (tsv)")

    def remember(self, session_id: str, task: str, summary: str) -> int:
        row = self.db.execute("INSERT INTO agent_memories(session_id, task, summary) VALUES (%s,%s,%s) RETURNING id",
                              (session_id, task, summary)).fetchone()
        return row[0]

    def recall(self, query: str, k: int = 3, exclude_session: str | None = None) -> list[Memory]:
        ts = terms(query)
        if not ts:
            return []
        tsquery = " | ".join(ts)  # terms are [a-z0-9]+ only, so this is safe to pass as a to_tsquery expression
        rows = self.db.execute(
            """SELECT id, session_id, task, summary, EXTRACT(EPOCH FROM created_at), ts_rank_cd(tsv, q) AS score
               FROM agent_memories, to_tsquery('english', %s) q
               WHERE tsv @@ q AND (%s::text IS NULL OR session_id <> %s)
               ORDER BY score DESC, created_at DESC LIMIT %s""", (tsquery, exclude_session, exclude_session, k)).fetchall()
        return [Memory(r[0], r[1], r[2], r[3], float(r[4]), float(r[5])) for r in rows]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM agent_memories").fetchone()[0]

    def clear(self):
        self.db.execute("TRUNCATE agent_memories RESTART IDENTITY")

    def close(self):
        self.db.close()


def open_memory(dsn: str | None = None):
    """MEMORY_DSN=postgresql://user:pass@host/db  -> PostgreSQL; anything else (or unset) -> SQLite file."""
    dsn = dsn or os.getenv("MEMORY_DSN", "")
    if dsn.startswith(("postgres://", "postgresql://")):
        return PostgresMemory(dsn)
    return SqliteMemory(dsn or DEFAULT_SQLITE)


@dataclass
class Session:
    """One conversation. Short-term: its own turns. Long-term: the shared store."""
    store: object | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    turns: list[tuple[str, str]] = field(default_factory=list)
    max_turns: int = 6
    recall_k: int = 3
    last_recalled: int = 0

    def recall(self, query: str) -> list[Memory]:
        if self.store is None:
            return []
        return self.store.recall(query, k=self.recall_k, exclude_session=self.session_id)

    def history_messages(self) -> list[dict]:
        """Earlier turns of THIS session as chat messages (short-term memory)."""
        msgs = []
        for task, answer in self.turns[-self.max_turns:]:
            msgs += [{"role": "user", "content": task}, {"role": "assistant", "content": answer}]
        return msgs

    def memory_prompt(self, task: str) -> str:
        """System-prompt addendum with long-term memories relevant to `task` ('' when there are none)."""
        found = self.recall(task)
        self.last_recalled = len(found)
        if not found:
            return ""
        lines = "\n".join(f"- {m.summary}" for m in found)
        return ("\n\nNotes from earlier sessions (may be out of date; for anything about current database contents, "
                "check with the tools):\n" + lines)

    def record(self, task: str, answer: str) -> int | None:
        """End of a turn: keep it in short-term memory and write a summary to long-term memory."""
        self.turns.append((task, answer))
        if self.store is None:
            return None
        return self.store.remember(self.session_id, task, make_summary(task, answer))
