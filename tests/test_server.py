import pytest

from server import seed, shop_server as s


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    p = seed.build(tmp_path / "t.db")
    monkeypatch.setattr(s, "DB", p)
    monkeypatch.setattr(s, "LOG", tmp_path / "log.jsonl")
    return p


def test_schema_has_tables():
    assert set(s.get_schema()) == {"customers", "orders", "products", "tickets"}


def test_sql_rejects_writes():
    with pytest.raises(ValueError):
        s.run_sql("DELETE FROM customers")
    with pytest.raises(ValueError):
        s.run_sql("SELECT 1; DROP TABLE customers")


def test_sql_select_and_limit():
    out = s.run_sql("SELECT * FROM orders")
    assert len(out["rows"]) == 50 and out["truncated"] is True


def test_customer_and_ticket_roundtrip():
    before = s.get_customer(customer_id=3)["open_tickets"]
    t = s.create_ticket(3, "Test", "high")
    assert s.get_customer(customer_id=3)["open_tickets"] == before + 1
    assert t["ticket_id"] > 40


def test_audit_log_written():
    s.get_schema()
    assert s.LOG.read_text().count("get_schema") == 1
