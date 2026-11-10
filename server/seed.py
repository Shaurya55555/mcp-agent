"""Create a deterministic demo SQLite DB (customers, products, orders, tickets)."""
import random
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "shop.db"

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vikram", "Anaya", "Arjun", "Saanvi",
         "Dev", "Tara", "Neel", "Riya", "Karan", "Zoya", "Ishaan", "Pooja", "Aditya", "Nisha"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Gupta", "Reddy", "Mehta", "Das", "Kapoor", "Rao"]
PRODUCTS = [("Laptop Stand", 1499), ("USB-C Hub", 2299), ("Mechanical Keyboard", 5499), ("Webcam HD", 3199),
            ("Monitor Arm", 4299), ("Noise Cancelling Headset", 8999), ("Desk Mat", 999), ("Cable Organizer", 399)]
STATUS = ["pending", "shipped", "delivered", "delivered", "delivered", "cancelled"]
SUBJECTS = ["Late delivery", "Wrong item received", "Refund not received", "Product defective",
            "Change shipping address", "Invoice request"]


def build(path: Path = DB, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rnd = random.Random(seed)
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, city TEXT, joined TEXT);
    CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, price INTEGER);
    CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, qty INTEGER,
                        status TEXT, ordered_on TEXT);
    CREATE TABLE tickets(id INTEGER PRIMARY KEY, customer_id INTEGER, subject TEXT, priority TEXT,
                         status TEXT, created_on TEXT);
    """)
    cities = ["Bengaluru", "Jaipur", "Pune", "Delhi", "Mumbai", "Hyderabad"]
    for i in range(1, 21):
        n = f"{FIRST[i - 1]} {rnd.choice(LAST)}"
        c.execute("INSERT INTO customers VALUES(?,?,?,?,?)",
                  (i, n, f"{n.split()[0].lower()}{i}@example.com", rnd.choice(cities), f"2024-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"))
    for i, (n, p) in enumerate(PRODUCTS, 1):
        c.execute("INSERT INTO products VALUES(?,?,?)", (i, n, p))
    for i in range(1, 121):
        c.execute("INSERT INTO orders VALUES(?,?,?,?,?,?)",
                  (i, rnd.randint(1, 20), rnd.randint(1, len(PRODUCTS)), rnd.randint(1, 3), rnd.choice(STATUS),
                   f"2025-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"))
    for i in range(1, 41):
        c.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?)",
                  (i, rnd.randint(1, 20), rnd.choice(SUBJECTS), rnd.choice(["low", "medium", "high"]),
                   rnd.choice(["open", "open", "closed"]), f"2025-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"))
    c.commit()
    c.close()
    return path


if __name__ == "__main__":
    print(build())
