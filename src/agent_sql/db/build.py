"""Generate and seed the synthetic e-commerce SQLite DB.

Fully deterministic: same seed -> byte-for-byte identical table contents.
No datetime.now(), no unseeded randomness. Gold SQL results depend on this.
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from random import Random

from agent_sql.config import DB_SEED

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_CATEGORIES = [
    "Electronics", "Books", "Clothing", "Home & Kitchen",
    "Toys", "Sports", "Beauty", "Grocery",
]

# 5 products per category = 40 products.
_PRODUCTS: dict[str, list[str]] = {
    "Electronics": ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Bluetooth Speaker",
                    "Webcam HD"],
    "Books": ["Mystery Novel", "Cookbook Basics", "Sci-Fi Anthology", "History of Rome",
              "Python Primer"],
    "Clothing": ["Cotton T-Shirt", "Denim Jeans", "Wool Sweater", "Rain Jacket",
                 "Running Socks"],
    "Home & Kitchen": ["Chef Knife", "Cast Iron Pan", "French Press", "Cutting Board",
                       "Storage Jars"],
    "Toys": ["Building Blocks", "Puzzle 1000pc", "RC Car", "Plush Bear", "Board Game Classic"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Tennis Racket", "Cycling Helmet", "Water Bottle"],
    "Beauty": ["Face Moisturizer", "Shampoo Herbal", "Lip Balm Trio", "Sunscreen SPF50",
               "Hand Cream"],
    "Grocery": ["Olive Oil 1L", "Dark Chocolate", "Green Tea Box", "Almond Butter",
                "Pasta Bundle"],
}

_FIRST_NAMES = ["Aarav", "Bella", "Chen", "Diya", "Emil", "Fatima", "Grace", "Hiro",
                "Ines", "Jamal", "Kira", "Luca", "Mei", "Noah", "Olga", "Priya",
                "Quinn", "Ravi", "Sofia", "Tomas"]
_LAST_NAMES = ["Anderson", "Bhat", "Costa", "Dubois", "Eriksen", "Fernandez", "Gupta",
               "Haddad", "Ivanov", "Jensen", "Kim", "Lopez", "Mehta", "Nakamura",
               "Okafor", "Patel", "Qureshi", "Rossi", "Singh", "Tanaka"]
_CITIES = ["Austin", "Bangalore", "Chicago", "Denver", "Edinburgh",
           "Frankfurt", "Geneva", "Houston", "Istanbul", "Jakarta"]

_STATUSES = ["pending", "shipped", "delivered", "cancelled"]
_STATUS_WEIGHTS = [10, 25, 55, 10]

N_CUSTOMERS = 50
N_ORDERS = 200


def build_db(path: str | Path, seed: int = DB_SEED) -> dict[str, int]:
    """Create the DB at `path` (overwriting any existing file) and seed it.

    Returns a table -> row count dict for reporting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    rng = Random(seed)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA_PATH.read_text())

        for cat_id, cat_name in enumerate(_CATEGORIES, start=1):
            conn.execute("INSERT INTO categories VALUES (?, ?)", (cat_id, cat_name))

        product_prices: dict[int, float] = {}
        product_id = 0
        for cat_id, cat_name in enumerate(_CATEGORIES, start=1):
            for prod_name in _PRODUCTS[cat_name]:
                product_id += 1
                price = round(rng.uniform(2.99, 499.99), 2)
                product_prices[product_id] = price
                conn.execute(
                    "INSERT INTO products VALUES (?, ?, ?, ?)",
                    (product_id, prod_name, cat_id, price),
                )

        signup_base = date(2023, 1, 1)
        for cust_id in range(1, N_CUSTOMERS + 1):
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            signup = signup_base + timedelta(days=rng.randrange(365))
            conn.execute(
                "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
                (
                    cust_id,
                    f"{first} {last}",
                    f"{first.lower()}.{last.lower()}.{cust_id}@example.com",
                    rng.choice(_CITIES),
                    signup.isoformat(),
                ),
            )

        order_base = date(2024, 1, 1)
        item_id = 0
        n_items = 0
        for order_id in range(1, N_ORDERS + 1):
            order_date = order_base + timedelta(days=rng.randrange(547))  # thru 2025-06-30
            conn.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?)",
                (
                    order_id,
                    rng.randrange(1, N_CUSTOMERS + 1),
                    order_date.isoformat(),
                    rng.choices(_STATUSES, weights=_STATUS_WEIGHTS, k=1)[0],
                ),
            )
            for _ in range(rng.randint(1, 4)):
                item_id += 1
                n_items += 1
                pid = rng.randrange(1, product_id + 1)
                conn.execute(
                    "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
                    (item_id, order_id, pid, rng.randint(1, 5), product_prices[pid]),
                )

        conn.commit()
        return {
            "categories": len(_CATEGORIES),
            "products": product_id,
            "customers": N_CUSTOMERS,
            "orders": N_ORDERS,
            "order_items": n_items,
        }
    finally:
        conn.close()
