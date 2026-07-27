"""PostgreSQL database connection and data loading utilities."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # type: ignore

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "database" / "schema.sql"


def get_connection_params() -> dict[str, str]:
    """Read database connection parameters from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "sales_analytics"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }


@contextmanager
def get_connection() -> Generator:
    """
    Context manager for PostgreSQL connections.

    Yields:
        psycopg2 connection object.

    Raises:
        ImportError: If psycopg2 is not installed.
        ConnectionError: If connection fails.
    """
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for database operations. "
            "Install with: pip install psycopg2-binary"
        )

    params = get_connection_params()
    conn = None
    try:
        conn = psycopg2.connect(**params)
        yield conn
        conn.commit()
    except Exception as exc:
        if conn:
            conn.rollback()
        raise ConnectionError(f"Database connection failed: {exc}") from exc
    finally:
        if conn:
            conn.close()


def init_schema(schema_path: Path | None = None) -> None:
    """Execute schema SQL to create tables."""
    path = schema_path or SCHEMA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, encoding="utf-8") as f:
        schema_sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)


def load_customers(df: pd.DataFrame) -> None:
    """Insert unique customers into the customers table."""
    customers = (
        df[["Customer_ID", "Customer_Name"]]
        .drop_duplicates(subset=["Customer_ID"])
        .values.tolist()
    )
    query = """
        INSERT INTO customers (customer_id, customer_name)
        VALUES %s
        ON CONFLICT (customer_id) DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, customers)


def load_products(df: pd.DataFrame) -> None:
    """Insert unique products into the products table."""
    products = (
        df[["Product_Name", "Product_Category"]]
        .drop_duplicates(subset=["Product_Name"])
        .values.tolist()
    )
    query = """
        INSERT INTO products (product_name, product_category)
        VALUES %s
        ON CONFLICT (product_name) DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, products)


def load_orders_and_sales(df: pd.DataFrame) -> None:
    """Insert order and sales records into respective tables."""
    orders = df[
        ["Order_ID", "Order_Date", "Customer_ID", "Region", "Sales_Channel"]
    ].drop_duplicates(subset=["Order_ID"])

    order_rows = [
        (
            row.Order_ID,
            row.Order_Date.strftime("%Y-%m-%d")
            if hasattr(row.Order_Date, "strftime")
            else str(row.Order_Date),
            row.Customer_ID,
            row.Region,
            row.Sales_Channel,
        )
        for row in orders.itertuples(index=False)
    ]

    sales_rows = [
        (
            row.Order_ID,
            row.Product_Name,
            int(row.Quantity),
            float(row.Unit_Price),
            float(row.Discount),
            float(row.Revenue),
            float(row.Profit),
        )
        for row in df.itertuples(index=False)
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO orders (order_id, order_date, customer_id, region, sales_channel)
                VALUES %s
                ON CONFLICT (order_id) DO NOTHING
                """,
                order_rows,
            )
            execute_values(
                cur,
                """
                INSERT INTO sales (order_id, product_name, quantity, unit_price,
                                   discount, revenue, profit)
                VALUES %s
                """,
                sales_rows,
            )


def load_all_data(df: pd.DataFrame) -> None:
    """Load full dataset into PostgreSQL tables."""
    init_schema()
    load_customers(df)
    load_products(df)
    load_orders_and_sales(df)


def query_sales_summary() -> pd.DataFrame:
    """Fetch aggregated sales summary from the database."""
    query = """
        SELECT
            o.region,
            p.product_category,
            DATE_TRUNC('month', o.order_date) AS month,
            SUM(s.revenue) AS total_revenue,
            SUM(s.profit) AS total_profit,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM sales s
        JOIN orders o ON s.order_id = o.order_id
        JOIN products p ON s.product_name = p.product_name
        GROUP BY o.region, p.product_category, DATE_TRUNC('month', o.order_date)
        ORDER BY month, total_revenue DESC
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)
