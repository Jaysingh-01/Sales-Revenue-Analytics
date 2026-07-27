"""Data loading, cleaning, and KPI calculation pipeline."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load sales data from CSV file.

    Args:
        file_path: Path to CSV. Defaults to data/sales_data.csv.

    Returns:
        Raw sales DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file cannot be parsed.
    """
    path = Path(file_path) if file_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sales data file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to load data from {path}: {exc}") from exc

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values using business rules.

    - Discount: fill with 0 (no discount applied)
    - Profit: derive from Revenue when missing (assume 30% margin)
    - Other numeric columns: median imputation
    - Categorical columns: fill with 'Unknown'
    """
    cleaned = df.copy()

    if "Discount" in cleaned.columns:
        cleaned["Discount"] = cleaned["Discount"].fillna(0)

    if "Profit" in cleaned.columns and "Revenue" in cleaned.columns:
        profit_missing = cleaned["Profit"].isna()
        cleaned.loc[profit_missing, "Profit"] = cleaned.loc[profit_missing, "Revenue"] * 0.30

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned


def remove_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """
    Remove duplicate records.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for duplication. Defaults to Order_ID.
    """
    key = subset or ["Order_ID"]
    return df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def convert_dates(df: pd.DataFrame, date_column: str = "Order_Date") -> pd.DataFrame:
    """Convert date column to datetime and add derived time features."""
    converted = df.copy()
    converted[date_column] = pd.to_datetime(converted[date_column], errors="coerce")
    converted = converted.dropna(subset=[date_column])

    converted["Year"] = converted[date_column].dt.year
    converted["Month"] = converted[date_column].dt.month
    converted["YearMonth"] = converted[date_column].dt.to_period("M").astype(str)
    converted["Quarter"] = converted[date_column].dt.quarter

    return converted


def calculate_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate executive and operational KPIs.

    Returns:
        Dictionary containing revenue, profit, growth, and ranking metrics.
    """
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_profit": 0.0,
            "average_order_value": 0.0,
            "total_customers": 0,
            "total_orders": 0,
            "monthly_revenue_growth": pd.DataFrame(),
            "top_selling_products": pd.DataFrame(),
            "best_performing_regions": pd.DataFrame(),
        }

    total_revenue = float(df["Revenue"].sum())
    total_profit = float(df["Profit"].sum())
    total_orders = int(df["Order_ID"].nunique())
    total_customers = int(df["Customer_ID"].nunique())
    average_order_value = total_revenue / total_orders if total_orders else 0.0

    monthly = (
        df.groupby("YearMonth", as_index=False)["Revenue"]
        .sum()
        .sort_values("YearMonth")
    )
    monthly["Revenue_Growth_Pct"] = monthly["Revenue"].pct_change() * 100

    top_products = (
        df.groupby("Product_Name", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
    )

    best_regions = (
        df.groupby("Region", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
        .sort_values("Revenue", ascending=False)
    )

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "average_order_value": average_order_value,
        "total_customers": total_customers,
        "total_orders": total_orders,
        "monthly_revenue_growth": monthly,
        "top_selling_products": top_products,
        "best_performing_regions": best_regions,
    }


def preprocess_pipeline(file_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run full preprocessing pipeline: load, clean, dedupe, convert dates, KPIs.

    Returns:
        Tuple of (cleaned DataFrame, KPI dictionary).
    """
    raw = load_data(file_path)
    cleaned = handle_missing_values(raw)
    cleaned = remove_duplicates(cleaned)
    cleaned = convert_dates(cleaned)
    kpis = calculate_kpis(cleaned)
    return cleaned, kpis


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple | None = None,
    regions: list[str] | None = None,
    categories: list[str] | None = None,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters to the dataset."""
    filtered = df.copy()

    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[
            (filtered["Order_Date"] >= start) & (filtered["Order_Date"] <= end)
        ]

    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]

    if categories:
        filtered = filtered[filtered["Product_Category"].isin(categories)]

    if channels:
        filtered = filtered[filtered["Sales_Channel"].isin(channels)]

    return filtered
