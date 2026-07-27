"""Generate synthetic corporate sales dataset."""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
CATEGORIES = {
    "Electronics": [
        "Laptop Pro 15",
        "Wireless Headphones",
        "Smart Watch",
        "Tablet Air",
        "4K Monitor",
    ],
    "Furniture": [
        "Executive Desk",
        "Ergonomic Chair",
        "Conference Table",
        "Bookshelf Unit",
        "Standing Desk",
    ],
    "Office Supplies": [
        "Printer Paper Pack",
        "Ink Cartridge Set",
        "Stapler Deluxe",
        "Whiteboard Kit",
        "Desk Organizer",
    ],
    "Clothing": [
        "Business Shirt",
        "Corporate Polo",
        "Safety Vest",
        "Winter Jacket",
        "Work Boots",
    ],
    "Food & Beverage": [
        "Coffee Beans 5lb",
        "Snack Variety Pack",
        "Bottled Water Case",
        "Energy Drink Pack",
        "Organic Tea Set",
    ],
}
SALES_CHANNELS = ["Online", "Retail Store", "Direct Sales", "Partner", "Phone"]
FIRST_NAMES = [
    "James", "Maria", "Robert", "Sarah", "Michael", "Emily", "David", "Lisa",
    "John", "Anna", "William", "Jennifer", "Richard", "Jessica", "Thomas",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas",
]


def generate_sales_data(n_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sales records with realistic business patterns."""
    random.seed(seed)
    np.random.seed(seed)

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    n_customers = 500
    customers = [
        {
            "Customer_ID": f"CUST-{i:04d}",
            "Customer_Name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        }
        for i in range(1, n_customers + 1)
    ]

    records = []
    for idx in range(1, n_records + 1):
        customer = random.choice(customers)
        category = random.choice(list(CATEGORIES.keys()))
        product = random.choice(CATEGORIES[category])
        region = random.choice(REGIONS)
        channel = random.choice(SALES_CHANNELS)

        order_date = start_date + timedelta(days=random.randint(0, date_range))
        quantity = random.randint(1, 20)
        unit_price = round(random.uniform(10, 500), 2)
        discount = round(random.uniform(0, 0.25), 2)
        revenue = round(quantity * unit_price * (1 - discount), 2)
        cost_ratio = random.uniform(0.55, 0.85)
        profit = round(revenue * (1 - cost_ratio), 2)

        records.append(
            {
                "Order_ID": f"ORD-{idx:06d}",
                "Order_Date": order_date.strftime("%Y-%m-%d"),
                "Customer_ID": customer["Customer_ID"],
                "Customer_Name": customer["Customer_Name"],
                "Product_Category": category,
                "Product_Name": product,
                "Region": region,
                "Quantity": quantity,
                "Unit_Price": unit_price,
                "Discount": discount,
                "Revenue": revenue,
                "Profit": profit,
                "Sales_Channel": channel,
            }
        )

    df = pd.DataFrame(records)

    # Inject controlled missing values for EDA realism (~1%)
    missing_indices = random.sample(range(len(df)), k=int(len(df) * 0.01))
    for i in missing_indices[: len(missing_indices) // 2]:
        df.at[i, "Discount"] = np.nan
    for i in missing_indices[len(missing_indices) // 2 :]:
        df.at[i, "Profit"] = np.nan

    return df


def save_dataset(output_path: Path, n_records: int = 10000) -> None:
    """Generate and persist sales dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sales_data(n_records=n_records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} records to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    save_dataset(project_root / "data" / "sales_data.csv")
