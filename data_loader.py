"""
Auto Spare Parts Shop Sales Intelligence
Course: Data Science / 21CSS303T

data_loader.py — Load and validate the dataset
NOTE: Replace CSV path with your actual collected dataset.
      Minimum: 200 records, 5 meaningful attributes, real/verifiable data.
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# Dataset Info (update before submission)
# ──────────────────────────────────────────────
DATA_SOURCE = "Field-collected from Sri Murugan Auto Parts, Chennai (Jan–Dec 2024)"
COLLECTION_METHOD = "Manual entry from shop ledger / billing software export"
DATASET_LINK = "https://your-github-repo/raw/dataset/auto_parts_raw.csv"
DURATION = "January 2024 – December 2024 (12 months)"
LIMITATIONS = [
    "Data from single shop — may not represent broader market",
    "Seasonal festivals (Pongal, Diwali) may distort monthly patterns",
    "Returns/refunds not separately tracked in source ledger",
]

# ──────────────────────────────────────────────
# Columns expected in raw dataset
# ──────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "date",           # Transaction date (DD-MM-YYYY or YYYY-MM-DD)
    "product_name",   # Name of spare part
    "category",       # Product category (Engine, Electrical, etc.)
    "quantity",       # Units sold
    "unit_price",     # Price per unit (INR)
    "total_amount",   # quantity × unit_price
    "customer_type",  # Retail / Workshop / Fleet
]


def load_raw_data(filepath: str = "data/auto_parts_raw.csv") -> pd.DataFrame:
    """Load raw dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place your collected dataset CSV there."
        )
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded {len(df)} records from {filepath}")
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Check minimum dataset requirements per project guidelines."""
    errors = []
    if len(df) < 200:
        errors.append(f"Only {len(df)} records — minimum 200 required.")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if errors:
        for e in errors:
            print(f"[✗] {e}")
        return False
    print(f"[✓] Dataset validation passed — {len(df)} records, {len(df.columns)} columns")
    return True


def print_dataset_info():
    """Print data source metadata for report documentation."""
    print("=" * 55)
    print("DATASET INFORMATION")
    print("=" * 55)
    print(f"Source      : {DATA_SOURCE}")
    print(f"Method      : {COLLECTION_METHOD}")
    print(f"Duration    : {DURATION}")
    print(f"Link        : {DATASET_LINK}")
    print("Limitations :")
    for lim in LIMITATIONS:
        print(f"  - {lim}")
    print("=" * 55)


if __name__ == "__main__":
    print_dataset_info()
