"""
Auto Spare Parts Shop Sales Intelligence
Course: Data Science / 21CSS303T

preprocessing.py — Data cleaning, outlier detection, feature engineering
"""

import pandas as pd
import numpy as np
from scipy import stats


# ──────────────────────────────────────────────
# 1. Missing Value Handling
# ──────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with documented strategy."""
    missing_before = df.isnull().sum()
    print("\n[Missing Values — Before]")
    print(missing_before[missing_before > 0])

    # Numeric columns: fill with median (robust to outliers)
    for col in ["quantity", "unit_price", "total_amount"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill with mode
    for col in ["category", "customer_type"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = df.isnull().sum().sum()
    print(f"[✓] Missing values after handling: {missing_after}")
    return df


# ──────────────────────────────────────────────
# 2. Duplicate Removal
# ──────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[✓] Duplicates removed: {before - after} rows (kept {after})")
    return df


# ──────────────────────────────────────────────
# 3. Data Type Correction
# ──────────────────────────────────────────────
def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types for all columns."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in ["quantity", "unit_price", "total_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["product_name", "category", "customer_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    print("[✓] Data types corrected")
    return df


# ──────────────────────────────────────────────
# 4. Outlier Detection (IQR Method)
# ──────────────────────────────────────────────
def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Return boolean mask of outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = (df[column] < lower) | (df[column] > upper)
    print(f"[IQR] {column}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f} → {outliers.sum()} outliers found")
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outlier_idx = df[column].dropna().index[z_scores > threshold]
    outliers = df.index.isin(outlier_idx)
    print(f"[Z-score] {column}: {outliers.sum()} outliers (|z| > {threshold})")
    return outliers


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using IQR bounds (Winsorization)."""
    for col in ["total_amount", "quantity"]:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    print("[✓] Outliers capped using Winsorization (IQR bounds)")
    return df


# ──────────────────────────────────────────────
# 5. Feature Engineering (minimum 3 derived columns)
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for analysis."""

    # Feature 1: Month and Year
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["day_of_week"] = df["date"].dt.day_name()
        print("[✓] Feature 1: Temporal features (month, quarter, day_of_week)")

    # Feature 2: Revenue per Unit (unit margin proxy)
    if "total_amount" in df.columns and "quantity" in df.columns:
        df["revenue_per_unit"] = (df["total_amount"] / df["quantity"]).round(2)
        print("[✓] Feature 2: revenue_per_unit (unit value proxy)")

    # Feature 3: Volume Band (Low / Medium / High quantity)
    if "quantity" in df.columns:
        df["volume_band"] = pd.cut(
            df["quantity"],
            bins=[0, 2, 10, float("inf")],
            labels=["Low (<= 2)", "Medium (3-10)", "High (> 10)"],
        )
        print("[✓] Feature 3: volume_band (Low / Medium / High)")

    # Feature 4: High-value transaction flag
    if "total_amount" in df.columns:
        threshold = df["total_amount"].quantile(0.75)
        df["is_high_value"] = (df["total_amount"] >= threshold).astype(int)
        print(f"[✓] Feature 4: is_high_value (top 25% transactions, threshold ₹{threshold:.0f})")

    # Feature 5: Season
    if "month" in df.columns:
        def get_season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Summer"
            elif m in [6, 7, 8, 9]: return "Monsoon"
            else: return "Post-Monsoon"
        df["season"] = df["month"].apply(get_season)
        print("[✓] Feature 5: season (Winter / Summer / Monsoon / Post-Monsoon)")

    return df


# ──────────────────────────────────────────────
# 6. Basic Statistical Summary
# ──────────────────────────────────────────────
def statistical_summary(df: pd.DataFrame):
    """Print descriptive statistics for numeric columns."""
    print("\n" + "=" * 55)
    print("STATISTICAL SUMMARY")
    print("=" * 55)
    numeric_cols = ["quantity", "unit_price", "total_amount", "revenue_per_unit"]
    available = [c for c in numeric_cols if c in df.columns]
    print(df[available].describe().round(2))
    print("=" * 55)


# ──────────────────────────────────────────────
# Main preprocessing pipeline
# ──────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing pipeline and return cleaned dataframe."""
    print("\n─── Starting Preprocessing Pipeline ───")
    df = fix_data_types(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = engineer_features(df)
    statistical_summary(df)
    print("─── Preprocessing Complete ───\n")
    return df


if __name__ == "__main__":
    # Quick test with dummy data
    sample = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="15D"),
        "product_name": ["Engine Oil"] * 5 + ["Brake Pad"] * 5,
        "category": ["Consumables"] * 5 + ["Engine Parts"] * 5,
        "quantity": [5, 3, 2, 10, 4, 2, 1, 3, 2, 6],
        "unit_price": [450, 450, 450, 450, 450, 800, 800, 800, 800, 800],
        "total_amount": [2250, 1350, 900, 4500, 1800, 1600, 800, 2400, 1600, 4800],
        "customer_type": ["Retail", "Workshop"] * 5,
    })
    cleaned = preprocess(sample)
    print(cleaned.head())
