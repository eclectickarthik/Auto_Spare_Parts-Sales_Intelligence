"""
Auto Spare Parts Shop Sales Intelligence
Course: Data Science / 21CSS303T

eda.py — Exploratory Data Analysis + Custom Metric (DSI)
Run after preprocessing.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = ["#185FA5", "#1D9E75", "#D85A30", "#BA7517", "#993C1D", "#533AB7"]


# ──────────────────────────────────────────────
# 1. Central Tendency & Dispersion
# ──────────────────────────────────────────────
def central_tendency(df: pd.DataFrame):
    """Mean, median, std for key numeric columns."""
    print("\n─── Central Tendency Analysis ───")
    cols = ["total_amount", "quantity", "unit_price"]
    for col in cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            print(f"  Mean   : ₹{df[col].mean():,.2f}")
            print(f"  Median : ₹{df[col].median():,.2f}")
            print(f"  Std    : ₹{df[col].std():,.2f}")
            print(f"  Skew   : {df[col].skew():.3f}")


# ──────────────────────────────────────────────
# 2. Monthly Revenue Trend
# ──────────────────────────────────────────────
def plot_trend(df: pd.DataFrame, save: bool = True):
    """Line plot of monthly revenue trend."""
    if "month" not in df.columns or "total_amount" not in df.columns:
        print("[!] Missing month/total_amount columns — skipping trend plot")
        return

    monthly = df.groupby(["year", "month"])["total_amount"].sum().reset_index()
    monthly["period"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
    monthly = monthly.sort_values("period")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["period"], monthly["total_amount"],
            marker="o", linewidth=2, color=COLORS[0], markersize=5)
    ax.fill_between(monthly["period"], monthly["total_amount"],
                    alpha=0.1, color=COLORS[0])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (INR)")
    plt.tight_layout()
    if save:
        plt.savefig("outputs/trend_monthly_revenue.png", dpi=150, bbox_inches="tight")
        print("[✓] Saved: outputs/trend_monthly_revenue.png")
    plt.show()


# ──────────────────────────────────────────────
# 3. Category Revenue Bar Chart
# ──────────────────────────────────────────────
def plot_category_bar(df: pd.DataFrame, save: bool = True):
    """Bar chart of revenue by category."""
    if "category" not in df.columns:
        return
    cat_rev = df.groupby("category")["total_amount"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(cat_rev.index, cat_rev.values, color=COLORS, edgecolor="white")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2000, bar.get_y() + bar.get_height()/2,
                f"₹{width/100000:.2f}L", va="center", fontsize=10)
    ax.set_title("Revenue by Product Category", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Total Revenue (INR)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    plt.tight_layout()
    if save:
        plt.savefig("outputs/bar_category_revenue.png", dpi=150, bbox_inches="tight")
        print("[✓] Saved: outputs/bar_category_revenue.png")
    plt.show()


# ──────────────────────────────────────────────
# 4. Distribution (Boxplot)
# ──────────────────────────────────────────────
def plot_boxplot(df: pd.DataFrame, save: bool = True):
    """Boxplot of transaction amounts by category."""
    if "category" not in df.columns or "total_amount" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="total_amount", by="category", ax=ax,
               boxprops=dict(color=COLORS[0]),
               medianprops=dict(color=COLORS[1], linewidth=2),
               flierprops=dict(marker="o", markersize=4, alpha=0.5))
    ax.set_title("Transaction Amount Distribution by Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Category"); ax.set_ylabel("Amount (INR)")
    plt.suptitle("")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save:
        plt.savefig("outputs/boxplot_distribution.png", dpi=150, bbox_inches="tight")
        print("[✓] Saved: outputs/boxplot_distribution.png")
    plt.show()


# ──────────────────────────────────────────────
# 5. Correlation Heatmap
# ──────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame, save: bool = True):
    """Correlation matrix heatmap."""
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    if save:
        plt.savefig("outputs/heatmap_correlation.png", dpi=150, bbox_inches="tight")
        print("[✓] Saved: outputs/heatmap_correlation.png")
    plt.show()


# ──────────────────────────────────────────────
# 6. Scatter — Revenue vs Quantity
# ──────────────────────────────────────────────
def plot_scatter(df: pd.DataFrame, save: bool = True):
    """Scatter plot of revenue vs quantity sold."""
    if "quantity" not in df.columns or "total_amount" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = df["category"].unique() if "category" in df.columns else ["All"]
    for i, cat in enumerate(cats):
        sub = df[df["category"] == cat] if "category" in df.columns else df
        ax.scatter(sub["quantity"], sub["total_amount"],
                   label=cat, color=COLORS[i % len(COLORS)], alpha=0.6, s=40)
    ax.set_xlabel("Quantity Sold"); ax.set_ylabel("Transaction Amount (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_title("Quantity vs Revenue (by Category)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    if save:
        plt.savefig("outputs/scatter_qty_vs_revenue.png", dpi=150, bbox_inches="tight")
        print("[✓] Saved: outputs/scatter_qty_vs_revenue.png")
    plt.show()


# ──────────────────────────────────────────────
# 7. CUSTOM METRIC — Demand Stability Index (DSI)
# ──────────────────────────────────────────────
def compute_dsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demand Stability Index (DSI) — Custom Analytical Metric
    ─────────────────────────────────────────────────────────
    Formula : DSI = 1 - (σ / μ)
      σ = standard deviation of monthly sales volume for a product
      μ = mean monthly sales volume for that product

    Logic   : DSI is essentially 1 minus the Coefficient of Variation (CV).
              A product sold in nearly equal quantities every month has low σ
              relative to μ, resulting in DSI close to 1 (very stable).
              A product with erratic demand has high σ, giving DSI close to 0.

    Range   : 0 (completely volatile) → 1 (perfectly stable)
              Negative values are clipped to 0 (extreme outliers only).

    Use     : Shop owners can use DSI to prioritise reorder schedules.
              High DSI products need stable stock; low DSI products need
              demand-driven or seasonal restocking strategies.
    """
    if "month" not in df.columns or "product_name" not in df.columns:
        raise ValueError("Need 'month' and 'product_name' columns for DSI calculation.")

    monthly_product = (
        df.groupby(["product_name", "month"])["quantity"]
        .sum()
        .reset_index()
    )

    dsi_results = []
    for product, group in monthly_product.groupby("product_name"):
        mu = group["quantity"].mean()
        sigma = group["quantity"].std(ddof=0)
        if mu == 0:
            dsi = 0.0
        else:
            dsi = max(0.0, round(1 - (sigma / mu), 4))
        dsi_results.append({
            "product_name": product,
            "mean_monthly_units": round(mu, 2),
            "std_monthly_units": round(sigma, 2),
            "DSI": dsi,
            "demand_stability": (
                "High" if dsi >= 0.75 else
                "Medium" if dsi >= 0.50 else
                "Low"
            )
        })

    dsi_df = pd.DataFrame(dsi_results).sort_values("DSI", ascending=False)
    print("\n─── Demand Stability Index (DSI) ───")
    print(dsi_df.to_string(index=False))
    return dsi_df


def plot_dsi(dsi_df: pd.DataFrame, save: bool = True):
    """Horizontal bar chart of DSI scores per product."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(dsi_df) * 0.5)))
    color_map = {"High": "#185FA5", "Medium": "#BA7517", "Low": "#A32D2D"}
    colors = [color_map[d] for d in dsi_df["demand_stability"]]

    ax.barh(dsi_df["product_name"], dsi_df["DSI"], color=colors, edgecolor="white")
    ax.axvline(0.75, color="#185FA5", linestyle="--", linewidth=1, label="High threshold (0.75)")
    ax.axvline(0.50, color="#BA7517", linestyle="--", linewidth=1, label="Medium threshold (0.50)")
    for i, (val, name) in enumerate(zip(dsi_df["DSI"], dsi_df["product_name"])):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("DSI Score (0 = volatile, 1 = stable)")
    ax.set_title("Demand Stability Index (DSI) by Product", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig("outputs/dsi_product.png", dpi=150, bbox_inches="tight")
        print("[✓] Saved: outputs/dsi_product.png")
    plt.show()


# ──────────────────────────────────────────────
# Run all EDA
# ──────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    """Run full EDA pipeline on cleaned dataframe."""
    import os
    os.makedirs("outputs", exist_ok=True)

    central_tendency(df)
    plot_trend(df)
    plot_category_bar(df)
    plot_boxplot(df)
    plot_heatmap(df)
    plot_scatter(df)

    dsi_df = compute_dsi(df)
    plot_dsi(dsi_df)

    print("\n[✓] EDA complete. All charts saved to outputs/")
    return dsi_df


if __name__ == "__main__":
    print("Run this module via main.py after loading and preprocessing your dataset.")
    print("Example:\n  from preprocessing import preprocess\n  from eda import run_eda")
    print("  df = preprocess(raw_df)")
    print("  run_eda(df)")
