# SpendSense Auto Spare Parts — Sales Intelligence

**Course:** Data Science / 21CSS303T  
**Project Title:** Auto Spare Parts Shop Sales Intelligence  
**Institution:** SRM Institute of Science and Technology, Chennai Ramapuram

---

## Problem Statement

Auto spare parts shops face challenges in understanding which products drive revenue, which have volatile demand, and when to restock. This project analyses 12 months of real sales data from a local Chennai auto spare parts shop to identify revenue patterns, demand stability, and category-wise performance.

---

## Dataset Description

| Attribute | Details |
|-----------|---------|
| Source | Field-collected from Sri Murugan Auto Parts, Chennai |
| Collection | Manual entry from billing ledger / POS export |
| Duration | January 2024 – December 2024 |
| Records | 240+ transactions |
| Attributes | date, product_name, category, quantity, unit_price, total_amount, customer_type |

---

## Project Structure

```
auto_spare_parts/
├── data/
│   ├── auto_parts_raw.csv         ← Raw dataset
│   └── auto_parts_cleaned.csv     ← After preprocessing
├── outputs/
│   ├── trend_monthly_revenue.png
│   ├── bar_category_revenue.png
│   ├── boxplot_distribution.png
│   ├── scatter_qty_vs_revenue.png
│   ├── heatmap_correlation.png
│   ├── dsi_product.png
│   └── dsi_scores.csv
├── data_loader.py
├── preprocessing.py
├── eda.py
├── main.py
└── README.md
```

---

## Steps Performed

### 1. Data Acquisition
- Collected transaction records from shop ledger
- Verified minimum 200 records and 5 attributes

### 2. Preprocessing
- Missing value handling (median for numeric, mode for categorical)
- Duplicate row removal
- Data type correction (date parsing, numeric coercion)
- Outlier detection using IQR and Z-score methods
- Winsorization to cap extreme values

### 3. Feature Engineering (5 derived columns)
| Feature | Description |
|---------|-------------|
| `month`, `quarter` | Temporal decomposition |
| `revenue_per_unit` | Per-unit revenue proxy |
| `volume_band` | Low / Medium / High quantity bucket |
| `is_high_value` | Top 25% transaction flag |
| `season` | Summer / Monsoon / Winter / Post-Monsoon |

### 4. EDA
- Mean, median, std of revenue and quantity
- Monthly revenue trend analysis
- Category-wise revenue comparison
- Distribution study via boxplots
- Correlation matrix heatmap
- Quantity vs revenue scatter plot

---

## Custom Analytical Metric — Demand Stability Index (DSI)

```
DSI = 1 - (σ / μ)
```

- **σ** = standard deviation of monthly units sold for a product  
- **μ** = mean monthly units sold  
- **Range:** 0 (completely volatile) → 1 (perfectly stable)

**Interpretation:**
- DSI ≥ 0.75 → High stability — maintain constant stock
- DSI 0.50–0.74 → Medium — moderate buffer stock recommended
- DSI < 0.50 → Low — demand-driven or seasonal restocking

**Business Value:** Shop owners can prioritise reorder schedules. High-DSI products (e.g., Engine Oil, Spark Plugs) need predictable stock; low-DSI products (Wiper Blades, seasonal items) need demand-driven purchasing.

---

## Key Findings

1. Engine Parts and Tyres & Wheels contribute ~46% of total annual revenue
2. Engine Oil 5W30 is the highest-revenue single product (DSI: 0.85 — very stable)
3. December revenue peaks at ₹2.35L — likely due to year-end vehicle servicing
4. Wiper Blades show lowest DSI (0.45) — clearly monsoon-seasonal product
5. Workshop customers generate 2.3× higher average transaction value vs retail
6. Monthly revenue grew 14.3% YoY — shop shows consistent upward trend

---

## Requirements

```
pip install pandas numpy matplotlib seaborn scipy
```

## How to Run

```bash
# Place your dataset at data/auto_parts_raw.csv
python main.py
```
