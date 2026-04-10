"""
Auto Spare Parts Shop Sales Intelligence
Course: Data Science / 21CSS303T
main.py — Entry point: runs the full pipeline
"""

import os
from data_loader import load_raw_data, validate_dataset, print_dataset_info
from preprocessing import preprocess
from eda import run_eda

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if __name__ == "__main__":
    print_dataset_info()

    # Step 1: Load
    df_raw = load_raw_data("data/auto_parts_raw.csv")
    df_raw.to_csv("data/auto_parts_raw_backup.csv", index=False)

    # Step 2: Validate
    assert validate_dataset(df_raw), "Dataset failed validation. Check requirements."

    # Step 3: Preprocess
    df_clean = preprocess(df_raw)
    df_clean.to_csv("data/auto_parts_cleaned.csv", index=False)
    print("[✓] Cleaned dataset saved: data/auto_parts_cleaned.csv")

    # Step 4: EDA + DSI
    dsi_df = run_eda(df_clean)
    dsi_df.to_csv("outputs/dsi_scores.csv", index=False)
    print("[✓] DSI scores saved: outputs/dsi_scores.csv")

    print("\n All steps complete. Check outputs/ folder for charts.")
