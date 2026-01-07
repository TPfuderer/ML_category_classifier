import pandas as pd
from pathlib import Path
import shutil

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FLYERS_DIR = BASE_DIR / "data" / "raw_flyers"
PROCESSED_DIR = BASE_DIR / "data" / "raw_flyers_processed"
OUTPUT_DIR = BASE_DIR / "data" / "raw extracts"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Processing
# ============================================================

csv_files = list(RAW_FLYERS_DIR.glob("*.csv"))

if not csv_files:
    print("ℹ️ No new flyer CSVs found.")
    exit(0)

for csv_path in csv_files:
    print(f"▶ Processing: {csv_path.name}")

    df = pd.read_csv(csv_path)

    # --- Required columns check ---
    if not {"Produkt", "Marke"}.issubset(df.columns):
        print(f"⚠️ Skipped (missing columns): {csv_path.name}")
        continue

    # --- Clean brand OCR artifacts ---
    mask = df["Marke"].astype(str).str.contains(r"\d|\*", regex=True)
    df.loc[mask, "Marke"] = ""

    # --- Keep only ML-relevant columns ---
    df_out = df[["Produkt", "Marke"]].fillna("")

    # --- Clean product names ---
    df_out["Produkt"] = df_out["Produkt"].astype(str).str.strip()
    df_out = df_out[df_out["Produkt"] != ""]

    # --- Output filename ---
    out_name = f"{csv_path.stem}_csv_ml_train.csv"
    out_path = OUTPUT_DIR / out_name

    df_out.to_csv(out_path, index=False)
    print(f"✅ Saved ML-ready CSV: {out_name} ({len(df_out)} rows)")

    # --- Move processed file ---
    shutil.move(
        csv_path,
        PROCESSED_DIR / csv_path.name
    )

    print(f"📦 Moved original to raw_flyers_processed/\n")

print("🎯 All available flyer CSVs processed.")
