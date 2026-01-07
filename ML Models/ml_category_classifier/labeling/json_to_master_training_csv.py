import json
import pandas as pd
from pathlib import Path

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "labeling" / "To Check"
ARCHIVE_DIR = BASE_DIR / "labeling" / "archived"

TRAINING_DIR = BASE_DIR / "data" / "training"
MASTER_CSV = TRAINING_DIR / "master_training.csv"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Pick newest batch JSON
    # --------------------------------------------------------
    batch_files = sorted(
        INPUT_DIR.glob("batch_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not batch_files:
        print("ℹ️ No batch JSONs found. Nothing to do.")
        return

    INPUT_JSON = batch_files[0]
    print(f"📥 Using batch: {INPUT_JSON.name}")

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for entry in data:
        raw_marke = entry.get("Marke", "")
        if pd.isna(raw_marke):
            raw_marke = ""

        row = {
            "Produkt": entry.get("Produkt", "").strip(),
            "Marke": raw_marke
        }

        labels = entry.get("labels", {})
        row.update(labels)

        rows.append(row)

    new_df = pd.DataFrame(rows)

    # stabile Spaltenreihenfolge
    label_cols = sorted(
        [c for c in new_df.columns if c.startswith(("cat_", "sub_", "tag_", "diet_"))]
    )
    new_df = new_df[["Produkt", "Marke"] + label_cols]

    # --------------------------------------------------------
    # Append to master_training.csv
    # --------------------------------------------------------
    if MASTER_CSV.exists():
        master_df = pd.read_csv(MASTER_CSV)

        for col in master_df.columns:
            if col not in new_df:
                new_df[col] = 0
        for col in new_df.columns:
            if col not in master_df:
                master_df[col] = 0

        new_df = new_df[master_df.columns]
        combined = pd.concat([master_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(MASTER_CSV, index=False, encoding="utf-8")

    # --------------------------------------------------------
    # Archive processed batch
    # --------------------------------------------------------
    archived_path = ARCHIVE_DIR / INPUT_JSON.name
    INPUT_JSON.rename(archived_path)

    print("✅ master_training.csv updated")
    print(f"📄 Path: {MASTER_CSV}")
    print(f"➕ Added rows: {len(new_df)}")
    print(f"📊 Total rows: {len(combined)}")
    print(f"📦 Archived batch → {archived_path}")

if __name__ == "__main__":
    main()
