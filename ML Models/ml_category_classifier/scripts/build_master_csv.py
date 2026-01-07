import pandas as pd
from pathlib import Path

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_EXTRACTS_DIR = BASE_DIR / "data" / "raw extracts"
MASTER_DIR = BASE_DIR / "data" / "master"
MASTER_PATH = MASTER_DIR / "master_products.csv"

MASTER_DIR.mkdir(parents=True, exist_ok=True)

def clean_marke(value) -> str:
    if value is None:
        return ""

    # echtes NaN (float)
    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "nan", "none", "null", "keine marke"}:
            return ""
        return value.strip()

    return ""

# ============================================================
# Load all ML-ready CSVs
# ============================================================

csv_files = list(RAW_EXTRACTS_DIR.glob("*.csv"))

if not csv_files:
    print("ℹ️ No ML-ready CSVs found in raw extracts/. Master CSV not created.")
    exit(0)

dfs = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Sanity check
    if not {"Produkt", "Marke"}.issubset(df.columns):
        print(f"⚠️ Skipped (missing columns): {csv_file.name}")
        continue

    df = df[["Produkt", "Marke"]].copy()
    dfs.append(df)

if not dfs:
    print("ℹ️ No valid CSVs to build master list. Exiting.")
    exit(0)

# ============================================================
# Build master list
# ============================================================

master = pd.concat(dfs, ignore_index=True)

# Clean
master["Produkt"] = master["Produkt"].astype(str).str.strip()
master["Marke"] = master["Marke"].apply(clean_marke)
master = master[master["Produkt"] != ""]

# Deduplicate
master = master.drop_duplicates(subset=["Produkt", "Marke"])

# Shuffle (reproducible)
master = master.sample(frac=1, random_state=42).reset_index(drop=True)

master.to_csv(MASTER_PATH, index=False)

print(f"✅ Master CSV created: {MASTER_PATH}")
print(f"→ {len(master)} unique products")
