import pandas as pd
import joblib
from pathlib import Path

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "category_classifier.pkl"
LABEL_COLS_PATH = BASE_DIR / "artifacts" / "label_columns.pkl"

INPUT_CSV = BASE_DIR / "data" / "raw_products.csv"
OUTPUT_CSV = BASE_DIR / "data" / "raw_products_labeled.csv"

# --------------------
# Load model + labels
# --------------------
model = joblib.load(MODEL_PATH)
LABEL_COLS = joblib.load(LABEL_COLS_PATH)
CAT_COLS = [c for c in LABEL_COLS if c.startswith("cat_")]
SUB_COLS = [c for c in LABEL_COLS if c.startswith(("sub_", "tag_", "diet_"))]

# --------------------
# Load input data
# --------------------
df = pd.read_csv(INPUT_CSV)

# --------------------
# Build text feature (EXACTLY like training)
# --------------------
df["text"] = (
    df["Produkt"].fillna("").astype(str).str.strip()
    + " "
    + df["Marke"].fillna("").astype(str).str.strip()
).str.strip()

# --------------------
# Predict (DEBUG + MIXED LOGIC)
# --------------------
probas = model.predict_proba(df["text"])
proba_df = pd.DataFrame(probas, columns=LABEL_COLS)

# --- Debug ---
print("\n🔍 Raw probabilities (head):")
print(proba_df.head())

print("\n🔍 Max probability per row (cats):")
print(proba_df[CAT_COLS].max(axis=1).head())

print("\n🔍 Top main category per row:")
print(proba_df[CAT_COLS].idxmax(axis=1).head())

# --------------------
# 1️⃣ MAIN CATEGORIES → TOP-1
# --------------------
pred_df = pd.DataFrame(0, index=df.index, columns=LABEL_COLS)

top_cat_idx = proba_df[CAT_COLS].values.argmax(axis=1)
for i, j in enumerate(top_cat_idx):
    pred_df.at[i, CAT_COLS[j]] = 1

# --------------------
# 2️⃣ SUB / TAG / DIET → THRESHOLD
# --------------------
SUB_THRESHOLD = 0.15  # später feinjustieren

if SUB_COLS:
    sub_preds = (proba_df[SUB_COLS] >= SUB_THRESHOLD).astype(int)
    pred_df[SUB_COLS] = sub_preds

# --------------------
# Optional: Confidence der Hauptkategorie
# --------------------
pred_df["main_confidence"] = proba_df[CAT_COLS].max(axis=1)

# --------------------
# Merge + Save
# --------------------
out = pd.concat([df.drop(columns=["text"]), pred_df], axis=1)
out.to_csv(OUTPUT_CSV, index=False)

print("✅ Classification finished")
print("🏷️ Labels used:", LABEL_COLS)
print(f"📄 Output saved to: {OUTPUT_CSV}")

