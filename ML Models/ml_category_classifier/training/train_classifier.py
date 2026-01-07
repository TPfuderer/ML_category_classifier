import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "training" / "master_training.csv"
ARTIFACTS = BASE_DIR / "artifacts"

ARTIFACTS.mkdir(exist_ok=True)

# --------------------
# Load data
# --------------------
df = pd.read_csv(DATA_PATH)

# --------------------
# Build text feature (Produkt + Marke)
# --------------------
df["text"] = (
    df["Produkt"].fillna("").astype(str).str.strip()
    + " "
    + df["Marke"].fillna("").astype(str).str.strip()
).str.strip()

TEXT_COL = "text"

# ============================================================
# LABEL SELECTION
# ============================================================
# TODO (DATA): Aktuell NUR Hauptkategorien trainieren.
# Ab ~500+ gelabelten Produkten:
#  - sub_ hinzufügen
# Ab ~800–1000:
#  - tag_ hinzufügen
# Ab ~1200:
#  - diet_ hinzufügen

#LABEL_COLS = [
    #c for c in df.columns
    #if c.startswith("cat_")
#]

LABEL_COLS = [
    c for c in df.columns
    if c.startswith(("cat_", "sub_", "tag_", "diet_"))
]

X = df[TEXT_COL]
y = df[LABEL_COLS].fillna(0).astype(int)

# --------------------
# Save label schema
# --------------------
joblib.dump(LABEL_COLS, ARTIFACTS / "label_columns.pkl")

print("\n📊 Positive labels per category:")
print(y.sum().sort_values(ascending=False))

# --------------------
# Pipeline (unchanged logic)
# --------------------
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1
        )

    ),
    (
        "clf",
        OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        )
    )
])

# --------------------
# Train
# --------------------
pipeline.fit(X, y)

# --------------------
# Save
# --------------------
joblib.dump(pipeline, ARTIFACTS / "category_classifier.pkl")

print("✅ Model trained and saved")
print(f"📄 Training rows: {len(df)}")
print(f"🏷️ Labels: {LABEL_COLS}")
