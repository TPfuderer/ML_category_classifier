# ml_category_classifier/app/portfolio_app.py
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

# --------------------
# Paths (project layout)
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1] / "ML Models" / "ml_category_classifier"

MODEL_PATH = BASE_DIR / "artifacts" / "category_classifier.pkl"
LABEL_COLS_PATH = BASE_DIR / "artifacts" / "label_columns.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_cols = joblib.load(LABEL_COLS_PATH)
    return model, label_cols

def parse_lines_to_df(raw_text: str, max_items: int = 5) -> pd.DataFrame:
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    lines = lines[:max_items]

    rows = []
    for ln in lines:
        if "|" in ln:
            left, right = ln.split("|", 1)
            produkt = left.strip()
            marke = right.strip()
        else:
            produkt = ln.strip()
            marke = ""

        rows.append({"Produkt": produkt, "Marke": marke})

    return pd.DataFrame(rows)

def run_prediction(df: pd.DataFrame) -> pd.DataFrame:
    model, label_cols = load_artifacts()

    CAT_COLS = [c for c in label_cols if c.startswith("cat_")]
    SUB_COLS = [c for c in label_cols if c.startswith(("sub_", "tag_", "diet_"))]

    df = df.copy()
    df["text"] = (
        df["Produkt"].fillna("").astype(str).str.strip()
        + " "
        + df["Marke"].fillna("").astype(str).str.strip()
    ).str.strip()

    # --- Predict probabilities ---
    probas = model.predict_proba(df["text"])
    proba_df = pd.DataFrame(probas, columns=label_cols, index=df.index)

    # --- Build prediction table ---
    pred_df = pd.DataFrame(0, index=df.index, columns=label_cols)

    # 1️⃣ MAIN CATEGORY → TOP-1
    top_cat_idx = proba_df[CAT_COLS].values.argmax(axis=1)
    for i, j in enumerate(top_cat_idx):
        pred_df.at[i, CAT_COLS[j]] = 1

    # 2️⃣ SUB / TAG / DIET → THRESHOLD (EXACT MATCH)
    SUB_THRESHOLD = 0.15
    if SUB_COLS:
        pred_df[SUB_COLS] = (proba_df[SUB_COLS] >= SUB_THRESHOLD).astype(int)

    # 3️⃣ Confidence
    pred_df["main_confidence"] = proba_df[CAT_COLS].max(axis=1)

    out = pd.concat([df.drop(columns=["text"]), pred_df], axis=1)
    return out

def build_pretty_table(out: pd.DataFrame, label_cols: list) -> pd.DataFrame:
    CAT_COLS  = [c for c in label_cols if c.startswith("cat_")]
    SUB_COLS  = [c for c in label_cols if c.startswith("sub_")]
    TAG_COLS  = [c for c in label_cols if c.startswith("tag_")]
    DIET_COLS = [c for c in label_cols if c.startswith("diet_")]

    def pretty(c: str) -> str:
        return (
            c.replace("cat_", "")
             .replace("sub_", "")
             .replace("tag_", "")
             .replace("diet_", "")
             .replace("_", " ")
             .title()
        )

    rows = []

    for _, row in out.iterrows():
        name = row["Produkt"]
        if row.get("Marke"):
            name += f" | {row['Marke']}"

        cats = [pretty(c) for c in CAT_COLS  if int(row.get(c, 0)) == 1]
        subs = [pretty(c) for c in SUB_COLS  if int(row.get(c, 0)) == 1]
        tags = [pretty(c) for c in TAG_COLS  if int(row.get(c, 0)) == 1]
        diets = [pretty(c) for c in DIET_COLS if int(row.get(c, 0)) == 1]

        rows.append({
            "Product": name,
            "Categories": ", ".join(cats) if cats else "-",
            "Subcategories": ", ".join(subs) if subs else "-",
            "Tags": ", ".join(tags) if tags else "-",
            "Diet Labels": ", ".join(diets) if diets else "-",
            "Confidence": round(float(row.get("main_confidence", 0)), 3),
        })

    return pd.DataFrame(rows)



# --------------------
# UI
# --------------------
st.set_page_config(page_title="Product Category Classifier (Demo)", layout="wide")

st.title("🧠 Product Category Classifier – Portfolio Demo")
st.caption(
    "Enter up to 5 products. The model predicts product categories and tags. "
    "The table below shows the raw model output without any UI-side manipulation."
)

with st.expander("ℹ️ How the model was built & trained"):
    st.write("""
    **Overview**

    This application demonstrates a multi-label product classification model
    trained on real-world supermarket data.

    **Data source**
    - Product data was extracted from weekly supermarket flyers and retailer web-scraping
      (e.g. Rewe, Kaufland, Aldi).
    - This resulted in thousands of raw product entries reflecting a real retail environment.

    **Labeling process**
    - Labels (main category, subcategories, tags, dietary labels) were created using a
      semi-automated pipeline.
    - The ChatGPT API was used to assist with scalable and consistent labeling.
    - All labels were stored explicitly and used for supervised training.

    **Model architecture**
    - Text input: `Product name + Brand`
    - Feature extraction: TF-IDF
    - Classifier: Logistic Regression (multi-label setup)

    **Prediction logic**
    - The model outputs a full binary label table.
    - This app intentionally performs no post-processing or thresholding.
    - The UI only visualizes the raw model output.

    **Purpose**
    - Demonstrate a realistic end-to-end NLP pipeline:
        • noisy real-world data
        • scalable labeling
        • transparent model output
        • deployment as an interactive web app
    """)

default_text = (
    "Kulturheidelbeeren\n"
    "Shampoo | Head & Shoulders\n"
    "GQB Strohschwein Frischwurst-Aufschnitt | Schiller\n"
    "Laktosefreie H-Milch | Milsani\n"
    "Protein-Riegel | Wellmix Sport\n"
)

raw_text = st.text_area(
    "Products (max 5 lines)\nNo \"|\" divider required",
    value=default_text,
    height=200,
)

if st.button("Classify", type="primary", use_container_width=True):
    df_in = parse_lines_to_df(raw_text)

    if df_in.empty:
        st.warning("Please enter at least one product.")
        st.stop()

    if not MODEL_PATH.exists() or not LABEL_COLS_PATH.exists():
        st.error("Model artifacts not found.")
        st.stop()

    out = run_prediction(df_in)

    st.subheader("Classification Results")

    pretty_df = build_pretty_table(out, load_artifacts()[1])
    st.dataframe(pretty_df, use_container_width=True, hide_index=True)

    st.subheader("Raw Model Output")
    st.dataframe(out, use_container_width=True)
