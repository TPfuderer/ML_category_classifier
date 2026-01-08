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
    """
    IMPORTANT:
    This function returns the model output EXACTLY as produced by the classifier.
    No post-processing, no thresholds, no UI logic.
    """
    model, label_cols = load_artifacts()

    df = df.copy()
    df["text"] = (
        df["Produkt"].fillna("").astype(str).str.strip()
        + " "
        + df["Marke"].fillna("").astype(str).str.strip()
    ).str.strip()

    preds = model.predict(df["text"])

    out = pd.DataFrame(preds, columns=label_cols)
    out = pd.concat([df[["Produkt", "Marke"]], out], axis=1)

    return out

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
    "Buttermilch | Müllermilch\n"
    "Butter\n"
    "Frischkäse | Arla Buko\n"
    "Protein Bar | IronMaxx\n"
    "Cola Zero | Pepsi\n"
)

raw_text = st.text_area(
    "Products (max 5 lines)",
    value=default_text,
    height=160,
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

    st.subheader("Classification Results (Raw Model Output)")
    st.dataframe(out, use_container_width=True)

    with st.expander("Debug"):
        st.write("Raw prediction table exactly as returned by the model.")
        st.dataframe(out, use_container_width=True)
