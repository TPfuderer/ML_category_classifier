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

# --------------------
# Load label columns globally
# --------------------
LABEL_COLS = joblib.load(LABEL_COLS_PATH)

SUB_COLS = [c for c in LABEL_COLS if c.startswith("sub_")]
TAG_COLS = [c for c in LABEL_COLS if c.startswith("tag_")]
DIET_COLS = [c for c in LABEL_COLS if c.startswith("diet_")]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_cols = joblib.load(LABEL_COLS_PATH)

    cat_cols = [c for c in label_cols if c.startswith("cat_")]
    sub_cols = [c for c in label_cols if c.startswith(("sub_", "tag_", "diet_"))]
    return model, label_cols, cat_cols, sub_cols

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

def prettify_label(label: str) -> str:
    s = label
    for prefix in ("cat_", "sub_", "tag_", "diet_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:]

def collect_active_from_prefix(row: pd.Series, cols: list) -> list:
    return [prettify_label(c) for c in cols if int(row.get(c, 0)) == 1]


def run_prediction(df: pd.DataFrame) -> pd.DataFrame:
    model, label_cols, cat_cols, _ = load_artifacts()

    df = df.copy()
    df["text"] = (
        df["Produkt"].fillna("").astype(str).str.strip()
        + " "
        + df["Marke"].fillna("").astype(str).str.strip()
    ).str.strip()

    # ---------- Predict probabilities ----------
    probas = model.predict_proba(df["text"])

    # !!! WICHTIG: Index explizit setzen !!!
    proba_df = pd.DataFrame(probas, columns=label_cols, index=df.index)

    # ---------- Main category (Top-1) ----------
    df["main_category"] = proba_df[cat_cols].idxmax(axis=1)
    df["main_confidence"] = proba_df[cat_cols].max(axis=1)

    # ---------- Subcategories & Tags (produkt-spezifisch) ----------
    K_SUB = 3
    K_TAG = 2

    def top_k_from_row(row, cols, k):
        return ", ".join(
            prettify_label(c)
            for c in row[cols].nlargest(k).index
        )

    df["Subcategory"] = proba_df.apply(
        lambda row: top_k_from_row(row, SUB_COLS, K_SUB),
        axis=1
    )

    df["Tag"] = proba_df.apply(
        lambda row: top_k_from_row(row, TAG_COLS, K_TAG),
        axis=1
    )

    return df.drop(columns=["text"])





# --------------------
# UI – WITH PREFIX GROUPING
# --------------------
st.set_page_config(page_title="Product Category Classifier (Demo)", layout="wide")

st.title("🧠 Product Category Classifier – Portfolio Demo")
st.caption(
    "Enter up to 5 products. The model predicts a main category (Top-1) "
    "and displays all additional labels grouped by type."
)

with st.expander("ℹ️ How the model was built & trained"):
    st.write("""
    **Overview**

    This application demonstrates a multi-label product classification model
    trained on real-world supermarket data.

    **Data source**
    - Product data was extracted from weekly supermarket flyers and retailer web-scarping (e.g. Rewe, Kaufland, Aldi).
    - This resulted in thousands of raw product entries reflecting real retail product environment.

    **Labeling process**
    - Initial labels (main category, subcategories, tags, dietary labels) were created using
      a semi-automated pipeline.
    - ChatGPT API was used to assist for labeling. 
      (e.g. assigning categories like *dairy*, *snacks*, *high-protein*, *vegan*).
    - All labels were used for supervised training.

    **Model architecture**
    - Text input is built from: `Product name + Brand`.
    - Features are generated using TF-IDF vectorization + logistic regression. 
    - A multi-label classifier was trained to predict:
        • one **main category** (Top-1 selection)
        • multiple **subcategories**
        • multiple **tags** (e.g. protein-related, processing level)
        • optional **diet labels** (e.g. vegan, lactose-free)

    **Prediction logic**
    - Main category: selected as the class with the highest predicted probability.
    - Subcategories and tags: predicted via the model's multi-label output.
    - No hard thresholds are used in the UI — only labels explicitly predicted as active are shown.

    **Purpose**
    - This project focuses on building a realistic, production-oriented NLP pipeline:
        • noisy real-world data
        • scalable labeling
        • explainable predictions
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

col_a, col_b = st.columns([1, 2])

with col_a:
    run_btn = st.button("Classify", type="primary", use_container_width=True)

with col_b:
    st.write("")

if run_btn:
    df_in = parse_lines_to_df(raw_text, max_items=5)

    if df_in.empty:
        st.warning("Please enter at least one product.")
        st.stop()

    if not MODEL_PATH.exists() or not LABEL_COLS_PATH.exists():
        st.error("Model artifacts not found.")
        st.stop()

    # Run prediction
    out = run_prediction(df_in)

    model, label_cols, cat_cols, sub_cols = load_artifacts()

# ---------- RESULTS TABLE ADAPTED ----------
    st.subheader("Classification Results")

    results = []

    for idx, row in out.iterrows():

        active_subs = collect_active_from_prefix(row, SUB_COLS)
        active_tags = collect_active_from_prefix(row, TAG_COLS)

        name = f"{row['Produkt']}"
        if row["Marke"]:
            name += f" | {row['Marke']}"

        results.append({
            "Name": name,
            "Main Category": prettify_label(row["main_category"]),
            "Subcategory": ", ".join(active_subs) if active_subs else "-",
            "Tag": ", ".join(active_tags) if active_tags else "-",
        })

    display_df = pd.DataFrame(results)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---------- DEBUG + OVERVIEW AS EXPANDERS ----------
    with st.expander("Debug"):
        st.write("Model output including all predicted label columns.")

        with st.expander("Available Labels Overview"):
            st.markdown("### All Subcategories")
            st.write(", ".join(prettify_label(c) for c in SUB_COLS))

            st.markdown("### All Tags")
            st.write(", ".join(prettify_label(c) for c in TAG_COLS))

            st.markdown("### All Diet Labels")
            st.write(", ".join(prettify_label(c) for c in DIET_COLS))

        st.markdown("### Raw Prediction Table")
        st.dataframe(out, use_container_width=True)




