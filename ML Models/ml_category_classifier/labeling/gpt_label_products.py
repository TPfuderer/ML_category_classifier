import pandas as pd
import json
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "to_train"
OUTPUT_DIR = BASE_DIR / "labeling" / "To Check"
ARCHIVE_DIR = BASE_DIR / "labeling" / "labeled_batches"

MODEL_NAME = "gpt-5-mini"
SLEEP_SECONDS = 0.2
LIMIT = None   # None = alle Produkte

client = OpenAI()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# GPT Prompt
# ============================================================

SYSTEM_PROMPT = """
You are classifying German supermarket products.

Rules:
- Use ONLY the provided product name and brand.
- Do NOT guess nutrition values.
- Be conservative: if unsure, return 0.
- Use binary values only (0 or 1).
- Choose EXACTLY ONE main category.
- Subcategories and tags are OPTIONAL.
- Return ONLY valid JSON.
- Do NOT add explanations or text outside JSON.
"""

USER_PROMPT_TEMPLATE = """
Classify the following product.

Product name: "{produkt}"
Brand: "{marke}"

MAIN CATEGORIES (exactly one must be 1):
- cat_obst_gemuese
- cat_molkerei_kaese
- cat_fleisch_wurst
- cat_fisch_meeresfruechte
- cat_grundnahrung
- cat_tiefkuehl
- cat_getraenke
- cat_knabbern_naschen
- cat_drogerie
- cat_tiernahrung
- cat_non_food
- cat_sonstiges

SUBCATEGORIES (set to 1 if clearly applicable):
- sub_frisches_obst
- sub_frisches_gemuese
- sub_salate
- sub_pilze
- sub_kraeuter
- sub_exoten
- sub_milchprodukte
- sub_joghurt_skyr_quark
- sub_kaese
- sub_butter_margarine
- sub_sahne_creme
- sub_rind
- sub_schwein
- sub_gefluegel
- sub_wurst
- sub_hackfleisch
- sub_fleischersatz
- sub_reis
- sub_nudeln
- sub_kartoffeln
- sub_huelsenfruechte
- sub_tk_gemuese
- sub_tk_fertiggerichte
- sub_tk_pizza
- sub_tk_eis

TAGS:
- tag_ultra_processed
- tag_whole_food
- tag_high_protein
- tag_proteinreich

DIET TAGS:
- diet_vegetarian
- diet_vegan
- diet_lactose_free
- diet_gluten_free

OUTPUT FORMAT:
Return a single valid JSON object with ALL fields listed above.
Use only 0 or 1.
"""

# ============================================================
# Helpers
# ============================================================

def get_next_batch() -> Path | None:
    batches = sorted(INPUT_DIR.glob("batch_*.csv"))
    return batches[0] if batches else None


def call_gpt(produkt: str, marke: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    produkt=produkt,
                    marke=marke if isinstance(marke, str) else ""
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)

# ============================================================
# Main
# ============================================================

def main():
    batch_path = get_next_batch()

    if batch_path is None:
        print("ℹ️ No batches found in data/to_train/. Nothing to label.")
        return

    print(f"📦 Processing batch: {batch_path.name}")

    df = pd.read_csv(batch_path)
    df_iter = df.head(LIMIT) if LIMIT is not None else df

    all_results = []

    for _, row in tqdm(df_iter.iterrows(), total=len(df_iter)):
        produkt = row.get("Produkt", "")
        marke = row.get("Marke", "")

        try:
            labels = call_gpt(produkt, marke)

            entry = {
                "Produkt": produkt,
                "Marke": marke,
                "labels": labels,
                "label_source": "gpt"
            }

            all_results.append(entry)

        except Exception as e:
            print(f"❌ Error for product: {produkt}")
            print(e)

        time.sleep(SLEEP_SECONDS)

    output_json = OUTPUT_DIR / batch_path.with_suffix(".json").name

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    archived_batch = ARCHIVE_DIR / batch_path.name
    batch_path.rename(archived_batch)

    print("✅ GPT pre-labeling finished")
    print(f"📝 Output saved to: {output_json}")
    print(f"📦 Batch archived to: {archived_batch}")

if __name__ == "__main__":
    main()
