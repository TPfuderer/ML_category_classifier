import subprocess
from pathlib import Path
import sys
import pandas as pd


# ============================================================
# Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FLYERS_DIR = BASE_DIR / "data" / "raw_flyers"
SCRIPT_DIR = BASE_DIR / "scripts"

EXTRACT_SCRIPT = SCRIPT_DIR / "extract_relevant_columns.py"
BUILD_MASTER_SCRIPT = SCRIPT_DIR / "build_master_csv.py"

# ============================================================
# Helpers
# ============================================================

def has_new_flyers() -> bool:
    return any(RAW_FLYERS_DIR.glob("*.csv"))

def run_script(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\n▶ Running: {script_path.name}")
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True
    )

# ============================================================
# Main
# ============================================================

def main():
    print("🚀 Pipeline start: run_until_master\n")

    if not has_new_flyers():
        print("ℹ️ No new CSVs in raw_flyers/. Nothing to do.")
        return

    print("📂 New flyer CSVs detected.\n")

    run_script(EXTRACT_SCRIPT)
    run_script(BUILD_MASTER_SCRIPT)

    print("\n✅ Pipeline finished up to master_products.csv")

if __name__ == "__main__":
    main()
