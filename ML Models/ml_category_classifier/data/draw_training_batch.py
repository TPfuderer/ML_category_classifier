import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MASTER_PATH = BASE_DIR / "data" / "master" / "master_products.csv"
TO_TRAIN_DIR = BASE_DIR / "data" / "to_train"
ARCHIVE_DIR = BASE_DIR / "data" / "training_archive"

BATCH_SIZE = 50

TO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MASTER_PATH)

if len(df) < BATCH_SIZE:
    raise ValueError("Not enough products left in master list.")

batch = df.iloc[:BATCH_SIZE].copy()
remaining = df.iloc[BATCH_SIZE:].copy()

# Batch-ID bestimmen
existing_batches = sorted(ARCHIVE_DIR.glob("batch_*.csv"))
batch_id = len(existing_batches) + 1
batch_name = f"batch_{batch_id:03d}.csv"

# Speichern
batch_path = TO_TRAIN_DIR / batch_name
archive_path = ARCHIVE_DIR / batch_name

batch.to_csv(batch_path, index=False)
batch.to_csv(archive_path, index=False)
remaining.to_csv(MASTER_PATH, index=False)

print(f"✅ Created {batch_name}")
print(f"→ {len(batch)} products to train")
print(f"→ {len(remaining)} remaining in master")
