# ML Category Classifier

## 1. Short Project Summary
This project classifies German retail products into main categories, subcategories, tags, and diet labels based on product name and brand text. 
It addresses the problem of organizing noisy retail catalog data into a consistent taxonomy for analytics or downstream filtering. 
The ML task is multi-label text classification with a single mandatory main category and optional secondary labels. 
Input is a list of product names (optionally with brand); output is a table of predicted labels plus a main-category confidence score. 
The system ships both an offline training pipeline and a Streamlit demo for interactive inference. 

## 2. Technical Overview
The pipeline begins with raw product extracts from supermarket flyers and retailer sources stored as CSVs. 
A master product list is built by selecting `Produkt` and `Marke`, cleaning missing brands, deduplicating, and shuffling for batching. 
Labeling is performed via a semi-automated workflow where batches are sent to the OpenAI API to assign a single main category and optional sub/tag/diet labels. 
These labeled batches are consolidated into `master_training.csv`, which becomes the supervised training set. 

Text preprocessing is minimal and deterministic: the model concatenates `Produkt` and `Marke` into a single text field, lower-level cleaning happens during CSV preparation, and the classifier operates on the raw string. 
Feature extraction uses character n-gram TF-IDF (`analyzer="char_wb"`, n-grams 3–5) to capture spelling variants, abbreviations, and short product tokens common in retail text. 
The model is a One-vs-Rest Logistic Regression classifier, which supports multi-label outputs while remaining interpretable and fast to train on modest data. 
Training writes both the fitted pipeline (`category_classifier.pkl`) and the label schema (`label_columns.pkl`) for consistent inference. 

TF-IDF is used instead of neural embeddings because the dataset is small-to-midsized, labels are sparse, and the pipeline needs transparency and low infrastructure overhead. 
This assumes that surface-form character patterns are sufficient to separate categories, and that semantic generalization beyond observed strings is limited. 
As a result, performance will degrade on unseen product naming conventions or languages without retraining or richer embeddings. 

## 3. Folder Structure (Core Section)
```
ML_category_classifier
├── .idea
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── .gitignore
│   ├── misc.xml
│   ├── ML_category_classifier.iml
│   ├── modules.xml
│   └── vcs.xml
├── app
│   └── portfolio_app.py
├── ML Models
│   └── ml_category_classifier
│       ├── artifacts
│       │   ├── category_classifier.pkl
│       │   ├── category_model.pkl
│       │   ├── label_columns.pkl
│       │   └── tfidf_vectorizer.pkl
│       ├── data
│       │   ├── master
│       │   │   └── master_products.csv
│       │   ├── raw extracts
│       │   │   ├── aldi_preisaktion_2025-12-14_csv_ml_train.csv
│       │   │   ├── aldi_sued_04.10-01.11__2025-10-13_csv_ml_train.csv
│       │   │   ├── kaufland_angebote_dedup_2025-11-30_11-55_csv_ml_train.csv
│       │   │   ├── netto_angebote_2025-12-14_csv_ml_train.csv
│       │   │   ├── rewe_angebote_107produkte_2025-11-17_bis_2025-11-23_csv_ml_train.csv
│       │   │   ├── rewe_angebote_108produkte_2025-11-10_bis_2025-11-16_csv_ml_train.csv
│       │   │   ├── rewe_angebote_332produkte_2025-12-15_bis_2025-12-21_csv_ml_train.csv
│       │   │   └── rossmann13-17.10__2025-10-13_csv_ml_train.csv
│       │   ├── raw_flyers_processed
│       │   │   ├── aldi_preisaktion_2025-12-14.csv
│       │   │   ├── aldi_sued_04.10-01.11__2025-10-13.csv
│       │   │   ├── kaufland_angebote_dedup_2025-11-30_11-55.csv
│       │   │   ├── netto_angebote_2025-12-14.csv
│       │   │   ├── rewe_angebote_107produkte_2025-11-17_bis_2025-11-23.csv
│       │   │   ├── rewe_angebote_108produkte_2025-11-10_bis_2025-11-16.csv
│       │   │   ├── rewe_angebote_332produkte_2025-12-15_bis_2025-12-21.csv
│       │   │   └── rossmann13-17.10__2025-10-13.csv
│       │   ├── training
│       │   │   └── master_training.csv
│       │   ├── training_archive
│       │   │   ├── batch_001.csv
│       │   │   ├── batch_002.csv
│       │   │   ├── batch_003.csv
│       │   │   ├── batch_004.csv
│       │   │   ├── batch_005.csv
│       │   │   ├── batch_006.csv
│       │   │   ├── batch_007.csv
│       │   │   └── batch_008.csv
│       │   ├── draw_training_batch.py
│       │   ├── labeled_products.csv
│       │   ├── raw_products.csv
│       │   └── raw_products_labeled.csv
│       ├── evaluation
│       │   └── metrics.ipynb
│       ├── inference
│       │   └── classify_csv.py
│       ├── labeling
│       │   ├── archived
│       │   │   ├── batch_001.json
│       │   │   ├── batch_002.json
│       │   │   ├── batch_003.json
│       │   │   ├── batch_004.json
│       │   │   ├── batch_005.json
│       │   │   ├── batch_006.json
│       │   │   ├── batch_007.json
│       │   │   └── batch_008.json
│       │   ├── labeled_batches
│       │   │   ├── batch_001.csv
│       │   │   ├── batch_002.csv
│       │   │   ├── batch_003.csv
│       │   │   ├── batch_004.csv
│       │   │   ├── batch_005.csv
│       │   │   ├── batch_006.csv
│       │   │   ├── batch_007.csv
│       │   │   ├── batch_008.csv
│       │   │   └── batch_009.csv
│       │   ├── training_archive
│       │   │   └── training_batch_20251219_144116.csv
│       │   ├── gpt_label_products.py
│       │   └── json_to_master_training_csv.py
│       ├── scripts
│       │   ├── build_master_csv.py
│       │   ├── extract_relevant_columns.py
│       │   └── run_until_master.py
│       ├── training
│       │   ├── new
│       │   └── train_classifier.py
│       └── README.md
└── requirements.txt
```

**Folder responsibilities**
- `app/`: Streamlit UI for interactive classification; loads the trained pipeline and label schema and applies the same text concatenation used in training. 
- `ML Models/ml_category_classifier/artifacts/`: Serialized model artifacts, including the fitted TF-IDF + classifier pipeline and label column order. 
- `ML Models/ml_category_classifier/data/`: Raw extracts, processed flyer CSVs, master product list, and training datasets (including historical batch archives). 
- `ML Models/ml_category_classifier/labeling/`: GPT-assisted labeling workflow, archived JSON responses, and labeled CSV batches. 
- `ML Models/ml_category_classifier/scripts/`: Data preparation utilities for building the master CSV and extracting relevant columns. 
- `ML Models/ml_category_classifier/training/`: Training scripts and working directories for model training iterations. 
- `ML Models/ml_category_classifier/inference/`: Batch inference script for labeling CSVs using saved artifacts. 
- `ML Models/ml_category_classifier/evaluation/`: Notebook used for metrics analysis. 

**Folder interaction flow**
Data flows from `data/raw extracts` → `scripts/` (master list) → `labeling/` (GPT labels) → `data/training/master_training.csv` → `training/train_classifier.py` → `artifacts/`. 
Inference reads `artifacts/` and applies predictions to `data/raw_products.csv` or the Streamlit UI in `app/`. 

**Where key concerns live**
- **ML logic**: `training/train_classifier.py`, `inference/classify_csv.py`, and the model artifacts in `artifacts/`. 
- **Data preprocessing**: `scripts/` and `data/` (cleaning, deduplication, and batching). 
- **Experimentation**: `evaluation/metrics.ipynb` and `training/` iteration folders. 
- **Configuration**: Mostly embedded in scripts (paths, label schema, model settings); no standalone config module is present. 

## 4. Engineering & Design Decisions
A classical ML pipeline was chosen to keep training reproducible and lightweight while working with sparse, noisy product text. 
TF-IDF with character n-grams captures short tokens and spelling variants common in retail catalogs, making it robust without requiring large embedding models. 
A linear One-vs-Rest classifier provides straightforward multi-label predictions and simplifies debugging with explicit probability outputs. 

Modularity is achieved by separating data preparation, labeling, training, and inference into dedicated folders and scripts. 
This allows the labeling workflow to evolve independently from the model training code and keeps artifacts versionable. 
To scale, the project could add stronger quality checks for labels, move to word/phrase embeddings or transformer models, and introduce a configurable training pipeline. 
Another extension would be an automated evaluation report and dataset versioning to compare model runs over time. 

## 5. What This Project Demonstrates
This repository demonstrates applied NLP for product taxonomy classification using a multi-label pipeline. 
It includes data ingestion, dataset curation, GPT-assisted labeling, and classic ML training with reproducible artifacts. 
The structure highlights clear separation between preprocessing, modeling, and serving, which is relevant to ML engineering roles and applied data work. 
It is most relevant for roles that value end-to-end ML pipelines, practical NLP classification, and lightweight model deployment. 
