# M26CS4.406 — Assignment 1: Lexical & Semantic Retrieval on EB-NeRD and MIND

## Sriharini Margapuri (2026701030)

**Course:** CS4.406 Information Retrieval & Extraction\
**Assignment:** Part I — Lexical & Semantic Retrieval\
**Due:** August 27, 2026 

This repository implements the Part I pipeline for the assignment: a
reproducible data pipeline, BM25 lexical candidate generation, FAISS-based
semantic candidate generation, and an offline evaluation harness, run against
both the **MIND-small** and **EB-NeRD (demo)** news recommendation datasets.

The pipeline is organized into four stages — one per assignment question
(Q1–Q4) — plus a single driver script that rebuilds everything from raw data
in one command, as required by the assignment.

## Project Structure

```
.
├── build_pipeline.py          # Main driver script
├── requirements.txt           # Python dependencies
├── bash_scripts/
│   ├── setup_env.sh           # Creates .venv and installs requirements
│   └── download_data.sh       # Downloads + extracts MIND and EB-NeRD data
├── python_scripts/
│   ├── question1.py           # Data cleaning, splitting, and embeddings
│   ├── question2.py           # BM25 lexical candidate generation
│   ├── question3.py           # FAISS semantic candidate generation
│   └── question4.py           # Offline evaluation harness
├── data/                      # Raw downloaded datasets (created by download_data.sh)
├── zip/                       # Downloaded zip files (created by download_data.sh)
├── processed_data/            # Cleaned news/article + behavior tables
├── split_data/                # Train / val / test splits (+ per-user features)
├── bm25_indexes/              # Pickled BM25 indexes
├── faiss_indexes/             # Pickled FAISS indexes
├── .gitignore                 # Mandatory .gitignore file
└── Sriharini_Margapuri_2026701030_M26CS4_406_Assignment_1.pdf # Design Document (Submitted on Moodle)
```

## Setup

The driver script (`build_pipeline.py`) handles environment setup automatically:

1. Runs `bash_scripts/setup_env.sh`, which creates a `.venv` virtual environment
   and installs everything in `requirements.txt`.
2. Re-executes itself inside `.venv` if it isn't already running there, so all
   downstream imports (`sentence-transformers`, `faiss`, `rank_bm25`, etc.) are
   guaranteed to come from the freshly-installed environment.

You don't need to activate the virtual environment yourself — just run the
driver script with your system Python and it will bootstrap everything.

## Usage

Run the full pipeline (all four stages, in order):

```bash
python build_pipeline.py
```

Run a single stage:

```bash
python build_pipeline.py --question 1   # Data pipeline
python build_pipeline.py --question 2   # BM25 evaluation
python build_pipeline.py --question 3   # FAISS evaluation
python build_pipeline.py --question 4   # Offline evaluation harness
```

> **Note:** Question 2–4 read parquet files produced by Question 1
> (`split_data/`, `processed_data/`), so Question 1 must be run at least once
> before the others.

## Pipeline Stages

Each stage below maps directly to a numbered question in the assignment brief.

### Q1 — Reproducible Data Pipeline (`question1.py`)
- Downloads raw data via `bash_scripts/download_data.sh`.
- Cleans and unifies the **MIND** (`news.tsv`, `behaviors.tsv`) and **EB-NeRD**
  (`articles.parquet`, `behaviors.parquet`, `history.parquet`) datasets into a
  common schema: articles (`title`, `abstract`, `body`, `category`,
  `subcategory`, `entities`) and behaviors/impressions (`history`,
  `candidates`, `labels`, `timestamp`).
- Splits each dataset into train / validation / test sets **chronologically**
  (by timestamp) to avoid future-click leakage, with an assertion that
  verifies no time-based leakage across splits.
- Generates sentence embeddings for article text (`all-MiniLM-L6-v2` for MIND,
  `paraphrase-multilingual-MiniLM-L12-v2` for EB-NeRD) and builds per-user
  history/recency features.
- Outputs cleaned tables to `processed_data/` and split tables to `split_data/`.

### Q2 — Lexical Candidate Generation (`question2.py`)
- Builds a **BM25** index over article titles + abstracts (with stopword
  removal, language-aware for English/Danish).
- For each impression, builds a user query from their recent reading history
  and retrieves the top-k candidates.
- Evaluates **Recall@k** (k = 50, 100, 200) on both datasets' validation sets,
  caching repeated queries for speed.
- Persists indexes to `bm25_indexes/` and writes results to `question2.txt`.

### Q3 — Semantic Candidate Generation (`question3.py`)
- Builds a **FAISS** flat inner-product index over normalized article
  embeddings from Q1.
- Represents each user as the mean (normalized) embedding of their recent
  history, falling back to a global centroid for users with no history.
- Evaluates **Recall@k** (k = 50, 100, 200) the same way as Q2.
- Persists indexes to `faiss_indexes/` and writes results to `question3.txt`.

### Q4 — Offline Evaluation Harness (`question4.py`)
- Re-scores candidates from Q2 (BM25) and Q3 (FAISS) at the *ranking* level
  (not just retrieval) and computes:
  - **Accuracy metrics:** AUC, MRR, NDCG@5, NDCG@10
  - **Beyond-accuracy metrics:** intra-list diversity@10, novelty@10
    (popularity-weighted), and catalog coverage@10
- Slices results into `all`, `cold` (≤5 history items), and `warm` user
  segments.
- Reports each metric with a bootstrapped 95% confidence interval
  (1000 resamples).
- Writes a full report to `question4.txt`.

### Q9 — Anti-Gaming / Leakage Check
Per the assignment's anti-gaming requirement, `question1.py` includes
`assert_no_future_leakage()`, which asserts that `train.timestamp.max() <=
val.timestamp.min()` and `val.timestamp.max() <= test.timestamp.min()` for
both datasets, enforcing the behaviour-window boundary before any
downstream stage runs.

This repo also contains a folder called `large` which is used for the larger forms fo the datasets needed for the CodaBench submissions. 

## Outputs

| File | Description |
|---|---|
| `processed_data/*.parquet` | Cleaned article and behavior tables |
| `split_data/*.parquet` | Chronological train/val/test splits + user features |
| `bm25_indexes/*.pkl` | Serialized BM25 indexes |
| `faiss_indexes/*.pkl` | Serialized FAISS indexes |
| `question2.txt` | BM25 recall@k results |
| `question3.txt` | FAISS recall@k results |
| `question4.txt` | Full offline evaluation report (AUC, MRR, NDCG, diversity, novelty, coverage) |

## Datasets

- **MIND-small**: [Microsoft News Dataset](https://msnews.github.io/) (train + dev splits)
- **EB-NeRD demo**: [Ekstra Bladet News Recommendation Dataset](https://recsys.eb.dk/) (demo split)

Both are downloaded automatically by `bash_scripts/download_data.sh` into `./zip`
and extracted into `./data`.
