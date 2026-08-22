# Pipeline for Part I: Q4. Offline Evaluation Harness

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, ndcg_score
from python_scripts_large.question2_large import build_user_query, get_stop_words, load_bm25_index
from python_scripts_large.question3_large import build_user_query_embedding, load_faiss_index
from collections import Counter

COLD_START_THRESHOLD = 5
K_LIST_BEYOND_ACCURACY = 10 

def mrr(labels_in_score_order):
    for i, l in enumerate(labels_in_score_order):
        if l == 1:
            return 1.0 / (i + 1)
    return 0.0

def bootstrap_ci(values, n_boot=1000, ci=95, seed=42):
    values = np.array([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(values.mean()), float(lo), float(hi)

def diversity_at_k(ranked_ids, article_embeddings, id_to_idx, k):
    recs = [aid for aid in ranked_ids[:k] if aid in id_to_idx]
    if len(recs) < 2:
        return None
    vecs = np.array([article_embeddings[id_to_idx[aid]] for aid in recs])
    unit_vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sim = unit_vecs @ unit_vecs.T
    upper = np.triu_indices(len(recs), k=1)
    return float(1 - sim[upper].mean())

def novelty_at_k(ranked_ids, popularity, k, min_pop=1e-4):
    recs = ranked_ids[:k]
    if not recs:
        return None
    return float(np.mean([-np.log2(popularity.get(aid, min_pop)) for aid in recs]))

def coverage_at_k(rec_sets, catalog_size):
    if not catalog_size:
        return {slice_name: 0.0 for slice_name in rec_sets}
    return {slice_name: len(ids) / catalog_size for slice_name, ids in rec_sets.items()}

def compute_item_popularity(train_behaviours_df):
    counter, total = Counter(), 0
    for row in train_behaviours_df.itertuples(index=False):
        candidates, labels = row.candidates, row.labels
        if candidates is None or labels is None:
            continue
        for c, l in zip(candidates, labels):
            if l == 1:
                counter[str(c)] += 1
                total += 1
    return {aid: cnt / total for aid, cnt in counter.items()} if total else {}

def score_bm25_fast(bm25, article_dict, id_to_idx, stop_words, history, candidates):
    query_tokens = build_user_query(history, article_dict, stop_words=stop_words)
    if not query_tokens:
        return np.zeros(len(candidates))

    cand_indices = [id_to_idx.get(c, -1) for c in candidates]
    scores = np.zeros(len(candidates))

    for q in query_tokens:
        if q not in bm25.idf:
            continue
        idf_val = bm25.idf[q]
        for i, c_idx in enumerate(cand_indices):
            if c_idx == -1:
                continue
            freq = bm25.doc_freqs[c_idx].get(q, 0)
            if freq == 0:
                continue
            num = freq * (bm25.k1 + 1)
            den = freq + bm25.k1 * (1 - bm25.b + bm25.b * (bm25.doc_len[c_idx] / bm25.avgdl))
            scores[i] += idf_val * (num / den)

    for i, c_idx in enumerate(cand_indices):
        if c_idx == -1:
            scores[i] = -1e9
    return scores

def score_faiss(article_embeddings, id_to_idx, history, candidates):
    user_vec = build_user_query_embedding(history, article_embeddings, id_to_idx)
    if user_vec is None:
        return np.zeros(len(candidates))
    return np.array([
        float(np.dot(user_vec[0], article_embeddings[id_to_idx[c]])) if c in id_to_idx else -1e9
        for c in candidates
    ])

def evaluate(behaviours_df, method, index_bundle, popularity):
    rows = {"all": [], "cold": [], "warm": []}
    rec_sets = {"all": set(), "cold": set(), "warm": set()}

    for row in tqdm(behaviours_df.itertuples(index=False), total=len(behaviours_df), desc=f"Offline Evaluation [{method}]"):
        raw_candidates, raw_labels = row.candidates, row.labels
        if raw_candidates is None or raw_labels is None or len(raw_candidates) == 0:
            continue

        if isinstance(raw_candidates, str):
            candidates = raw_candidates.strip().split()
        else:
            candidates = [str(c).strip() for c in raw_candidates]

        if isinstance(raw_labels, str):
            labels = [int(l) for l in raw_labels.strip().split()]
        else:
            labels = [int(l) for l in raw_labels]

        if len(set(labels)) < 2: 
            continue

        if isinstance(row.history, (list, np.ndarray)):
            history = [str(x) for x in row.history]
        elif isinstance(row.history, str):
            history = row.history.strip().split()
        else:
            history = []

        slice_name = "cold" if len(history) <= COLD_START_THRESHOLD else "warm"

        if method == "bm25":
            scores = score_bm25_fast(
                index_bundle["bm25"], index_bundle["article_dict"], index_bundle["id_to_idx"],
                index_bundle["stop_words"], history, candidates
            )
        else:
            scores = score_faiss(index_bundle["embeddings"], index_bundle["id_to_idx"], history, candidates)

        y_true = np.array([labels])
        y_score = np.array([scores])

        order = np.argsort(scores)[::-1]
        ranked_ids = [candidates[i] for i in order]
        ranked_labels = [labels[i] for i in order]

        record = {
            "auc": float(roc_auc_score(labels, scores)),
            "mrr": mrr(ranked_labels),
            "ndcg5": float(ndcg_score(y_true, y_score, k=5)),
            "ndcg10": float(ndcg_score(y_true, y_score, k=10)),
            "diversity": diversity_at_k(ranked_ids, index_bundle["embeddings"], index_bundle["emb_id_to_idx"], K_LIST_BEYOND_ACCURACY),
            "novelty": novelty_at_k(ranked_ids, popularity, K_LIST_BEYOND_ACCURACY),
        }

        rows["all"].append(record)
        rows[slice_name].append(record)
        rec_sets["all"].update(ranked_ids[:K_LIST_BEYOND_ACCURACY])
        rec_sets[slice_name].update(ranked_ids[:K_LIST_BEYOND_ACCURACY])

    return rows, rec_sets

def summarize(rows, rec_sets, catalog_size):
    metric_names = ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]
    coverage_by_slice = coverage_at_k(rec_sets, catalog_size)
    summary = {}
    for slice_name, records in rows.items():
        slice_summary = {"n": len(records)}
        for m in metric_names:
            vals = [r[m] for r in records if r[m] is not None]
            mean, lo, hi = bootstrap_ci(vals)
            slice_summary[m] = {"mean": mean, "ci_lower": lo, "ci_upper": hi, "n_valid": len(vals)}
        slice_summary["coverage"] = coverage_by_slice[slice_name]
        summary[slice_name] = slice_summary
    return summary

def write_results(f, dataset_name, method, summary):
    f.write(f"Dataset: {dataset_name} | Method: {method.upper()}\n")
    for slice_name, res in summary.items():
        f.write(f"  Slice: {slice_name} (n={res['n']})\n")
        for m in ["auc", "mrr", "ndcg5", "ndcg10", "diversity", "novelty"]:
            v = res[m]
            f.write(f"    {m}: {v['mean']:.4f} (95% CI [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}], n={v['n_valid']})\n")
        f.write(f"    coverage: {res['coverage']:.4f}\n")
    f.write("\n")

def run_q4_dataset(dataset_name, train_path, val_path, bm25_index_path, faiss_index_path, language="english"):
    print(f"Running Offline Evaluation for {dataset_name}...")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    stop_words = get_stop_words(language)

    bm25, bm25_article_ids, article_dict = load_bm25_index(bm25_index_path)
    bm25_id_to_idx = {aid: i for i, aid in enumerate(bm25_article_ids)}

    _, faiss_article_ids, faiss_embeddings, faiss_id_to_idx = load_faiss_index(faiss_index_path)

    popularity = compute_item_popularity(train_df)
    catalog_size = len(bm25_article_ids)

    with open("question4_large.txt", "a") as f:
        for method in ["bm25", "faiss"]:
            index_bundle = {
                "bm25": bm25, "article_dict": article_dict, "stop_words": stop_words,
                "id_to_idx": bm25_id_to_idx if method == "bm25" else faiss_id_to_idx,
                "embeddings": faiss_embeddings, "emb_id_to_idx": faiss_id_to_idx,
            }
            rows, rec_sets = evaluate(val_df, method, index_bundle, popularity)
            summary = summarize(rows, rec_sets, catalog_size)
            write_results(f, dataset_name, method, summary)

    print(f"Results for {dataset_name} written to question4_large.txt")

def run_q4():
    # run_q4_dataset(
    #     dataset_name="MIND-Large Validation",
    #     train_path="split_data_large/mind_train.parquet",
    #     val_path="split_data_large/mind_dev.parquet",
    #     bm25_index_path="bm25_indexes_large/mind_bm25.pkl",
    #     faiss_index_path="faiss_indexes_large/mind_faiss.pkl",
    #     language="english",
    # )
    run_q4_dataset(
        dataset_name="EB-NeRD Large Validation",
        train_path="split_data_large/ebnerd_train.parquet",
        val_path="split_data_large/ebnerd_val.parquet",
        bm25_index_path="bm25_indexes_large/ebnerd_large_bm25.pkl",
        faiss_index_path="faiss_indexes_large/ebnerd_large_faiss.pkl",
        language="danish",
    )

if __name__ == "__main__":
    run_q4()