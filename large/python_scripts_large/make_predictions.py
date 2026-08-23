import argparse
import os
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from python_scripts_large.question2_large import build_user_query, get_stop_words, load_bm25_index
from python_scripts_large.question3_large import build_user_query_embedding, load_faiss_index

def load_raw_mind_test_behaviors(path):
    cols = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    df["impression_id"] = df["impression_id"].astype(str)
    
    df["history"] = (
        df["history"]
        .fillna("")
        .astype(str)
        .replace("nan", "")
        .str.split()
    )
    df["candidates"] = (
        df["impressions"]
        .fillna("")
        .astype(str)
        .replace("nan", "")
        .str.split()
    )
    return df[["impression_id", "history", "candidates"]], None

def load_ebnerd_test_behaviors(test_dir):
    behaviors_path = os.path.join(test_dir, "behaviors.parquet")
    history_path = os.path.join(test_dir, "history.parquet")

    df_beh = pd.read_parquet(behaviors_path)
    df_hist = pd.read_parquet(history_path)

    df_beh["user_id"] = df_beh["user_id"].astype(str)
    df_hist["user_id"] = df_hist["user_id"].astype(str)
    
    # Create history dictionary instead of merging
    history_dict = dict(zip(df_hist['user_id'], df_hist['article_id_fixed']))

    df_beh["impression_id"] = df_beh["impression_id"].astype(str)
    
    df_beh["candidates"] = df_beh["article_ids_inview"].apply(
        lambda x: [str(i) for i in x] if isinstance(x, np.ndarray) else []
    )

    return df_beh[["impression_id", "user_id", "candidates"]], history_dict

def scores_to_ranks(scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()

def score_impression_bm25_fast(bm25, article_dict, id_to_idx, stop_words, history, candidates):
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

def score_impression_faiss(article_embeddings, id_to_idx, history, candidates):
    user_vec = build_user_query_embedding(history, article_embeddings, id_to_idx)
    if user_vec is None:
        return np.zeros(len(candidates))
    return np.array([
        float(np.dot(user_vec[0], article_embeddings[id_to_idx[c]])) if c in id_to_idx else -1e9
        for c in candidates
    ])

def generate_prediction_file(
    dataset_type,
    test_path_or_dir,
    method,
    bm25_index_path,
    faiss_index_path,
    language="english",
    output_dir=None,
):
    if output_dir is None:
        output_dir = os.path.join(dataset_type.lower(), method.lower())
    
    os.makedirs(output_dir, exist_ok=True)
    txt_name = "predictions.txt" if dataset_type.lower() == "ebnerd" else "prediction.txt"
    output_path = os.path.join(output_dir, txt_name)

    if dataset_type.lower() == "mind":
        test_df, history_dict = load_raw_mind_test_behaviors(test_path_or_dir)
    elif dataset_type.lower() == "ebnerd":
        test_df, history_dict = load_ebnerd_test_behaviors(test_path_or_dir)
    else:
        raise ValueError("dataset_type must be 'mind' or 'ebnerd'")

    if method == "bm25":
        bm25, article_ids, article_dict = load_bm25_index(bm25_index_path)
        id_to_idx = {aid: i for i, aid in enumerate(article_ids)}
        stop_words = get_stop_words(language)
    elif method == "faiss":
        _, article_ids, article_embeddings, id_to_idx = load_faiss_index(faiss_index_path)
    else:
        raise ValueError("method must be 'bm25' or 'faiss'")

    lines = []
    
    for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc=f"Scoring [{dataset_type} | {method}]"):
        if len(row.candidates) == 0:
            lines.append(f"{row.impression_id} []")
            continue

        # Resolve history exactly as in Q4
        if hasattr(row, 'history'):
            raw_history = row.history
        else:
            raw_history = history_dict.get(row.user_id, [])

        if isinstance(raw_history, (list, np.ndarray)):
            history = [str(x) for x in raw_history]
        elif isinstance(raw_history, str):
            history = raw_history.strip().split()
        else:
            history = []

        if method == "bm25":
            scores = score_impression_bm25_fast(bm25, article_dict, id_to_idx, stop_words, history, row.candidates)
        else:
            scores = score_impression_faiss(article_embeddings, id_to_idx, history, row.candidates)

        ranks = scores_to_ranks(scores)
        rank_str = ",".join(str(r) for r in ranks)
        lines.append(f"{row.impression_id} [{rank_str}]")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} predictions to {output_path}")
    return output_path

def zip_prediction(prediction_path, dataset_type="mind"):
    folder_dir = os.path.dirname(prediction_path)
    zip_filename = "predictions.zip" if dataset_type.lower() == "ebnerd" else "prediction.zip"
    zip_path = os.path.join(folder_dir, zip_filename)
    arc_filename = "predictions.txt" if dataset_type.lower() == "ebnerd" else "prediction.txt"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname=arc_filename)
    print(f"Zipped submission written to {zip_path}")
    return zip_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--test_path", required=True, help="Path to behaviors.tsv (MIND) or test/ folder (EB-NeRD)")
    parser.add_argument("--method", choices=["bm25", "faiss"], required=True)
    parser.add_argument("--bm25_index", default="bm25_indexes_large/mind_bm25.pkl")
    parser.add_argument("--faiss_index", default="faiss_indexes_large/mind_faiss.pkl")
    parser.add_argument("--language", default="english")
    args = parser.parse_args()

    pred_path = generate_prediction_file(
        dataset_type=args.dataset,
        test_path_or_dir=args.test_path,
        method=args.method,
        bm25_index_path=args.bm25_index,
        faiss_index_path=args.faiss_index,
        language=args.language,
    )
    zip_prediction(pred_path, dataset_type=args.dataset)