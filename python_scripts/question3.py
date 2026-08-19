# Pipeline for Part I: Q3. Semantic Candidate Generation (Embeddings)

import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import os
import pickle

#This saving part was also slighly AI-generated
def save_faiss_index(index, article_ids, article_embeddings, id_to_idx, filepath):
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    serialized_index = faiss.serialize_index(index)
    
    with open(filepath, "wb") as f:
        pickle.dump({
            "index_bytes": serialized_index,
            "article_ids": article_ids,
            "article_embeddings": article_embeddings,
            "id_to_idx": id_to_idx
        }, f)
    print(f"FAISS index saved to {filepath}")

def load_faiss_index(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    index = faiss.deserialize_index(data["index_bytes"])
    print(f"FAISS index successfully loaded from {filepath}")
    return index, data["article_ids"], data["article_embeddings"], data["id_to_idx"]

def build_faiss_index(embeddings_df):
    article_ids = embeddings_df['article_id'].astype(str).str.strip().tolist()
    title_embeddings = np.array(embeddings_df['title_embedding'].tolist(), dtype=np.float32)
    abstract_embeddings = np.array(embeddings_df['abstract_embedding'].tolist(), dtype=np.float32)
    
    article_embeddings = (title_embeddings + abstract_embeddings) / 2.0
    
    faiss.normalize_L2(article_embeddings)
    
    dimension = article_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(article_embeddings)
    
    id_to_idx = {aid: idx for idx, aid in enumerate(article_ids)}
    
    return index, article_ids, article_embeddings, id_to_idx

def build_user_query_embedding(user_history, article_embeddings, id_to_idx, max_history_length=5):
    if user_history is None:
        return None

    if isinstance(user_history, str):
        history_list = user_history.strip().split()
    elif isinstance(user_history, (list, np.ndarray)):
        history_list = [str(aid).strip() for aid in user_history]
    else:
        return None

    if len(history_list) == 0:
        return None

    recent_history = history_list[-max_history_length:]
    valid_indices = [id_to_idx[aid] for aid in recent_history if aid in id_to_idx]

    if not valid_indices:
        return None

    user_embedding = article_embeddings[valid_indices]
    user_vector = np.mean(user_embedding, axis=0, keepdims=True).astype(np.float32) 

    faiss.normalize_L2(user_vector)
    return user_vector

def evaluate_faiss(behaviours_df, embeddings_df, index_path=None, k_list=[50, 100, 200], max_history_length=5):
    if index_path and os.path.exists(index_path):
        index, article_ids, article_embeddings, id_to_idx = load_faiss_index(index_path)
    else:
        index, article_ids, article_embeddings, id_to_idx = build_faiss_index(embeddings_df)
        if index_path:
            save_faiss_index(index, article_ids, article_embeddings, id_to_idx, index_path)

    global_centroid = np.mean(article_embeddings, axis=0, keepdims=True).astype(np.float32)
    faiss.normalize_L2(global_centroid)

    recalls = {k: [] for k in k_list}
    max_k = max(k_list)

    for _, row in tqdm(behaviours_df.iterrows(), total=len(behaviours_df), desc="Evaluating FAISS"):

        raw_candidates = row['candidates']
        raw_labels = row['labels']

        if raw_candidates is None or raw_labels is None:
            continue

        if isinstance(raw_candidates, str):
            candidates = raw_candidates.strip().split()
        else:
            candidates = [str(c).strip() for c in raw_candidates]

        if isinstance(raw_labels, str):
            labels = [int(l) for l in raw_labels.strip().split()]
        else:
            labels = [int(l) for l in raw_labels]

        relevant_articles = set(c for c, l in zip(candidates, labels) if l == 1)
        if not relevant_articles:
            continue

        user_history = row['history']
        user_query_embedding = build_user_query_embedding(user_history, article_embeddings, id_to_idx, max_history_length=max_history_length)
        
        if user_query_embedding is None:
            user_query_embedding = global_centroid

        D, I = index.search(user_query_embedding, max_k)
        top_k_indices = I[0]
        top_k_article_ids = [article_ids[i] for i in top_k_indices if i < len(article_ids)]

        for k in k_list:
            retrieved_set = set(top_k_article_ids[:k])
            hits = len(relevant_articles.intersection(retrieved_set))
            recall_at_k = hits / len(relevant_articles)
            recalls[k].append(recall_at_k)

    avg_recalls = {k: float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in k_list}
    return avg_recalls

def run_q3_dataset(dataset_name, behaviors_path, articles_path, index_path, k_list=[50, 100, 200]):
    print(f"Evaluating FAISS Semantic Retrieval for {dataset_name}...")
    behaviours_df = pd.read_parquet(behaviors_path)
    articles_df = pd.read_parquet(articles_path)

    avg_recalls = evaluate_faiss(behaviours_df, articles_df, index_path=index_path, k_list=k_list)
    
    print(f"Dataset: {dataset_name}")
    for k, recall in avg_recalls.items():
        print(f"Average Recall@{k}: {recall:.4f}")

    with open("question3.txt", "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        for k, recall in avg_recalls.items():
            f.write(f"Average Recall@{k}: {recall:.4f}\n")
        f.write("\n")

def run_q3():
    run_q3_dataset(
        dataset_name="MIND-Small Validation",
        behaviors_path="split_data/mind_val.parquet",
        articles_path="processed_data/mind_news.parquet",
        index_path="faiss_indexes/mind_faiss.pkl"
    )
    run_q3_dataset(
        dataset_name="EB-NeRD Validation",
        behaviors_path="split_data/ebnerd_val.parquet",
        articles_path="processed_data/ebnerd_articles.parquet",
        index_path="faiss_indexes/ebnerd_faiss.pkl"
    )

if __name__ == "__main__":
    run_q3()