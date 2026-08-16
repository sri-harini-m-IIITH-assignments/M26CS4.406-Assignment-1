# Pipeline for Part I: Q2. Lexical Candidate Generation (BM25)

import re 
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def tokenize(text):
    return re.findall(r'\w+', text.lower()) if text else []

def build_bm25_index(article_df):
    article_df = article_df.copy()
    article_ids = article_df['article_id'].astype(str).str.strip().tolist()
    article_titles = (article_df['title'].fillna('').astype(str).tolist())
    article_abstract = (article_df['abstract'].fillna('').astype(str).tolist())
    article_texts = [f"{title} {abstract}" for title, abstract in zip(article_titles, article_abstract)]
    tokenized_articles = [tokenize(text) for text in article_texts]
    article_dict = dict(zip(article_ids, article_titles))
    bm25 = BM25Okapi(tokenized_articles)
    return bm25, article_ids, article_dict

def build_user_query(user_history, article_dict, max_history_length=5):
    if user_history is None:
        return []

    if isinstance(user_history, str):
        history_list = user_history.strip().split()
    elif isinstance(user_history, (list, np.ndarray)):
        history_list = [str(article_id).strip() for article_id in user_history]
    else:
        return []

    if len(history_list) == 0:
        return []

    recent_history = history_list[-max_history_length:]
    query_titles = [article_dict[article_id] for article_id in recent_history if article_id in article_dict]
    query_text = " ".join(query_titles)
    return tokenize(query_text)

def retrieve_top_k(bm25, article_ids, user_query, k=200):
    if not user_query:
        return []

    scores = bm25.get_scores(user_query)
    if len(scores) <= k:
        top_k_indices = np.argsort(scores)[::-1]
    else:
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        
    return [article_ids[i] for i in top_k_indices]

def evaluate_bm25(behaviours_df, articles_df, k_list=[50, 100, 200], max_history_length=5):
    bm25, article_ids, article_dict = build_bm25_index(articles_df)

    recalls = {k: [] for k in k_list}
    max_k = max(k_list)

    for _, row in tqdm(behaviours_df.iterrows(), total=len(behaviours_df), desc="Evaluating BM25"):
        
        raw_candidates = row['candidates']
        raw_labels = row['labels']

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
        user_query = build_user_query(user_history, article_dict,  max_history_length=max_history_length)

        top_max_k_articles = retrieve_top_k(bm25, article_ids, user_query, k=max_k)

        for k in k_list:
            retrieved_k = set(top_max_k_articles[:k])
            hits = len(relevant_articles.intersection(retrieved_k))
            recall = hits / len(relevant_articles)
            recalls[k].append(recall)

    avg_recalls = {k: float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in k_list}
    return avg_recalls

def run_q2_dataset(dataset_name, behaviors_path, articles_path, k_list=[50, 100, 200]):
    print(f"Evaluating BM25 for {dataset_name} dataset...")
    behaviours_df = pd.read_parquet(behaviors_path)
    articles_df = pd.read_parquet(articles_path)

    avg_recalls = evaluate_bm25(behaviours_df, articles_df, k_list=k_list)
    
    print(f"Dataset: {dataset_name}")
    for k, recall in avg_recalls.items():
        print(f"Average Recall@{k}: {recall:.4f}")

    with open("question2.txt", "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        for k, recall in avg_recalls.items():
            f.write(f"Average Recall@{k}: {recall:.4f}\n")
        f.write("\n")

def run_q2():
    run_q2_dataset(
        dataset_name="MIND-Small Validation",
        behaviors_path="split_data/mind_val.parquet",
        articles_path="processed_data/mind_news.parquet"
    )
    run_q2_dataset(
        dataset_name="EB-NeRD Validation",
        behaviors_path="split_data/ebnerd_val.parquet",
        articles_path="processed_data/ebnerd_articles.parquet"
    )

if __name__ == "__main__":
    run_q2()