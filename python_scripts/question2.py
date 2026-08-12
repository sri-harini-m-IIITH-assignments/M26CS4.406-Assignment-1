#Pipeline for Part I: Q2. Lexical Candidate Generation (BM25)

import re 
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def tokenize(text):
    return re.findall(r'\w+', text.lower()) if text else []

def build_bm25_index(article_df):
    article_ids = article_df['article_id'].astype(str).tolist()
    article_texts = (article_df['title'].fillna('') + " " + article_df['abstract'].fillna('')).tolist()
    tokenized_articles = [tokenize(text) for text in article_texts]
    article_dict = dict(zip(article_ids, article_texts))
    bm25 = BM25Okapi(tokenized_articles)
    return bm25, article_ids, article_dict

def build_user_query(user_history, article_dict, max_history_length=5):
    if user_history is None or len(user_history) == 0:
        return []
    recent_history = user_history[-max_history_length:]
    query_text = " ".join([article_dict[article_id] for article_id in recent_history if article_id in article_dict])
    return tokenize(query_text)

def retrieve_top_k(bm25, article_ids, user_query, k=5):
    scores = bm25.get_scores(user_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_article_ids = [article_ids[i] for i in top_k_indices]
    return top_k_article_ids

def evaluate_bm25(behaviours_df, users_df, articles_df, k_list=[50, 100, 200]):
    bm25, article_ids, article_dict = build_bm25_index(articles_df)

    user_history_map = dict(zip(users_df['user_id'], users_df['history']))

    recalls = {k: [] for k in k_list}
    max_k = max(k_list)
    
    for user_id in tqdm(users_df['user_id'], desc="Evaluating users"):
        user_history = user_history_map.get(user_id, [])
        if user_history is None or len(user_history) == 0:
            continue
        
        user_query = build_user_query(user_history, article_dict)
        if not user_query:
            continue

        top_max_k_articles = retrieve_top_k(bm25, article_ids, user_query, k=max_k)
        relevant_articles = set(str(x) for x in user_history)

        for k in k_list:
            retrieved_k = set(top_max_k_articles[:k])
            true_positives = len(relevant_articles.intersection(retrieved_k))
            recall = true_positives / len(relevant_articles) if relevant_articles else 0.0
            recalls[k].append(recall)

    avg_recalls = {k: np.mean(recalls[k]) if recalls[k] else 0.0 for k in k_list}
    return avg_recalls

def run_q2_dataset(dataset_name, behaviors_path, users_path, articles_path, k=5):
    #Function to run Q2 pipeline
    print(f"Evaluating BM25 for {dataset_name} dataset...")
    behaviours_df = pd.read_parquet(behaviors_path)
    users_df = pd.read_parquet(users_path)
    articles_df = pd.read_parquet(articles_path)

    avg_recalls = evaluate_bm25(behaviours_df, users_df, articles_df, k_list=[50, 100, 200])
    
    print(f"Dataset: {dataset_name}")
    for k, recall in avg_recalls.items():
        print(f"Average Recall@{k}: {recall:.4f}")

def run_q2():
    run_q2_dataset(
        dataset_name="MIND-Small Validation",
        behaviors_path="split_data/mind_val.parquet",
        users_path="split_data/mind_val_users.parquet",
        articles_path="processed_data/cleaned_mind_news.parquet"
    )
    run_q2_dataset(
        dataset_name="EB-NeRD Validation",
        behaviors_path="split_data/ebnerd_val.parquet",
        users_path="split_data/ebnerd_val_users.parquet",
        articles_path="processed_data/cleaned_ebnerd_articles.parquet"
    )

if __name__ == "__main__":
    run_q2()