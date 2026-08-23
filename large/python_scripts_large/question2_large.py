# Pipeline for Part I: Q2. Lexical Candidate Generation (BM25)

import re 
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import os 
import pickle
import nltk

try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

def get_stop_words(language):
    try:
        return set(stopwords.words(language))
    except Exception:
        return set()

def tokenize(text, stop_words=None):
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'\w+', text.lower())
    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]
    return tokens

def save_bm25_index(bm25, article_ids, article_dict, filepath):
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "article_ids": article_ids,
            "article_dict": article_dict
        }, f)
    print(f"BM25 index saved to {filepath}")

def load_bm25_index(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"BM25 index successfully loaded from {filepath}")
    return data["bm25"], data["article_ids"], data["article_dict"]

def build_bm25_index(article_df, stop_words=None):
    article_df = article_df.copy()
    article_ids = article_df['article_id'].astype(str).str.strip().tolist()
    article_titles = (article_df['title'].fillna('').astype(str).tolist())
    article_abstract = (article_df['abstract'].fillna('').astype(str).tolist())
    article_texts = [f"{title} {abstract}" for title, abstract in zip(article_titles, article_abstract)]
    tokenized_articles = [tokenize(text, stop_words=stop_words) for text in article_texts]
    article_dict = dict(zip(article_ids, article_titles))
    bm25 = BM25Okapi(tokenized_articles)
    return bm25, article_ids, article_dict

def build_user_query(user_history, article_dict, stop_words=None, max_history_length=5):
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
    return tokenize(query_text, stop_words=stop_words)

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

def evaluate_bm25(behaviours_df, articles_df, index_path=None, language="english", k_list=[50, 100, 200], max_history_length=5, history_path=None):
    stop_words = get_stop_words(language)
    if index_path and os.path.exists(index_path):
        bm25, article_ids, article_dict = load_bm25_index(index_path)
    else:
        bm25, article_ids, article_dict = build_bm25_index(articles_df, stop_words=stop_words)
        if index_path:
            save_bm25_index(bm25, article_ids, article_dict, index_path)

    recalls = {k: [] for k in k_list}
    max_k = max(k_list)

    query_cache = {}

    history_dict = {}
    if history_path and os.path.exists(history_path):
        history_df = pd.read_parquet(history_path)
        history_dict = dict(zip(history_df['user_id'], history_df['history']))

    # Swapped iterrows for itertuples to massively speed up the loop
    for row in tqdm(behaviours_df.itertuples(index=False), total=len(behaviours_df), desc="Evaluating BM25"):
        if hasattr(row, 'history'):
            user_history = row.history
        else:
            user_history = history_dict.get(row.user_id, [])
        raw_candidates = row.candidates
        raw_labels = row.labels

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

        user_query = build_user_query(user_history, article_dict, stop_words=stop_words, max_history_length=max_history_length)

        query_key = tuple(user_query)

        if not query_key:
            top_max_k_articles = []
        elif query_key in query_cache:
            top_max_k_articles = query_cache[query_key]
        else:
            top_max_k_articles = retrieve_top_k(bm25, article_ids, user_query, k=max_k)
            query_cache[query_key] = top_max_k_articles

        for k in k_list:
            retrieved_k = set(top_max_k_articles[:k])
            hits = len(relevant_articles.intersection(retrieved_k))
            recall = hits / len(relevant_articles)
            recalls[k].append(recall)

    avg_recalls = {k: float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in k_list}
    return avg_recalls

def run_q2_dataset(dataset_name, behaviors_path, articles_path, index_path, language="english", k_list=[50, 100, 200], history_path=None):
    print(f"Evaluating BM25 for {dataset_name} dataset...")
    behaviours_df = pd.read_parquet(behaviors_path)
    articles_df = pd.read_parquet(articles_path)

    avg_recalls = evaluate_bm25(behaviours_df, articles_df, index_path=index_path, language=language, k_list=k_list, history_path=history_path)    

    print(f"Dataset: {dataset_name}")
    for k, recall in avg_recalls.items():
        print(f"Average Recall@{k}: {recall:.4f}")

    with open("question2_large.txt", "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        for k, recall in avg_recalls.items():
            f.write(f"Average Recall@{k}: {recall:.4f}\n")
        f.write("\n")

def run_q2():
    # run_q2_dataset(
    #     dataset_name="MIND-Large Validation",
    #     behaviors_path="split_data_large/mind_dev.parquet",
    #     articles_path="processed_data_large/mind_news_dev.parquet",
    #     index_path="bm25_indexes_large/mind_bm25.pkl",
    #     language="english"
    # )
    run_q2_dataset(
        dataset_name="EB-NeRD Large Validation",
        behaviors_path="split_data_large/ebnerd_val.parquet",
        articles_path="processed_data_large/ebnerd_articles_large.parquet",
        index_path="bm25_indexes_large/ebnerd_large_bm25.pkl",
        language="danish",
        history_path="split_data_large/ebnerd_val_history.parquet" 
    )
    
if __name__ == "__main__":
    run_q2()