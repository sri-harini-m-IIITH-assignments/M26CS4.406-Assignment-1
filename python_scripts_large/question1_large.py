# Pipeline for Part I: Q1. Reproducible Data Pipeline for Leaderboard Evaluation

import os
import json
import subprocess
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

def download_data(script_path="bash_scripts/download_data.sh"):
    print("Downloading data...")
    try:
        subprocess.run(["bash", script_path], check=True)
        print("Data download completed.")
    except subprocess.CalledProcessError:
        print("Error occurred while downloading data.")
        sys.exit(1)

def extract_entities(text):
    if pd.isna(text) or str(text) == "nan":
        return []
    try:
        entities = json.loads(text)
        return [item["Label"] for item in entities if isinstance(item, dict) and "Label" in item]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

def clean_mind_news(split):
    news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    usecols = ["article_id", "category", "subcategory", "title", "abstract", "title_entities", "abstract_entities"]
    news_df = pd.read_csv(f"data_large/MINDlarge_{split}/news.tsv", sep="\t", header=None, names=news_cols, usecols=usecols)
    news_df = news_df.drop_duplicates(subset="article_id")
 
    news_df["article_id"] = news_df["article_id"].astype(str)
    news_df["category"] = news_df["category"].astype(str)
    news_df["subcategory"] = news_df["subcategory"].astype(str)
    news_df["title"] = news_df["title"].fillna("").astype(str)
    news_df["abstract"] = news_df["abstract"].fillna("").astype(str)
    
    print(f"Extracting title entities for MIND {split}...")
    title_ents = news_df["title_entities"].progress_apply(extract_entities)
    print(f"Extracting abstract entities for MIND {split}...")
    abstract_ents = news_df["abstract_entities"].progress_apply(extract_entities)
    
    print(f"Merging entities for MIND {split}...")
    news_df["entities"] = (title_ents + abstract_ents).progress_apply(lambda lst: list(dict.fromkeys(lst)))
 
    return news_df[["article_id", "title", "abstract", "category", "subcategory", "entities"]]

def clean_mind_behaviors_train_val(split):
    behaviors_cols = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    df = pd.read_csv(f"data_large/MINDlarge_{split}/behaviors.tsv", sep="\t", header=None, names=behaviors_cols)
    df = df.drop_duplicates(subset="impression_id")
 
    df["impression_id"] = df["impression_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["history"] = df["history"].fillna("").astype(str).replace("nan", "").str.split()
    df["impressions"] = df["impressions"].fillna("").astype(str).replace("nan", "").str.split()
    
    print(f"Parsing candidates for MIND {split} behaviors...")
    df["candidates"] = df["impressions"].progress_apply(
        lambda x: [item.split("-")[0] for item in x if "-" in item]
    )
    
    print(f"Parsing labels for MIND {split} behaviors...")
    df["labels"] = df["impressions"].progress_apply(
        lambda x: [int(item.split("-")[1]) for item in x if "-" in item]
    )
 
    return df[["impression_id", "user_id", "history", "timestamp", "candidates", "labels"]]

def clean_mind_behaviors_test():
    behaviors_cols = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    df = pd.read_csv(f"data_large/MINDlarge_test/behaviors.tsv", sep="\t", header=None, names=behaviors_cols)
    df = df.drop_duplicates(subset="impression_id")
 
    df["impression_id"] = df["impression_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["history"] = df["history"].fillna("").astype(str).replace("nan", "").str.split()
    df["candidates"] = df["impressions"].fillna("").astype(str).replace("nan", "").str.split()
    df["labels"] = None
 
    return df[["impression_id", "user_id", "history", "timestamp", "candidates", "labels"]]

def clean_ebnerd_articles_large():
    print("Loading EB-NeRD articles...")
    ebnerd_articles = pd.read_parquet("data_large/ebnerd_large/articles.parquet")
    ebnerd_articles_clean = ebnerd_articles.drop_duplicates(subset="article_id")

    ebnerd_articles_clean["article_id"] = ebnerd_articles_clean["article_id"].astype(str)
    ebnerd_articles_clean["category"] = ebnerd_articles_clean["category_str"].fillna("").astype(str)
    ebnerd_articles_clean["subcategory"] = ebnerd_articles_clean["subcategory"].fillna("").astype(str)
    ebnerd_articles_clean["title"] = ebnerd_articles_clean["title"].fillna("").astype(str)
    ebnerd_articles_clean["abstract"] = ebnerd_articles_clean["subtitle"].fillna("").astype(str).str.replace(r'\r+|\n+', ' ', regex=True)
    ebnerd_articles_clean["body"] = ebnerd_articles_clean["body"].fillna("").astype(str).str.replace(r'\r+|\n+', ' ', regex=True)
    
    print("Extracting entities for EB-NeRD articles...")
    ebnerd_articles_clean["entities"] = ebnerd_articles_clean["ner_clusters"].progress_apply(
        lambda x: [str(item) for item in x] if isinstance(x, (list, np.ndarray)) else []
    )

    return ebnerd_articles_clean[["article_id", "title", "abstract", "body", "category", "subcategory", "entities"]]

def clean_ebnerd_history(split):
    history_path = f"data_large/ebnerd_large/{split}/history.parquet"
    if not os.path.exists(history_path):
        return None
        
    df_hist = pd.read_parquet(history_path).drop_duplicates(subset="user_id")
    df_hist["user_id"] = df_hist["user_id"].astype(str)
    
    print(f"Cleaning history for EB-NeRD {split}...")
    df_hist["history"] = df_hist["article_id_fixed"].progress_apply(
        lambda arr: [str(x) for x in arr] if arr is not None else []
    )
    return df_hist[["user_id", "history"]]

def clean_ebnerd_behaviors(split):
    behaviors_path = f"data_large/ebnerd_large/{split}/behaviors.parquet"
    df = pd.read_parquet(behaviors_path).drop_duplicates(subset="impression_id")

    df["impression_id"] = df["impression_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["impression_time"])
    
    print(f"Extracting candidates for EB-NeRD {split} behaviors...")
    df["candidates"] = df["article_ids_inview"].progress_apply(
        lambda arr: [str(x) for x in arr] if arr is not None else []
    )
    
    if "article_ids_clicked" in df.columns:
        print(f"Extracting labels for EB-NeRD {split} behaviors (this might take a moment)...")
        df["labels"] = df.progress_apply(
            lambda row: [
                1 if x in (row["article_ids_clicked"] if row["article_ids_clicked"] is not None else []) else 0 
                for x in (row["article_ids_inview"] if row["article_ids_inview"] is not None else [])
            ],
            axis=1
        )
    else:
        df["labels"] = None

    return df[["impression_id", "user_id", "timestamp", "candidates", "labels"]]

def generate_embeddings(df, model="all-MiniLM-L6-v2"):
    df = df.copy()
    print(f"Generating joint embeddings using model: {model} (Native TQDM applies here)")
    sentence_model = SentenceTransformer(model)    
    joint_text = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df["joint_embedding"] = sentence_model.encode(
        joint_text.tolist(), 
        show_progress_bar=True
    ).tolist()
    
    return df

def build_user_features(behaviours_df, history_df=None):
    behaviours_sorted = behaviours_df.sort_values("timestamp")
    latest_rows = behaviours_sorted.drop_duplicates(subset="user_id", keep="last")
    users_df = latest_rows[["user_id"]].copy()
    users_df["user_id"] = users_df["user_id"].astype(str)
    
    if history_df is not None:
        users_df = users_df.merge(history_df, on="user_id", how="left")
        users_df["history"] = users_df["history"].apply(lambda x: x if isinstance(x, list) else [])
    else:
        users_df["history"] = [[] for _ in range(len(users_df))]

    users_df["history_length"] = users_df["history"].apply(len)
    recency = behaviours_df.groupby("user_id")["timestamp"].max().reset_index().rename(columns={"timestamp": "recency"})
    users_df = users_df.merge(recency, on="user_id", how="left")

    return users_df.reset_index(drop=True)

def run_q1_mind():
    os.makedirs("processed_data_large", exist_ok=True)
    os.makedirs("split_data_large", exist_ok=True)
 
    mind_news_train = clean_mind_news("train")
    mind_news_dev = clean_mind_news("dev")
    mind_news_test = clean_mind_news("test")
 
    mind_news_train = generate_embeddings(mind_news_train, model="all-MiniLM-L6-v2")
    mind_news_dev = generate_embeddings(mind_news_dev, model="all-MiniLM-L6-v2")
    mind_news_test = generate_embeddings(mind_news_test, model="all-MiniLM-L6-v2")
 
    mind_news_train.to_parquet("processed_data_large/mind_news_train.parquet", index=False)
    mind_news_dev.to_parquet("processed_data_large/mind_news_dev.parquet", index=False)
    mind_news_test.to_parquet("processed_data_large/mind_news_test.parquet", index=False)
    print("Cleaned news data saved to processed_data_large/mind_news_{train,dev,test}.parquet")
 
    mind_train = clean_mind_behaviors_train_val("train")
    mind_dev = clean_mind_behaviors_train_val("dev")
    mind_test = clean_mind_behaviors_test()
 
    mind_train_users = build_user_features(mind_train)
    mind_dev_users = build_user_features(mind_dev)
    mind_test_users = build_user_features(mind_test)
    mind_train.to_parquet("split_data_large/mind_train.parquet", index=False)
    mind_dev.to_parquet("split_data_large/mind_dev.parquet", index=False)
    mind_test.to_parquet("split_data_large/mind_test.parquet", index=False)
    mind_train_users.to_parquet("split_data_large/mind_train_users.parquet", index=False)
    mind_dev_users.to_parquet("split_data_large/mind_dev_users.parquet", index=False)
    mind_test_users.to_parquet("split_data_large/mind_test_users.parquet", index=False)
    print("MIND train/dev/test saved.")

def run_q1_ebnerd():
    os.makedirs("processed_data_large", exist_ok=True)
    os.makedirs("split_data_large", exist_ok=True)

    ebnerd_articles = clean_ebnerd_articles_large()
    ebnerd_articles = generate_embeddings(ebnerd_articles, model="paraphrase-multilingual-MiniLM-L12-v2")
    ebnerd_articles.to_parquet("processed_data_large/ebnerd_articles_large.parquet", index=False)
    print("Cleaned articles data saved to processed_data_large/ebnerd_articles_large.parquet")

    for split in ["train", "validation"]:
        beh_df = clean_ebnerd_behaviors(split)
        hist_df = clean_ebnerd_history(split)

        out_split_name = "val" if split == "validation" else split

        beh_df.to_parquet(f"split_data_large/ebnerd_{out_split_name}.parquet", index=False)
        
        if hist_df is not None:
            hist_df.to_parquet(f"split_data_large/ebnerd_{out_split_name}_history.parquet", index=False)

        users_df = build_user_features(beh_df, hist_df)
        users_df.to_parquet(f"split_data_large/ebnerd_{out_split_name}_users.parquet", index=False)
        
        print(f"EB-NeRD {split} processing completed.")

if __name__ == "__main__":
    run_q1_mind()
    run_q1_ebnerd()