# Pipeline for Part I: Q1. Reproducible Data Pipeline for Leaderboard Evaluation

import os
import json
import subprocess
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def download_data(script_path="bash_scripts/download_data.sh"):
    #Function to download data from Part 0
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

def assert_no_future_leakage(train_df, val_df, test_df):
    max_train_time = train_df["timestamp"].max()
    min_val_time = val_df["timestamp"].min()
    min_test_time = test_df["timestamp"].min()
    max_val_time = val_df["timestamp"].max()
    
    assert max_train_time <= min_val_time, f"Leakage! Train ends {max_train_time} but Val starts {min_val_time}"
    assert max_val_time <= min_test_time, f"Leakage! Val ends {max_val_time} but Test starts {min_test_time}"
    
    print("Assertion passed: No future-click leakage detected.")

def clean_mind_news(split):
    #Function to clean the mind dataset (news.tsv)
    news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    usecols = ["article_id", "category", "subcategory", "title", "abstract", "title_entities", "abstract_entities"]
    news_df = pd.read_csv(f"data_large/MINDlarge_{split}/news.tsv", sep="\t", header=None, names=news_cols, usecols=usecols)
    news_df = news_df.drop_duplicates(subset="article_id")
 
    news_df["article_id"] = news_df["article_id"].astype(str)
    news_df["category"] = news_df["category"].astype(str)
    news_df["subcategory"] = news_df["subcategory"].astype(str)
    news_df["title"] = news_df["title"].fillna("").astype(str)
    news_df["abstract"] = news_df["abstract"].fillna("").astype(str)
    title_ents = news_df["title_entities"].apply(extract_entities)
    abstract_ents = news_df["abstract_entities"].apply(extract_entities)
    news_df["entities"] = (title_ents + abstract_ents).apply(lambda lst: list(dict.fromkeys(lst)))
 
    return news_df[["article_id", "title", "abstract", "category", "subcategory", "entities"]]

def clean_mind_behaviors_train_val(split):
    #Function to clean the mind dataset (behaviors.tsv)
    behaviors_cols = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    df = pd.read_csv(f"data_large/MINDlarge_{split}/behaviors.tsv", sep="\t", header=None, names=behaviors_cols)
    df = df.drop_duplicates(subset="impression_id")
 
    df["impression_id"] = df["impression_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["history"] = df["history"].fillna("").astype(str).replace("nan", "").str.split()
    df["impressions"] = df["impressions"].fillna("").astype(str).replace("nan", "").str.split()
    df["candidates"] = df["impressions"].apply(
        lambda x: [item.split("-")[0] for item in x if "-" in item]
    )
    df["labels"] = df["impressions"].apply(
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
    ebnerd_articles = pd.read_parquet("data_large/ebnerd_large/articles.parquet")
    ebnerd_articles_clean = ebnerd_articles.drop_duplicates(subset="article_id")

    ebnerd_articles_clean["article_id"] = ebnerd_articles_clean["article_id"].astype(str)
    ebnerd_articles_clean["category"] = ebnerd_articles_clean["category_str"].fillna("").astype(str)
    ebnerd_articles_clean["subcategory"] = ebnerd_articles_clean["subcategory"].fillna("").astype(str)
    ebnerd_articles_clean["title"] = ebnerd_articles_clean["title"].fillna("").astype(str)
    ebnerd_articles_clean["abstract"] = ebnerd_articles_clean["subtitle"].fillna("").astype(str).str.replace(r'\r+|\n+', ' ', regex=True)
    ebnerd_articles_clean["body"] = ebnerd_articles_clean["body"].fillna("").astype(str).str.replace(r'\r+|\n+', ' ', regex=True)
    ebnerd_articles_clean["entities"] = ebnerd_articles_clean["ner_clusters"].apply(
        lambda x: [str(item) for item in x] if isinstance(x, (list, np.ndarray)) else []
    )

    return ebnerd_articles_clean[["article_id", "title", "abstract", "body", "category", "subcategory", "entities"]]

def clean_ebnerd_behaviors_large():
    ebnerd_behaviours_train = pd.read_parquet("data_large/ebnerd_large/train/behaviors.parquet")
    ebnerd_behaviours_val = pd.read_parquet("data_large/ebnerd_large/validation/behaviors.parquet")

    ebnerd_history_train = pd.read_parquet("data_large/ebnerd_large/train/history.parquet")
    ebnerd_history_val = pd.read_parquet("data_large/ebnerd_large/validation/history.parquet")

    behaviours_combined = pd.concat([ebnerd_behaviours_train, ebnerd_behaviours_val], ignore_index=True)
    behaviours_combined = behaviours_combined.drop_duplicates(subset="impression_id")    

    history_combined = pd.concat([ebnerd_history_train, ebnerd_history_val], ignore_index=True)
    history_combined = history_combined.drop_duplicates(subset="user_id")   

    behaviours_combined["impression_id"] = behaviours_combined["impression_id"].astype(str)
    behaviours_combined["user_id"] = behaviours_combined["user_id"].astype(str)
    behaviours_combined["timestamp"] = pd.to_datetime(behaviours_combined["impression_time"])

    history_combined["user_id"] = history_combined["user_id"].astype(str)
    history_combined["history"] = history_combined["article_id_fixed"].apply(
        lambda arr: [str(x) for x in arr] if arr is not None else []
    )
    behaviours_combined = behaviours_combined.merge(history_combined[["user_id", "history"]], on="user_id", how="left")
    behaviours_combined["history"] = behaviours_combined["history"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    behaviours_combined["candidates"] = behaviours_combined["article_ids_inview"].apply(
        lambda arr: [str(x) for x in arr] if arr is not None else []
    )
    behaviours_combined["labels"] = behaviours_combined.apply(
            lambda row: [
                1 if x in (row["article_ids_clicked"] if row["article_ids_clicked"] is not None else []) else 0 
                for x in (row["article_ids_inview"] if row["article_ids_inview"] is not None else [])
            ],
            axis=1
    )

    return behaviours_combined[["impression_id", "user_id", "history", "timestamp", "candidates", "labels"]]

def split_by_timestamp(df, test_days, val_days):
    max_date = df["timestamp"].max()
    test_threshold = max_date - pd.Timedelta(days=test_days)
    val_threshold = test_threshold - pd.Timedelta(days=val_days)

    test_df = df[df["timestamp"] > test_threshold].copy()
    val_df = df[(df["timestamp"] > val_threshold) & (df["timestamp"] <= test_threshold)].copy()
    train_df = df[df["timestamp"] <= val_threshold].copy()

    print(f"Train set: {len(train_df)} rows, Validation set: {len(val_df)} rows, Test set: {len(test_df)} rows")

    return train_df, val_df, test_df

def generate_embeddings(df, model="all-MiniLM-L6-v2"):
    df = df.copy()
    print(f"Generating joint embeddings using model: {model}")
    sentence_model = SentenceTransformer(model)    
    joint_text = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df["joint_embedding"] = sentence_model.encode(
        joint_text.tolist(), 
        show_progress_bar=True
    ).tolist()
    
    return df

def build_user_features(behaviours_df):
    behaviours_sorted = behaviours_df.sort_values("timestamp")
    latest_rows = behaviours_sorted.drop_duplicates(subset="user_id", keep="last")
    users_df = latest_rows[["user_id", "history"]].copy()
    users_df["user_id"] = users_df["user_id"].astype(str)
    users_df["history"] = users_df["history"].apply(lambda x: x if isinstance(x, list) else [])
    users_df["history_length"] = users_df["history"].apply(len)
    recency = behaviours_df.groupby("user_id")["timestamp"].max().reset_index().rename(columns={"timestamp": "recency"})
    users_df = users_df.merge(recency, on="user_id", how="left")

    return users_df.reset_index(drop=True)

def run_q1_mind():
    os.makedirs("processed_data_large", exist_ok=True)
    os.makedirs("split_data_large", exist_ok=True)
 
    #download_data()
 
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
    print("Mind train/dev/test saved.")

def run_q1_ebnerd():
    os.makedirs("processed_data_large", exist_ok=True)
    os.makedirs("split_data_large", exist_ok=True)

    ebnerd_articles = clean_ebnerd_articles_large()
    ebnerd_articles = generate_embeddings(ebnerd_articles, model="paraphrase-multilingual-MiniLM-L12-v2")
    ebnerd_articles.to_parquet("processed_data_large/ebnerd_articles_large.parquet", index=False)
    print("Cleaned articles data saved to processed_data_large/ebnerd_articles_large.parquet")

    ebnerd_behaviours = clean_ebnerd_behaviors_large()
    ebnerd_behaviours.to_parquet("processed_data_large/ebnerd_behaviors_large.parquet", index=False)
    ebnerd_train, ebnerd_val, ebnerd_test = split_by_timestamp(ebnerd_behaviours, test_days=7, val_days=7)

    train_users = build_user_features(ebnerd_train)
    val_users = build_user_features(ebnerd_val)
    test_users = build_user_features(ebnerd_test)

    ebnerd_train.to_parquet("split_data_large/ebnerd_train.parquet", index=False)
    ebnerd_val.to_parquet("split_data_large/ebnerd_val.parquet", index=False)
    ebnerd_test.to_parquet("split_data_large/ebnerd_test.parquet", index=False)
    train_users.to_parquet("split_data_large/ebnerd_train_users.parquet", index=False)
    val_users.to_parquet("split_data_large/ebnerd_val_users.parquet", index=False)
    test_users.to_parquet("split_data_large/ebnerd_test_users.parquet", index=False)
    print("EbNERD train/dev/test saved.")

if __name__ == "__main__":
    run_q1_mind()
    run_q1_ebnerd()