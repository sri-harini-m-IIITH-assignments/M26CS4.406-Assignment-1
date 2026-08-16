#Pipeline for Part I: Q1. Reproducible Data Pipeline

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

def clean_mind_news():
    #Function to clean the mind dataset (news.tsv)
    news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    MINDsmall_train = pd.read_csv("data/MINDsmall_train/news.tsv", sep="\t", header=None, names=news_cols, usecols=["article_id", "category", "subcategory", "title", "abstract", "title_entities", "abstract_entities"])
    MINDsmall_dev = pd.read_csv("data/MINDsmall_dev/news.tsv", sep="\t", header=None, names=news_cols, usecols=["article_id", "category", "subcategory", "title", "abstract", "title_entities", "abstract_entities"])

    news_combined = pd.concat([MINDsmall_train, MINDsmall_dev], ignore_index=True)
    news_combined = news_combined.drop_duplicates(subset="article_id")

    news_combined["article_id"] = news_combined["article_id"].astype(str)
    news_combined["category"] = news_combined["category"].astype(str)
    news_combined["subcategory"] = news_combined["subcategory"].astype(str)
    news_combined["title"] = news_combined["title"].fillna("").astype(str)
    news_combined["abstract"] = news_combined["abstract"].fillna("").astype(str)
    news_combined["body"] = ""
    title_ents = news_combined["title_entities"].apply(extract_entities)
    abstract_ents = news_combined["abstract_entities"].apply(extract_entities)
    news_combined["entities"] = (title_ents + abstract_ents).apply(lambda lst: list(dict.fromkeys(lst)))

    return news_combined[["article_id", "title", "abstract", "body", "category", "subcategory", "entities"]]

def clean_mind_behaviors():
    #Function to clean the mind dataset (behaviors.tsv)
    behaviors_cols = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    MINDsmall_train = pd.read_csv("data/MINDsmall_train/behaviors.tsv", sep="\t", header=None, names=behaviors_cols, usecols=["impression_id", "user_id", "timestamp", "history", "impressions"])
    MINDsmall_dev = pd.read_csv("data/MINDsmall_dev/behaviors.tsv", sep="\t", header=None, names=behaviors_cols, usecols=["impression_id", "user_id", "timestamp", "history", "impressions"])

    behaviors_combined = pd.concat([MINDsmall_train, MINDsmall_dev], ignore_index=True)
    behaviors_combined = behaviors_combined.drop_duplicates(subset="impression_id")

    behaviors_combined["impression_id"] = behaviors_combined["impression_id"].astype(str)
    behaviors_combined["timestamp"] = pd.to_datetime(behaviors_combined["timestamp"])
    behaviors_combined["user_id"] = behaviors_combined["user_id"].astype(str)
    behaviors_combined["history"] = behaviors_combined["history"].fillna("").astype(str).replace("nan", "").str.split()
    behaviors_combined["impressions"] = behaviors_combined["impressions"].fillna("").astype(str).replace("nan", "").str.split()
    behaviors_combined["candidates"] = behaviors_combined["impressions"].apply(
        lambda x: [item.split("-")[0] for item in x if "-" in item]
    )
    behaviors_combined["labels"] = behaviors_combined["impressions"].apply(
        lambda x: [int(item.split("-")[1]) for item in x if "-" in item]
    )

    behaviors_clean = behaviors_combined[["impression_id", "user_id", "history", "timestamp", "candidates", "labels"]]

    return behaviors_clean

def clean_ebnerd_articles():
    #Function to clean the ebnerd dataset (articles.parquet)
    ebnerd_articles = pd.read_parquet("data/ebnerd_demo/articles.parquet")
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

def clean_ebnerd_behaviors():
    #Function to clean the ebnerd dataset (behaviors.parquet + history.parquet)
    ebnerd_behaviours_train = pd.read_parquet("data/ebnerd_demo/train/behaviors.parquet")
    ebnerd_behaviours_val = pd.read_parquet("data/ebnerd_demo/validation/behaviors.parquet")

    ebnerd_history_train = pd.read_parquet("data/ebnerd_demo/train/history.parquet")
    ebnerd_history_val = pd.read_parquet("data/ebnerd_demo/validation/history.parquet")

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

    behaviours_clean = behaviours_combined[["impression_id", "user_id", "history", "timestamp", "candidates", "labels"]]

    return behaviours_clean

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
    print(f"Generating embeddings using model: {model}")
    sentence_model = SentenceTransformer(model)
    df["title_embedding"] = sentence_model.encode(df["title"].fillna("").tolist(), show_progress_bar=True).tolist()
    df["abstract_embedding"] = sentence_model.encode(df["abstract"].fillna("").tolist(), show_progress_bar=True).tolist()
    
    if df["body"].str.strip().astype(bool).any():
        df["body_embedding"] = sentence_model.encode(df["body"].fillna("").tolist(), show_progress_bar=True).tolist()
    else:
        dim = sentence_model.get_sentence_embedding_dimension()
        df["body_embedding"] = [np.zeros(dim).tolist() for _ in range(len(df))]

    return df

def build_user_features(behaviours_df):
    users_df = behaviours_df[["user_id", "history"]].drop_duplicates(subset="user_id").copy()
    users_df["user_id"] = users_df["user_id"].astype(str)
    users_df["history"] = users_df["history"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    recency = behaviours_df.groupby("user_id")["timestamp"].max().reset_index().rename(columns={"timestamp": "recency"})
    users_df = users_df.merge(recency, on="user_id", how="left")
    users_df = users_df.sort_values(by="recency", ascending=False).drop_duplicates(subset="user_id", keep="first").reset_index(drop=True)
    return users_df

def run_q1():
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("split_data", exist_ok=True)

    download_data()

    #Clean the mind dataset (news.tsv)
    mind_news = clean_mind_news()
    mind_news = generate_embeddings(mind_news, model="all-MiniLM-L6-v2")
    mind_news.to_parquet("processed_data/mind_news.parquet", index=False)
    print("Cleaned news data saved to processed_data/mind_news.parquet")

    #Clean the mind dataset (behaviors.tsv)
    mind_behaviors = clean_mind_behaviors()
    mind_behaviors.to_parquet("processed_data/mind_behaviors.parquet", index=False)
    print("Cleaned behaviors data saved to processed_data/mind_behaviors.parquet")

    #Train/Validation/Test split for mind dataset
    mind_train, mind_val, mind_test = split_by_timestamp(mind_behaviors, test_days=1, val_days=1)
    mind_train_users = build_user_features(mind_train)
    mind_val_users = build_user_features(mind_val)
    mind_test_users = build_user_features(mind_test)
    mind_train.to_parquet("split_data/mind_train.parquet", index=False)
    mind_val.to_parquet("split_data/mind_val.parquet", index=False)
    mind_test.to_parquet("split_data/mind_test.parquet", index=False)
    mind_train_users.to_parquet("split_data/mind_train_users.parquet", index=False)
    mind_val_users.to_parquet("split_data/mind_val_users.parquet", index=False)
    mind_test_users.to_parquet("split_data/mind_test_users.parquet", index=False)
    print("Mind dataset split into train, validation, and test sets.")

    #Clean the ebnerd dataset (articles.parquet)
    ebnerd_articles_df = clean_ebnerd_articles()
    ebnerd_articles_df = generate_embeddings(ebnerd_articles_df, model="paraphrase-multilingual-MiniLM-L12-v2")
    ebnerd_articles_df.to_parquet("processed_data/ebnerd_articles.parquet", index=False)
    print("Cleaned ebnerd articles data saved to processed_data/ebnerd_articles.parquet")

    #Clean the ebnerd dataset (behaviors.parquet + history.parquet)
    ebnerd_behaviors = clean_ebnerd_behaviors()
    ebnerd_behaviors.to_parquet("processed_data/ebnerd_behaviors.parquet", index=False)
    print("Cleaned ebnerd behaviors data saved to processed_data/ebnerd_behaviors.parquet")

    #Train/Validation/Test split for ebnerd dataset
    ebnerd_train, ebnerd_val, ebnerd_test = split_by_timestamp(ebnerd_behaviors, test_days=1, val_days=1)
    ebnerd_train_users = build_user_features(ebnerd_train)
    ebnerd_val_users = build_user_features(ebnerd_val)
    ebnerd_test_users = build_user_features(ebnerd_test)
    ebnerd_train.to_parquet("split_data/ebnerd_train.parquet", index=False)
    ebnerd_val.to_parquet("split_data/ebnerd_val.parquet", index=False)
    ebnerd_test.to_parquet("split_data/ebnerd_test.parquet", index=False)
    ebnerd_train_users.to_parquet("split_data/ebnerd_train_users.parquet", index=False)
    ebnerd_val_users.to_parquet("split_data/ebnerd_val_users.parquet", index=False)
    ebnerd_test_users.to_parquet("split_data/ebnerd_test_users.parquet", index=False)
    print("Ebnerd dataset split into train, validation, and test sets.")

if __name__ == "__main__":
    run_q1()