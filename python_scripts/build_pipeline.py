#Main pipeline for Part I: Q1. Reproducible Data Pipeline
import subprocess
import sys
import pandas as pd
import numpy as np

def download_data(script_path):
    #Function to download data from Part 0
    print("Downloading data...")
    try:
        subprocess.run(["bash", script_path], check=True)
        print("Data download completed.")
    except subprocess.CalledProcessError:
        print("Error occurred while downloading data.")
        sys.exit(1)


def clean_mind_news():
    #Function to clean the mind dataset (news.tsv)
    news_cols = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
    MINDsmall_train = pd.read_csv("data/MINDsmall_train/news.tsv", sep="\t", header=None, names=news_cols, usecols=["article_id", "category", "subcategory", "title", "abstract"])
    MINDsmall_dev = pd.read_csv("data/MINDsmall_dev/news.tsv", sep="\t", header=None, names=news_cols, usecols=["article_id", "category", "subcategory", "title", "abstract"])

    news_combined = pd.concat([MINDsmall_train, MINDsmall_dev], ignore_index=True)
    news_combined = news_combined.drop_duplicates(subset="article_id")

    news_combined["article_id"] = news_combined["article_id"].astype(str)
    news_combined["category"] = news_combined["category"].astype(str)
    news_combined["subcategory"] = news_combined["subcategory"].astype(str)
    news_combined["title"] = news_combined["title"].astype(str)
    news_combined["abstract"] = news_combined["abstract"].astype(str)
    news_combined["body"] = ""

    return news_combined[["article_id", "category", "subcategory", "title", "abstract", "body"]]

def clean_mind_behaviors():
    #Function to clean the mind dataset (behaviors.tsv)
    behaviors_cols = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    MINDsmall_train = pd.read_csv("data/MINDsmall_train/behaviors.tsv", sep="\t", header=None, names=behaviors_cols, usecols=["impression_id", "user_id", "timestamp", "history", "impressions"])
    MINDsmall_dev = pd.read_csv("data/MINDsmall_dev/behaviors.tsv", sep="\t", header=None, names=behaviors_cols, usecols=["impression_id", "user_id", "timestamp", "history", "impressions"])

    behaviors_combined = pd.concat([MINDsmall_train, MINDsmall_dev], ignore_index=True)
    behaviors_combined = behaviors_combined.drop_duplicates(subset="impression_id")

    users_combined = behaviors_combined[["user_id", "history"]].copy()
    users_combined["user_id"] = users_combined["user_id"].astype(str)
    users_combined["history"] = users_combined["history"].astype(str).fillna("").str.split()
    users_combined = users_combined.drop_duplicates(subset="user_id")

    behaviors_combined["impression_id"] = behaviors_combined["impression_id"].astype(str)
    behaviors_combined["user_id"] = behaviors_combined["user_id"].astype(str)
    behaviors_combined["timestamp"] = pd.to_datetime(behaviors_combined["timestamp"])
    behaviors_combined["impressions"] = behaviors_combined["impressions"].apply(
        lambda x: (str(x).split())
    )
    behaviors_combined["candidates"] = behaviors_combined["impressions"].apply(
        lambda x: [item.split("-")[0] for item in x]
    )
    behaviors_combined["labels"] = behaviors_combined["impressions"].apply(
        lambda x: [int(item.split("-")[1]) for item in x]
    )

    behaviors_clean = behaviors_combined[["impression_id", "user_id", "timestamp", "candidates", "labels"]]

    return behaviors_clean, users_combined

def clean_ebnerd_articles():
    #Function to clean the ebnerd dataset (articles.parquet)
    ebnerd_articles = pd.read_parquet("data/ebnerd_demo/articles.parquet")
    ebnerd_articles_clean = ebnerd_articles.drop_duplicates(subset="article_id")

    ebnerd_articles_clean["article_id"] = ebnerd_articles_clean["article_id"].astype(str)
    ebnerd_articles_clean["category"] = ebnerd_articles_clean["category_str"].astype(str)
    ebnerd_articles_clean["subcategory"] = ebnerd_articles_clean["subcategory"].astype(str)
    ebnerd_articles_clean["title"] = ebnerd_articles_clean["title"].astype(str)
    ebnerd_articles_clean["abstract"] = ebnerd_articles_clean["subtitle"].fillna("").astype(str).str.replace(r'\r+|\n+', ' ', regex=True)
    ebnerd_articles_clean["body"] = ebnerd_articles_clean["body"].fillna("").astype(str).str.replace(r'\r+|\n+', ' ', regex=True)

    return ebnerd_articles_clean[["article_id", "category", "subcategory", "title", "abstract", "body"]]

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
        lambda arr: ([str(x) for x in arr])
    )
    users_combined = history_combined[["user_id", "history"]].copy()

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

    behaviors_clean = behaviours_combined[["impression_id", "user_id", "timestamp", "candidates", "labels"]]

    return behaviors_clean, users_combined

def split_by_timestamp(df, test_days, val_days):
    max_date = df["timestamp"].max()
    test_threshold = max_date - pd.Timedelta(days=test_days)
    val_threshold = test_threshold - pd.Timedelta(days=val_days)

    test_df = df[df["timestamp"] > test_threshold]
    val_df = df[(df["timestamp"] > val_threshold) & (df["timestamp"] <= test_threshold)].copy()
    train_df = df[df["timestamp"] <= val_threshold].copy()

    print(f"Train set: {len(train_df)} rows, Validation set: {len(val_df)} rows, Test set: {len(test_df)} rows")

    return train_df, val_df, test_df


def main():
    #Download data from Part 0
    download_data("bash_scripts/download_data.sh")

    #Clean the mind dataset (news.tsv)
    mind_news = clean_mind_news()
    mind_news.to_csv("processed_data/cleaned_mind_news.csv", index=False)
    print("Cleaned news data saved to processed_data/cleaned_mind_news.csv")

    #Clean the mind dataset (behaviors.tsv)
    mind_behaviors, mind_users = clean_mind_behaviors()
    mind_users.to_csv("processed_data/cleaned_mind_users.csv", index=False)
    mind_behaviors.to_csv("processed_data/cleaned_mind_behaviors.csv", index=False)
    print("Cleaned users data saved to processed_data/cleaned_mind_users.csv")
    print("Cleaned behaviors data saved to processed_data/cleaned_mind_behaviors.csv")

    #Clean the ebnerd dataset (articles.parquet)
    ebnerd_articles_df = clean_ebnerd_articles()
    ebnerd_articles_df.to_csv("processed_data/cleaned_ebnerd_articles.csv", index=False)
    print("Cleaned ebnerd articles data saved to processed_data/cleaned_ebnerd_articles.csv")

    #Clean the ebnerd dataset (behaviors.parquet + history.parquet)
    ebnerd_behaviors, ebnerd_users = clean_ebnerd_behaviors()
    ebnerd_users.to_csv("processed_data/cleaned_ebnerd_users.csv", index=False)
    ebnerd_behaviors.to_csv("processed_data/cleaned_ebnerd_behaviors.csv", index=False)
    print("Cleaned ebnerd users data saved to processed_data/cleaned_ebnerd_users.csv")
    print("Cleaned ebnerd behaviors data saved to processed_data/cleaned_ebnerd_behaviors.csv")

    #Train/Validation/Test split for mind dataset
    mind_train, mind_val, mind_test = split_by_timestamp(mind_behaviors, test_days=1, val_days=1)
    mind_train.to_csv("split_data/mind_train.csv", index=False)
    mind_val.to_csv("split_data/mind_val.csv", index=False)
    mind_test.to_csv("split_data/mind_test.csv", index=False)
    print("Mind dataset split into train, validation, and test sets.")

    #Train/Validation/Test split for ebnerd dataset
    ebnerd_train, ebnerd_val, ebnerd_test = split_by_timestamp(ebnerd_behaviors, test_days=1, val_days=1)
    ebnerd_train.to_csv("split_data/ebnerd_train.csv", index=False)
    ebnerd_val.to_csv("split_data/ebnerd_val.csv", index=False)
    ebnerd_test.to_csv("split_data/ebnerd_test.csv", index=False)
    print("Ebnerd dataset split into train, validation, and test sets.")


if __name__ == "__main__":
    main()