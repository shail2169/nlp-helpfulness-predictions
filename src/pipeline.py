"""
pipeline.py

Loads raw Amazon Electronics review data (JSONL format), filters and cleans
records, constructs the binary "helpful" label, and splits the dataset into
train/validation/test sets.

Input:  Electronics.jsonl (Amazon Reviews 2023, McAuley Lab)
Output: data/train.csv, data/val.csv, data/test.csv
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = "/Users/shail2169/Electronics.jsonl"
OUTPUT_DIR = "data/"
SAMPLE_SIZE = 200_000          # number of records to sample from the raw ~20M
HELPFULNESS_THRESHOLD = 3      # min helpful_vote count to be labeled "helpful"
RANDOM_STATE = 42              # for reproducibility across runs


def load_data(path, sample_size):
    """
    Stream the raw JSONL file line by line (avoids loading ~20M records
    into memory at once), keep only records with non-null text and title,
    and stop once `sample_size` valid records have been collected.

    Returns a DataFrame with the raw fields needed downstream.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)

            # skip records with missing review text or title
            if not record.get("text") or not record.get("title"):
                continue

            records.append({
                "text": record["text"],
                "review_title": record["title"],
                "rating": record["rating"],
                "helpful_vote": record["helpful_vote"],
                "verified_purchase": record["verified_purchase"],
            })

            if len(records) >= sample_size:
                break

            if i % 500_000 == 0 and i > 0:
                print(f"Scanned {i:,} lines, kept {len(records):,}")

    return pd.DataFrame(records)


def create_labels(df, threshold):
    """
    Create the binary target variable.

    Note: the original project proposal defined "helpful" using a ratio
    of helpful_vote / total_vote. This dataset version does not include
    a total_vote field, so the label is instead a direct threshold on
    the raw helpful_vote count (see METHODS.docx for full justification).
    """
    df["helpful"] = (df["helpful_vote"] >= threshold).astype(int)
    return df


def split_data(df):
    """
    Stratified 70/15/15 train/val/test split, preserving class balance
    across all three sets.
    """
    train, temp = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["helpful"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp["helpful"]
    )
    return train, val, test


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH, SAMPLE_SIZE)
    print(f"Loaded: {df.shape}")

    df = create_labels(df, HELPFULNESS_THRESHOLD)
    print(f"Class distribution:\n{df['helpful'].value_counts()}")
    print(f"Positive rate: {df['helpful'].mean():.2%}")

    train, val, test = split_data(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    train.to_csv(f"{OUTPUT_DIR}train.csv", index=False)
    val.to_csv(f"{OUTPUT_DIR}val.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}test.csv", index=False)
    print("Saved train/val/test to data/")
