import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = "/Users/shail2169/Electronics.jsonl"
OUTPUT_DIR = "data/"
SAMPLE_SIZE = 200_000
HELPFULNESS_THRESHOLD = 3
RANDOM_STATE = 42

# ── Load & Filter ────────────────────────────────────────────────────────
def load_data(path, sample_size):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)

            # drop records with no text
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


# ── Label ────────────────────────────────────────────────────────────────
def create_labels(df, threshold):
    df["helpful"] = (df["helpful_vote"] >= threshold).astype(int)
    return df


# ── Split ────────────────────────────────────────────────────────────────
def split_data(df):
    train, temp = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["helpful"])
    val, test = train_test_split(temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp["helpful"])
    return train, val, test


# ── Main ─────────────────────────────────────────────────────────────────
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