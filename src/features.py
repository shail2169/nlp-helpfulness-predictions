import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

analyzer = SentimentIntensityAnalyzer()

# ── Feature Extraction ───────────────────────────────────────────────────
def extract_features(df):
    df = df.copy()
    df = df.dropna(subset=["text", "review_title"]).reset_index(drop=True)
    print("Extracting text length features...")
    df["review_length"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["title_length"] = df["review_title"].str.len()

    print("Extracting lexical diversity...")
    def lexical_diversity(text):
        words = text.lower().split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    df["lexical_diversity"] = df["text"].apply(lexical_diversity)

    print("Extracting readability scores...")
    df["flesch_reading_ease"] = df["text"].apply(textstat.flesch_reading_ease)
    df["flesch_kincaid_grade"] = df["text"].apply(textstat.flesch_kincaid_grade)

    print("Extracting sentiment scores...")
    df["vader_positive"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["pos"])
    df["vader_negative"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["neg"])
    df["vader_neutral"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["neu"])
    df["vader_compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    print("Extracting metadata features...")
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df


FEATURE_COLS = [
    "review_length", "word_count", "avg_word_length", "title_length",
    "lexical_diversity", "flesch_reading_ease", "flesch_kincaid_grade",
    "vader_positive", "vader_negative", "vader_neutral", "vader_compound",
    "verified_purchase", "rating"
]

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        df = pd.read_csv(f"data/{split}.csv")
        df = extract_features(df)
        df.to_csv(f"data/{split}_features.csv", index=False)
        print(f"Saved data/{split}_features.csv with shape {df.shape}")
