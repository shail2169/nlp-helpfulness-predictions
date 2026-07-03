"""
features.py

Extracts NLP-derived and metadata features from cleaned review data.

Feature groups:
  - Text structure: length, word count, avg word length, lexical diversity
  - Readability: Flesch Reading Ease, Flesch-Kincaid Grade
  - Sentiment: VADER (positive, negative, neutral, compound)
  - Metadata: star rating, verified purchase

Input:  data/{split}.csv (from pipeline.py)
Output: data/{split}_features.csv
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

analyzer = SentimentIntensityAnalyzer()


def extract_features(df):
    """
    Compute all NLP and metadata features for a given DataFrame of reviews.
    Drops rows with missing text/title before processing, since feature
    extraction (e.g. .split(), textstat) requires valid strings.
    """
    df = df.copy()
    df = df.dropna(subset=["text", "review_title"]).reset_index(drop=True)

    # ── text structure features ──
    print("Extracting text length features...")
    df["review_length"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["title_length"] = df["review_title"].str.len()

    # ── lexical diversity: ratio of unique words to total words ──
    print("Extracting lexical diversity...")
    def lexical_diversity(text):
        words = text.lower().split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    df["lexical_diversity"] = df["text"].apply(lexical_diversity)

    # ── readability scores (higher = easier to read) ──
    print("Extracting readability scores...")
    df["flesch_reading_ease"] = df["text"].apply(textstat.flesch_reading_ease)
    df["flesch_kincaid_grade"] = df["text"].apply(textstat.flesch_kincaid_grade)

    # ── VADER sentiment scores ──
    print("Extracting sentiment scores...")
    df["vader_positive"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["pos"])
    df["vader_negative"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["neg"])
    df["vader_neutral"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["neu"])
    df["vader_compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # ── metadata features ──
    print("Extracting metadata features...")
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df


# feature columns used as model input (excludes text, labels, and IDs)
FEATURE_COLS = [
    "review_length", "word_count", "avg_word_length", "title_length",
    "lexical_diversity", "flesch_reading_ease", "flesch_kincaid_grade",
    "vader_positive", "vader_negative", "vader_neutral", "vader_compound",
    "verified_purchase", "rating"
]

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split}...")
        df = pd.read_csv(f"data/{split}.csv")
        df = extract_features(df)
        df.to_csv(f"data/{split}_features.csv", index=False)
        print(f"Saved data/{split}_features.csv with shape {df.shape}")
