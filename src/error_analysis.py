"""
error_analysis.py

Runs both trained models on the test set and extracts misclassified
examples for qualitative error analysis (Milestone 3).

For each model, this script identifies:
  - False positives: predicted helpful, actually not helpful
  - False negatives: predicted not helpful, actually helpful

And prints representative examples of each, along with summary stats
on what characteristics (length, rating, sentiment) tend to appear in
each error category.

Input:  data/test_features.csv, trained models (retrained here for simplicity)
Output: experiments/error_analysis.json
"""

import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_COLS = [
    "review_length", "word_count", "avg_word_length", "title_length",
    "lexical_diversity", "flesch_reading_ease", "flesch_kincaid_grade",
    "vader_positive", "vader_negative", "vader_neutral", "vader_compound",
    "verified_purchase", "rating"
]


def train_xgb(train):
    """Retrain XGBoost (same config as model_classical.py) for error analysis."""
    X_train, y_train = train[FEATURE_COLS], train["helpful"]
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    return model


def categorize_errors(df, preds, y_true):
    """
    Split predictions into TP/TN/FP/FN and attach diagnostic columns
    for later inspection (length, rating, sentiment).
    """
    df = df.copy()
    df["pred"] = preds
    df["true"] = y_true

    df["error_type"] = "correct"
    df.loc[(df["pred"] == 1) & (df["true"] == 0), "error_type"] = "false_positive"
    df.loc[(df["pred"] == 0) & (df["true"] == 1), "error_type"] = "false_negative"

    return df


def summarize_errors(df, model_name):
    """Print summary statistics for each error category."""
    print(f"\n{'='*60}")
    print(f"{model_name} — Error Breakdown")
    print(f"{'='*60}")

    for error_type in ["false_positive", "false_negative"]:
        subset = df[df["error_type"] == error_type]
        print(f"\n── {error_type} (n={len(subset)}) ──")
        if len(subset) == 0:
            continue
        print(f"Avg review length: {subset['review_length'].mean():.0f}")
        print(f"Avg word count: {subset['word_count'].mean():.0f}")
        print(f"Avg rating: {subset['rating'].mean():.2f}")
        print(f"Avg VADER compound: {subset['vader_compound'].mean():.3f}")
        print(f"Verified purchase rate: {subset['verified_purchase'].mean():.2%}")

    return {
        "false_positive_count": int((df["error_type"] == "false_positive").sum()),
        "false_negative_count": int((df["error_type"] == "false_negative").sum()),
        "false_positive_avg_length": float(df[df["error_type"] == "false_positive"]["review_length"].mean()) if (df["error_type"] == "false_positive").sum() > 0 else None,
        "false_negative_avg_length": float(df[df["error_type"] == "false_negative"]["review_length"].mean()) if (df["error_type"] == "false_negative").sum() > 0 else None,
    }


def print_examples(df, error_type, n=3):
    """Print a few representative examples of a given error type for qualitative review."""
    subset = df[df["error_type"] == error_type].head(n)
    print(f"\n── Sample {error_type} examples ──")
    for i, row in subset.iterrows():
        print(f"\nRating: {row['rating']}  |  Length: {row['review_length']}  |  VADER compound: {row['vader_compound']:.3f}")
        print(f"Text: {row['text'][:300]}...")


if __name__ == "__main__":
    print("Loading data...")
    train = pd.read_csv("data/train_features.csv")
    test = pd.read_csv("data/test_features.csv")

    print("Training XGBoost for error analysis...")
    model = train_xgb(train)

    X_test = test[FEATURE_COLS]
    y_test = test["helpful"]
    preds = model.predict(X_test)

    df_errors = categorize_errors(test, preds, y_test)

    results = summarize_errors(df_errors, "XGBoost")

    print_examples(df_errors, "false_positive", n=3)
    print_examples(df_errors, "false_negative", n=3)

    with open("experiments/error_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/error_analysis.json")
