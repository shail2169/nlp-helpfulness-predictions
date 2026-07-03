"""
model_classical.py

Trains and evaluates an XGBoost classifier on the engineered NLP + metadata
features to predict review helpfulness. Class weighting is applied to
address the imbalance between helpful and not-helpful reviews.

Input:  data/{split}_features.csv (from features.py)
Output: experiments/classical_results.json
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import json

# feature columns used as model input
FEATURE_COLS = [
    "review_length", "word_count", "avg_word_length", "title_length",
    "lexical_diversity", "flesch_reading_ease", "flesch_kincaid_grade",
    "vader_positive", "vader_negative", "vader_neutral", "vader_compound",
    "verified_purchase", "rating"
]


def load_splits():
    """Load the pre-computed feature CSVs for all three splits."""
    train = pd.read_csv("data/train_features.csv")
    val = pd.read_csv("data/val_features.csv")
    test = pd.read_csv("data/test_features.csv")
    return train, val, test


def train_model(train, val):
    """
    Train an XGBoost binary classifier.

    Sample weights (via compute_sample_weight) are used instead of
    oversampling to address class imbalance, since oversampling text-
    derived features can introduce redundant/duplicated feature rows.
    """
    X_train, y_train = train[FEATURE_COLS], train["helpful"]
    X_val, y_val = val[FEATURE_COLS], val["helpful"]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    return model


def evaluate(model, df, split_name):
    """
    Evaluate the model on a given split and print F1 (macro), AUC-ROC,
    and a full classification report. Macro F1 is used instead of
    accuracy because the dataset is heavily imbalanced (accuracy would
    be misleadingly high for a model that predicts the majority class).
    """
    X, y = df[FEATURE_COLS], df["helpful"]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    f1 = f1_score(y, preds, average="macro")
    auc = roc_auc_score(y, probs)

    print(f"\n── {split_name} ──")
    print(f"F1 Macro: {f1:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(classification_report(y, preds, target_names=["not helpful", "helpful"]))

    return {"split": split_name, "f1_macro": f1, "auc_roc": auc}


def feature_importance(model):
    """Print feature importance scores, sorted descending, for interpretability."""
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print("\n── Feature Importance ──")
    for feat, score in sorted_importance.items():
        print(f"{feat:<30} {score:.4f}")
    return sorted_importance


if __name__ == "__main__":
    print("Loading data...")
    train, val, test = load_splits()

    print("Training XGBoost...")
    model = train_model(train, val)

    results = []
    results.append(evaluate(model, val, "Validation"))
    results.append(evaluate(model, test, "Test"))

    feature_importance(model)

    with open("experiments/classical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiments/classical_results.json")
