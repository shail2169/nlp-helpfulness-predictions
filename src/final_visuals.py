import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import os

FEATURE_COLS = [
    "review_length", "word_count", "avg_word_length", "title_length",
    "lexical_diversity", "flesch_reading_ease", "flesch_kincaid_grade",
    "vader_positive", "vader_negative", "vader_neutral", "vader_compound",
    "verified_purchase", "rating"
]

os.makedirs("experiments/figures", exist_ok=True)

# ── Retrain XGBoost (fast, no saved model needed) ──
print("Loading data and retraining XGBoost...")
train = pd.read_csv("data/train_features.csv")
test = pd.read_csv("data/test_features.csv")

X_train, y_train = train[FEATURE_COLS], train["helpful"]
X_test, y_test = test[FEATURE_COLS], test["helpful"]
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

xgb_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = xgb_model.predict(X_test)

# ── Load DistilBERT test predictions (from model_transformer_v2.py run) ──
print("Loading DistilBERT test predictions...")
bert_df = pd.read_csv("experiments/transformer_test_predictions.csv")
bert_probs = bert_df["prob_helpful"]
bert_preds = bert_df["pred"]
bert_true = bert_df["true_label"]

# ── 1. ROC curve overlay ──
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_probs)
auc_x = auc(fpr_x, tpr_x)
fpr_b, tpr_b, _ = roc_curve(bert_true, bert_probs)
auc_b = auc(fpr_b, tpr_b)

plt.figure(figsize=(6, 6))
plt.plot(fpr_x, tpr_x, label=f"XGBoost (AUC = {auc_x:.2f})")
plt.plot(fpr_b, tpr_b, label=f"DistilBERT (AUC = {auc_b:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: XGBoost vs. DistilBERT (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("experiments/figures/roc_curve_comparison.png", dpi=200)
plt.close()
print("Saved experiments/figures/roc_curve_comparison.png")

# ── 2. Confusion matrix heatmaps ──
def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Not Helpful", "Helpful"])
    plt.yticks([0, 1], ["Not Helpful", "Helpful"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved {filename}")

plot_confusion(y_test, xgb_preds, "XGBoost Confusion Matrix (Test)", "experiments/figures/confusion_matrix_xgboost.png")
plot_confusion(bert_true, bert_preds, "DistilBERT Confusion Matrix (Test)", "experiments/figures/confusion_matrix_distilbert.png")

# ── 3. XGBoost feature importance ──
importance = dict(zip(FEATURE_COLS, xgb_model.feature_importances_))
sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(7, 5))
plt.barh(list(sorted_importance.keys())[::-1], list(sorted_importance.values())[::-1])
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("experiments/figures/xgboost_feature_importance.png", dpi=200)
plt.close()
print("Saved experiments/figures/xgboost_feature_importance.png")

print("\nAll figures saved to experiments/figures/. Pull these directly into the final report and appendix.")
