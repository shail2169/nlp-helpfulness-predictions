import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json

# ── Config ───────────────────────────────────────────────────────────────
# Same config as the original model_transformer.py so results are
# consistent with what's already reported (F1 macro 0.55, AUC-ROC 0.82).
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
TRAIN_SAMPLE = 10_000
VAL_SAMPLE = 3_000
RANDOM_STATE = 42


# ── Dataset ──────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ── Model ────────────────────────────────────────────────────────────────
class DistilBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


# ── Train ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluate ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device, split_name):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\n── {split_name} ──")
    print(f"F1 Macro: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["not helpful", "helpful"]))

    return {
        "split": split_name,
        "f1_macro": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "preds": all_preds,
        "probs": all_probs,
        "labels": all_labels,
    }


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_df = pd.read_csv("data/train_features.csv").dropna(subset=["text", "helpful"])
    val_df = pd.read_csv("data/val_features.csv").dropna(subset=["text", "helpful"])
    test_df = pd.read_csv("data/test_features.csv").dropna(subset=["text", "helpful"])

    def stratified_sample(df, n_total, seed):
        n_per_class = n_total // 2
        parts = []
        for label in [0, 1]:
            subset = df[df["helpful"] == label]
            parts.append(subset.sample(n=min(len(subset), n_per_class), random_state=seed))
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df = stratified_sample(train_df, TRAIN_SAMPLE, RANDOM_STATE)
    val_df = stratified_sample(val_df, VAL_SAMPLE, RANDOM_STATE)
    # test_df is used in full (the true, imbalanced distribution) and reset
    # so its index lines up 1:1 with the exported predictions below.
    test_df = test_df.reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print(f"Train class distribution:\n{train_df['helpful'].value_counts()}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = ReviewDataset(train_df["text"].tolist(), train_df["helpful"].tolist(), tokenizer)
    val_dataset = ReviewDataset(val_df["text"].tolist(), val_df["helpful"].tolist(), tokenizer)
    test_dataset = ReviewDataset(test_df["text"].tolist(), test_df["helpful"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DistilBertClassifier().to(device)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["helpful"])
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    print("\nTraining...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")
        evaluate(model, val_loader, device, f"Validation (epoch {epoch+1})")

    val_result = evaluate(model, val_loader, device, "Validation (final)")
    test_result = evaluate(model, test_loader, device, "Test (final)")

    # ── NEW: save the trained model so this 2-4 hour run never has to be repeated ──
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/distilbert_classifier.pt")
    print("\nModel checkpoint saved to models/distilbert_classifier.pt")

    # ── NEW: export test-set predictions for error analysis + charts ──
    test_predictions = test_df.copy()
    test_predictions["pred"] = test_result["preds"]
    test_predictions["prob_helpful"] = test_result["probs"]
    test_predictions["true_label"] = test_result["labels"]
    os.makedirs("experiments", exist_ok=True)
    test_predictions.to_csv("experiments/transformer_test_predictions.csv", index=False)
    print("Test predictions saved to experiments/transformer_test_predictions.csv")

    # ── NEW: confusion-matrix-derived error summary, same shape as error_analysis.py ──
    cm = confusion_matrix(test_result["labels"], test_result["preds"])
    tn, fp, fn, tp = cm.ravel()
    print("\n── DistilBERT Confusion Matrix (Test) ──")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    fp_mask = (test_predictions["pred"] == 1) & (test_predictions["true_label"] == 0)
    fn_mask = (test_predictions["pred"] == 0) & (test_predictions["true_label"] == 1)

    print(f"\n── false_positive (n={fp_mask.sum()}) ──")
    print(f"Avg review length: {test_predictions.loc[fp_mask, 'review_length'].mean():.0f}")
    print(f"Avg word count: {test_predictions.loc[fp_mask, 'word_count'].mean():.0f}")
    print(f"Avg rating: {test_predictions.loc[fp_mask, 'rating'].mean():.2f}")
    print(f"Avg VADER compound: {test_predictions.loc[fp_mask, 'vader_compound'].mean():.3f}")
    print(f"Verified purchase rate: {test_predictions.loc[fp_mask, 'verified_purchase'].mean()*100:.2f}%")

    print(f"\n── false_negative (n={fn_mask.sum()}) ──")
    print(f"Avg review length: {test_predictions.loc[fn_mask, 'review_length'].mean():.0f}")
    print(f"Avg word count: {test_predictions.loc[fn_mask, 'word_count'].mean():.0f}")
    print(f"Avg rating: {test_predictions.loc[fn_mask, 'rating'].mean():.2f}")
    print(f"Avg VADER compound: {test_predictions.loc[fn_mask, 'vader_compound'].mean():.3f}")
    print(f"Verified purchase rate: {test_predictions.loc[fn_mask, 'verified_purchase'].mean()*100:.2f}%")

    print("\n── Sample false_positive examples ──")
    for _, row in test_predictions.loc[fp_mask].head(3).iterrows():
        print(f"Rating: {row['rating']}  |  Length: {row['review_length']}  |  VADER compound: {row['vader_compound']:.3f}")
        print(f"Text: {row['text'][:300]}...")

    print("\n── Sample false_negative examples ──")
    for _, row in test_predictions.loc[fn_mask].head(3).iterrows():
        print(f"Rating: {row['rating']}  |  Length: {row['review_length']}  |  VADER compound: {row['vader_compound']:.3f}")
        print(f"Text: {row['text'][:300]}...")

    results = [
        {"split": val_result["split"], "f1_macro": val_result["f1_macro"], "auc_roc": val_result["auc_roc"]},
        {"split": test_result["split"], "f1_macro": test_result["f1_macro"], "auc_roc": test_result["auc_roc"]},
    ]
    with open("experiments/transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiments/transformer_results.json")
