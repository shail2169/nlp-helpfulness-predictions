import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import json

# ── Config ───────────────────────────────────────────────────────────────
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

        # freeze transformer weights
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
    print(f"AUC-ROC:  {auc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["not helpful", "helpful"]))

    return {"split": split_name, "f1_macro": round(f1, 4), "auc_roc": round(auc, 4)}

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_df = pd.read_csv("data/train_features.csv").dropna(subset=["text"])
    val_df = pd.read_csv("data/val_features.csv").dropna(subset=["text"])
    test_df = pd.read_csv("data/test_features.csv").dropna(subset=["text"])

    # stratified subsample
    # stratified subsample
    train_pos = train_df[train_df["helpful"] == 1].sample(min(TRAIN_SAMPLE // 2, train_df["helpful"].sum()), random_state=RANDOM_STATE)
    train_neg = train_df[train_df["helpful"] == 0].sample(TRAIN_SAMPLE // 2, random_state=RANDOM_STATE)
    train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    val_pos = val_df[val_df["helpful"] == 1].sample(min(VAL_SAMPLE // 2, val_df["helpful"].sum()), random_state=RANDOM_STATE)
    val_neg = val_df[val_df["helpful"] == 0].sample(VAL_SAMPLE // 2, random_state=RANDOM_STATE)
    val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    print(f"Train class distribution:\n{train_df['helpful'].value_counts()}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ReviewDataset(train_df["text"].tolist(), train_df["helpful"].tolist(), tokenizer)
    val_dataset = ReviewDataset(val_df["text"].tolist(), val_df["helpful"].tolist(), tokenizer)
    test_dataset = ReviewDataset(test_df["text"].tolist(), test_df["helpful"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DistilBertClassifier().to(device)

    # class weights for imbalance
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["helpful"].values)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    print("\nTraining...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.4f}")
        evaluate(model, val_loader, device, f"Validation (epoch {epoch+1})")

    results = []
    results.append(evaluate(model, val_loader, device, "Validation (final)"))
    results.append(evaluate(model, test_loader, device, "Test (final)"))

    with open("experiments/transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiments/transformer_results.json")
