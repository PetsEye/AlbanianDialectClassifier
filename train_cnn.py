import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from common import (
    CSV_PATH, PROCESSED_DIR, MODEL_PATH, LABELS,
    categorize_dialect, speaker_id_from_file,
)
from model_utils import build_model, make_mel_transform, load_features

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_speaker_labels():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    if "status" in df.columns:
        df = df[df["status"] == "Success"]
    df = df.assign(
        id=df["file_name"].apply(speaker_id_from_file),
        label=df["dialect"].apply(categorize_dialect),
    ).dropna(subset=["label"])
    return dict(zip(df["id"], df["label"]))


class ChunkDataset(Dataset):
    def __init__(self, data_dir, speaker_labels, label_to_idx):
        self.items = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".wav"):
                continue
            label = speaker_labels.get(fname.split("_chunk_")[0])
            if label is None:
                continue
            self.items.append((os.path.join(data_dir, fname), label_to_idx[label]))
        self.mel = make_mel_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return load_features(path, self.mel), torch.tensor(label, dtype=torch.long)

    def class_counts(self, num_classes):
        return np.bincount([label for _, label in self.items], minlength=num_classes)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    accuracy = float(np.mean(np.array(preds) == np.array(targets))) if targets else 0.0
    return total_loss / max(len(loader), 1), accuracy, preds, targets


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    speaker_labels = load_speaker_labels()
    labels = [name for name in LABELS if name in set(speaker_labels.values())]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    train_ds = ChunkDataset(os.path.join(PROCESSED_DIR, "train"), speaker_labels, label_to_idx)
    val_ds = ChunkDataset(os.path.join(PROCESSED_DIR, "test"), speaker_labels, label_to_idx)
    if len(train_ds) == 0:
        raise ValueError("No training chunks found. Run preprocess.py first.")

    train_counts = train_ds.class_counts(len(labels))
    print(f"Classes: {labels}")
    print(f"Train chunks per class: {dict(zip(labels, train_counts.tolist()))}")
    print(f"Val chunks: {len(val_ds)}")

    class_weights = train_counts.sum() / (len(labels) * np.maximum(train_counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model(len(labels), pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_state = -1.0, None
    print("🚀 Starting CNN Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels_batch in progress_bar:
            images, labels_batch = images.to(device), labels_batch.to(device)
            loss = loss_fn(model(images), labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        scheduler.step()

        train_loss = total_loss / len(train_loader)
        if len(val_ds) > 0:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch+1}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.2%}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"Epoch {epoch+1}: train loss {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "val_accuracy": best_acc}, MODEL_PATH)
    print(f"🎉 Model saved to {MODEL_PATH} (best val acc {best_acc:.2%})")

    if len(val_ds) > 0:
        _, _, preds, targets = evaluate(model, val_loader, loss_fn, device)
        print("\nValidation report:")
        print(classification_report(targets, preds, labels=list(range(len(labels))), target_names=labels, zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted):")
        print(pd.DataFrame(
            confusion_matrix(targets, preds, labels=list(range(len(labels)))),
            index=labels, columns=labels,
        ))


if __name__ == "__main__":
    main()
