import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

FEATURES = Path("ml/data/processed/donateacry_logmel.npz")
METADATA = Path("ml/data/processed/donateacry_logmel_metadata.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001


class CryCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def main():
    X = np.load(FEATURES)["X"]

    with METADATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = sorted(set(r["class"] for r in rows))
    label_to_id = {label: i for i, label in enumerate(labels)}

    y = np.array([label_to_id[r["class"]] for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])

    train_idx = folds != 1
    test_idx = folds == 1

    X_train = torch.from_numpy(X[train_idx]).float().unsqueeze(1)
    X_test = torch.from_numpy(X[test_idx]).float().unsqueeze(1)

    y_train = torch.from_numpy(y[train_idx]).long()
    y_test = torch.from_numpy(y[test_idx]).long()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = CryCNN(len(labels)).to(DEVICE)

    train_classes = np.unique(y_train.numpy())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=train_classes,
        y=y_train.numpy(),
    )

    class_weights = np.ones(len(labels), dtype=np.float32)

    for class_id, weight in zip(train_classes, weights):
        class_weights[class_id] = weight

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"DEVICE={DEVICE}")
    print(f"TRAIN_WINDOWS={len(X_train)}")
    print(f"TEST_WINDOWS={len(X_test)}")
    print(f"CLASSES={labels}")
    print(f"EPOCHS={EPOCHS}")
    print(f"BATCH_SIZE={BATCH_SIZE}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"EPOCH={epoch + 1}/{EPOCHS} "
            f"LOSS={total_loss / len(train_loader):.4f}"
        )

    model.eval()

    with torch.no_grad():
        logits = model(X_test.to(DEVICE))
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    actual = y_test.numpy()

    macro_f1 = f1_score(
        actual,
        predictions,
        labels=list(range(len(labels))),
        average="macro",
        zero_division=0,
    )

    print(f"MACRO_F1={macro_f1:.4f}")
    print(
        classification_report(
            actual,
            predictions,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()