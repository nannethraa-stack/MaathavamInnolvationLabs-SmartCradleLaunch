import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
FEATURES = BASE_DIR / "data" / "processed" / "donateacry_logmel.npz"
METADATA = BASE_DIR / "data" / "processed" / "donateacry_logmel_metadata.csv"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "donateacry_cnn_bootstrap.pt"
METADATA_PATH = MODELS_DIR / "donateacry_cnn_bootstrap.json"
REPORT_PATH = REPORTS_DIR / "cnn_grouped_cv.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_SEED = 42


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


def load_dataset():
    X = np.load(FEATURES)["X"]

    with METADATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = sorted(set(row["class"] for row in rows))
    label_to_id = {label: index for index, label in enumerate(labels)}

    y = np.array(
        [label_to_id[row["class"]] for row in rows],
        dtype=np.int64,
    )

    folds = np.array(
        [int(row["fold"]) for row in rows],
        dtype=np.int64,
    )

    source_ids = np.array(
        [row["source_id"] for row in rows]
    )

    return X, y, folds, source_ids, labels


def make_model(num_classes):
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    return CryCNN(num_classes).to(DEVICE)


def train_model(X_train, y_train, labels):
    X_tensor = torch.from_numpy(X_train).float().unsqueeze(1)
    y_tensor = torch.from_numpy(y_train).long()

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = make_model(len(labels))

    train_classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=train_classes,
        y=y_train,
    )

    class_weights = np.ones(
        len(labels),
        dtype=np.float32,
    )

    for class_id, weight in zip(train_classes, weights):
        class_weights[class_id] = weight

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)

        print(
            f"EPOCH={epoch + 1}/{EPOCHS} "
            f"LOSS={average_loss:.4f}"
        )

    return model


def predict(model, X):
    model.eval()

    X_tensor = (
        torch.from_numpy(X)
        .float()
        .unsqueeze(1)
        .to(DEVICE)
    )

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

    return (
        predictions.cpu().numpy(),
        probabilities.cpu().numpy(),
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("SMART CRADLE CNN BOOTSTRAP TRAINING")
    print("====================================")
    print(f"DEVICE={DEVICE}")
    print(f"FEATURES={FEATURES}")

    X, y, folds, source_ids, labels = load_dataset()

    print(f"WINDOWS={len(X)}")
    print(f"SOURCES={len(set(source_ids))}")
    print(f"CLASSES={labels}")
    print(f"FOLDS={sorted(set(folds))}")

    report_lines = [
        "SMART CRADLE CRY PATTERN CNN BOOTSTRAP",
        "=======================================",
        "Dataset: Donate-a-Cry",
        "Purpose: bootstrap cry-pattern experiment",
        "NOT production Smart Cradle model",
        "",
        f"Windows: {len(X)}",
        f"Source recordings: {len(set(source_ids))}",
        f"Classes: {labels}",
        "Evaluation: 5-fold source-grouped CV",
        f"Epochs: {EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        "",
    ]

    fold_scores = []

    for fold in sorted(set(folds)):
        print("")
        print(f"===== FOLD {fold} =====")

        train_idx = folds != fold
        test_idx = folds == fold

        print(f"TRAIN_WINDOWS={train_idx.sum()}")
        print(f"TEST_WINDOWS={test_idx.sum()}")

        model = train_model(
            X[train_idx],
            y[train_idx],
            labels,
        )

        predictions, probabilities = predict(
            model,
            X[test_idx],
        )

        macro_f1 = f1_score(
            y[test_idx],
            predictions,
            labels=list(range(len(labels))),
            average="macro",
            zero_division=0,
        )

        fold_scores.append(macro_f1)

        report = classification_report(
            y[test_idx],
            predictions,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )

        print(f"FOLD={fold}")
        print(f"MACRO_F1={macro_f1:.4f}")
        print(report)

        report_lines.append(f"FOLD {fold}")
        report_lines.append(f"Windows: {test_idx.sum()}")
        report_lines.append(f"Macro-F1: {macro_f1:.4f}")
        report_lines.append(report)

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))

    print("")
    print("===== CROSS-VALIDATION SUMMARY =====")
    print(f"MEAN_MACRO_F1={mean_f1:.4f}")
    print(f"STD_MACRO_F1={std_f1:.4f}")

    report_lines.append("SUMMARY")
    report_lines.append("=======")
    report_lines.append(f"Mean Macro-F1: {mean_f1:.4f}")
    report_lines.append(f"Std Macro-F1: {std_f1:.4f}")
    report_lines.append("")

    print("")
    print("===== FINAL MODEL =====")
    print("Training on all available Donate-a-Cry windows...")

    final_model = train_model(
        X,
        y,
        labels,
    )

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "model_class": "CryCNN",
        "num_classes": len(labels),
        "labels": labels,
        "input_shape": list(X.shape[1:]),
        "device": str(DEVICE),
        "dataset": "Donate-a-Cry",
        "model_type": "bootstrap_cry_pattern_classifier",
        "training_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    model_metadata = {
        "model_version": "donateacry-cnn-bootstrap-v1",
        "model_status": "BOOTSTRAP_NOT_PRODUCTION",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "name": "Donate-a-Cry",
            "windows": int(len(X)),
            "source_recordings": int(len(set(source_ids))),
            "folds": sorted(
                int(value)
                for value in set(folds)
            ),
            "classes": labels,
        },
        "features": {
            "file": str(FEATURES),
            "representation": "log-mel spectrogram",
            "input_shape": list(X.shape[1:]),
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
        },
        "evaluation": {
            "method": "5-fold source-grouped CV",
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "fold_macro_f1": [
                float(score)
                for score in fold_scores
            ],
        },
        "production_requirements_remaining": [
            "real non-cry dataset",
            "OTHER class data",
            "Smart Cradle production recordings",
            "human-curated labels",
            "production grouped train/validation/test split",
            "uncertainty calibration",
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            model_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines.append(
        f"Model artifact: {MODEL_PATH}"
    )
    report_lines.append(
        f"Metadata artifact: {METADATA_PATH}"
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("")
    print("===== ARTIFACTS CREATED =====")
    print(f"MODEL={MODEL_PATH}")
    print(f"METADATA={METADATA_PATH}")
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
