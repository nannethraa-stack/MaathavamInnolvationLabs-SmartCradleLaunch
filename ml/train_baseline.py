import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

FEATURES = Path("ml/data/processed/donateacry_logmel.npz")
METADATA = Path("ml/data/processed/donateacry_logmel_metadata.csv")
REPORT = Path("ml/reports/baseline_grouped_cv.txt")


def main():
    X = np.load(FEATURES)["X"]

    with METADATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    labels = np.array([r["class"] for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])

    X_mean = X.mean(axis=2)
    X_std = X.std(axis=2)
    features = np.concatenate([X_mean, X_std], axis=1)

    classes = sorted(set(labels))
    report_lines = []

    report_lines.append("SMART CRADLE CRY PATTERN BASELINE")
    report_lines.append("=================================")
    report_lines.append("Model: RandomForestClassifier")
    report_lines.append("Features: 64 Mel mean + 64 Mel std")
    report_lines.append("Evaluation: 5-fold source-grouped CV")
    report_lines.append("Class weighting: balanced")
    report_lines.append("")

    scores = []

    for fold in range(1, 6):
        train_idx = folds != fold
        test_idx = folds == fold

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(features[train_idx], labels[train_idx])
        predictions = model.predict(features[test_idx])

        macro_f1 = f1_score(
            labels[test_idx],
            predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        scores.append(macro_f1)

        report_lines.append(f"FOLD {fold}")
        report_lines.append(f"Windows: {test_idx.sum()}")
        report_lines.append(f"Macro-F1: {macro_f1:.4f}")
        report_lines.append(
            classification_report(
                labels[test_idx],
                predictions,
                labels=classes,
                zero_division=0,
            )
        )

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores))

    report_lines.append("SUMMARY")
    report_lines.append("=======")
    report_lines.append(f"Mean Macro-F1: {mean_f1:.4f}")
    report_lines.append(f"Std Macro-F1: {std_f1:.4f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"FOLDS=5")
    print(f"MEAN_MACRO_F1={mean_f1:.4f}")
    print(f"STD_MACRO_F1={std_f1:.4f}")
    print(f"REPORT={REPORT}")


if __name__ == "__main__":
    main()
