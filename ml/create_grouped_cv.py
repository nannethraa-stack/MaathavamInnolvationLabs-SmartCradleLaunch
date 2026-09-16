import csv
from pathlib import Path

from sklearn.model_selection import GroupKFold

MANIFEST = Path("ml/data/processed/donateacry_manifest.csv")
OUTPUT = Path("ml/data/processed/donateacry_grouped_cv.csv")

with MANIFEST.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

groups = [r["source_id"] for r in rows]

splitter = GroupKFold(n_splits=5)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()) + ["fold"]
    )
    writer.writeheader()

    for fold, (_, test_idx) in enumerate(splitter.split(rows, groups=groups), start=1):
        for idx in test_idx:
            writer.writerow({**rows[idx], "fold": fold})

print("FILES=", len(rows))
print("SOURCE_GROUPS=", len(set(groups)))
print("FOLDS=5")
print("OUTPUT=", OUTPUT)
