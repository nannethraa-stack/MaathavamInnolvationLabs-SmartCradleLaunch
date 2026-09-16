import csv
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold

MANIFEST = Path("ml/data/processed/donateacry_manifest.csv")
OUTPUT = Path("ml/data/processed/donateacry_split_manifest.csv")

with MANIFEST.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

# One representative label per source for the stratification algorithm.
# Multi-class sources use their first sorted label; the source itself
# remains indivisible and all of its recordings stay in one split.
groups = {}
for row in rows:
    groups.setdefault(row["source_id"], set()).add(row["class"])

source_ids = sorted(groups)
source_labels = [sorted(groups[sid])[0] for sid in source_ids]

dummy_x = list(range(len(source_ids)))

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(sgkf.split(dummy_x, source_labels, source_ids))

# Choose fold 0 as test, then split remaining sources into
# approximately 82/18 => approximately 70/15 overall.
_, test_idx = folds[0]

test_sources = {source_ids[i] for i in test_idx}
remaining_sources = [sid for sid in source_ids if sid not in test_sources]

remaining_labels = [sorted(groups[sid])[0] for sid in remaining_sources]

sgkf2 = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=43
)

remaining_x = list(range(len(remaining_sources)))
remaining_groups = remaining_sources

folds2 = list(
    sgkf2.split(remaining_x, remaining_labels, remaining_groups)
)

_, val_idx = folds2[0]

validation_sources = {remaining_sources[i] for i in val_idx}
train_sources = set(remaining_sources) - validation_sources

for row in rows:
    sid = row["source_id"]
    row["split"] = (
        "test" if sid in test_sources
        else "validation" if sid in validation_sources
        else "train"
    )

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("STRATIFIED_GROUP_SPLIT_CREATED=", OUTPUT)
print("TRAIN_SOURCES=", len(train_sources))
print("VALIDATION_SOURCES=", len(validation_sources))
print("TEST_SOURCES=", len(test_sources))

for split in ("train", "validation", "test"):
    subset = [r for r in rows if r["split"] == split]
    print(split.upper(), "FILES=", len(subset), "CLASSES=", dict(Counter(r["class"] for r in subset)))
