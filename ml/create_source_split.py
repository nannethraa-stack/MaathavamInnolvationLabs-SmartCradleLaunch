import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

MANIFEST = Path("ml/data/processed/donateacry_manifest.csv")
OUTPUT = Path("ml/data/processed/donateacry_split_manifest.csv")

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

with MANIFEST.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
for row in rows:
    groups[row["source_id"]].append(row)

source_ids = list(groups)
random.shuffle(source_ids)

n = len(source_ids)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_ids = set(source_ids[:train_end])
val_ids = set(source_ids[train_end:val_end])
test_ids = set(source_ids[val_end:])

for row in rows:
    sid = row["source_id"]
    row["split"] = (
        "train" if sid in train_ids
        else "validation" if sid in val_ids
        else "test"
    )

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("SPLIT_MANIFEST_CREATED=", OUTPUT)
print("SOURCE_GROUPS=", n)
print("TRAIN_SOURCES=", len(train_ids))
print("VALIDATION_SOURCES=", len(val_ids))
print("TEST_SOURCES=", len(test_ids))

counts = Counter(r["split"] for r in rows)
print("FILES_BY_SPLIT=", dict(counts))

for split in ("train", "validation", "test"):
    classes = Counter(r["class"] for r in rows if r["split"] == split)
    print(split.upper(), dict(classes))
