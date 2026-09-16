import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MANIFEST = Path("ml/data/processed/donateacry_manifest.csv")
OUT_ROOT = Path("ml/data/processed/donateacry_windows")

TARGET_SR = 16000
WINDOW_SECONDS = 3
HOP_SECONDS = 1

WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS
HOP_SAMPLES = TARGET_SR * HOP_SECONDS


def process_file(row):
    source = Path(row["file_path"])
    label = row["class"]
    source_id = row["source_id"]

    audio, _ = librosa.load(source, sr=TARGET_SR, mono=True)

    output_dir = OUT_ROOT / label / source_id
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for start in range(0, max(1, len(audio) - WINDOW_SAMPLES + 1), HOP_SAMPLES):
        window = audio[start:start + WINDOW_SAMPLES]

        if len(window) < WINDOW_SAMPLES:
            break

        output = output_dir / f"{source.stem}_w{count:03d}.wav"
        sf.write(output, audio[start:start + WINDOW_SAMPLES], TARGET_SR)
        count += 1

    return count


def main():
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"MANIFEST_FILES={len(rows)}")
    print(f"TARGET_SR={TARGET_SR}")
    print(f"WINDOW_SECONDS={WINDOW_SECONDS}")
    print(f"HOP_SECONDS={HOP_SECONDS}")
    print("DRY RUN: preprocessing not executed")


if __name__ == "__main__":
    main()
