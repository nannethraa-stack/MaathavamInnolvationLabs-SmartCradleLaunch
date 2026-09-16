import csv
from pathlib import Path

import librosa
import numpy as np

MANIFEST = Path("ml/data/processed/donateacry_grouped_cv.csv")
FEATURES = Path("ml/data/processed/donateacry_logmel.npz")
METADATA = Path("ml/data/processed/donateacry_logmel_metadata.csv")

TARGET_SR = 16000
WINDOW_SECONDS = 3
HOP_SECONDS = 1
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

WINDOW_SAMPLES = TARGET_SR * WINDOW_SECONDS
HOP_SAMPLES = TARGET_SR * HOP_SECONDS


def extract_file(row):
    audio, _ = librosa.load(row["file_path"], sr=TARGET_SR, mono=True)
    windows = []

    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        window = audio[start:start + WINDOW_SAMPLES]

        mel = librosa.feature.melspectrogram(
            y=window,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

        logmel = librosa.power_to_db(mel, ref=np.max)
        windows.append(logmel.astype(np.float32))

    return windows


def main():
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    features = []
    metadata = []

    for row in rows:
        windows = extract_file(row)

        for index, feature in enumerate(windows):
            features.append(feature)
            metadata.append({
                "source_id": row["source_id"],
                "class": row["class"],
                "fold": row["fold"],
                "source_file": row["file_path"],
                "window_index": index,
            })

    if not features:
        raise RuntimeError("No features extracted")

    X = np.stack(features)

    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES, X=X)

    with METADATA.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print(f"SOURCE_FILES={len(rows)}")
    print(f"WINDOWS={len(features)}")
    print(f"FEATURE_SHAPE={X.shape}")
    print(f"FEATURE_DTYPE={X.dtype}")
    print(f"FEATURES={FEATURES}")
    print(f"METADATA={METADATA}")


if __name__ == "__main__":
    main()
