import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "donateacry_cnn_bootstrap.pt"
METADATA_PATH = BASE_DIR / "models" / "donateacry_cnn_bootstrap.json"

TARGET_SR = 16000
WINDOW_SECONDS = 3
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.20


# Production cry-pattern contract.
#
# IMPORTANT:
# The current bootstrap checkpoint does NOT contain an OTHER output.
# Therefore OTHER must never be fabricated as a probability by this
# inference layer. A future production checkpoint must contain all
# six classes.
PRODUCTION_PATTERN_CLASSES = [
    "HUNGER",
    "PAIN",
    "DISCOMFORT",
    "TIRED",
    "BURPING",
    "OTHER",
]


# Current bootstrap model labels -> production terminology.
BOOTSTRAP_LABEL_MAP = {
    "hungry": "HUNGER",
    "belly_pain": "PAIN",
    "discomfort": "DISCOMFORT",
    "tired": "TIRED",
    "burping": "BURPING",
}


class CryCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CryInference:
    """
    Smart Cradle cry-pattern inference contract.

    IMPORTANT:
    The currently loaded Donate-a-Cry model is a bootstrap/research
    model only.

    It has five trained cry-pattern classes:
        HUNGER
        PAIN
        DISCOMFORT
        TIRED
        BURPING

    It has NO trained NON-CRY detector and NO trained OTHER class.

    Therefore:
      - the bootstrap model is never production-ready;
      - OTHER is never fabricated as a probability;
      - the probability distribution is explicitly marked incomplete;
      - inference remains REVIEW;
      - a future production model must provide the complete six-class
        cry-pattern distribution and a separate/non-cry-capable design.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {METADATA_PATH}"
            )

        with METADATA_PATH.open(encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.model_version = self.metadata["model_version"]
        self.model_status = self.metadata["model_status"]
        self.labels = self.metadata["dataset"]["classes"]

        self.device = torch.device("cpu")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.model = CryCNN(
            num_classes=checkpoint["num_classes"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def audio_to_logmel(audio):
        """
        Convert exactly one 3-second audio window
        into the same log-mel representation used during training.
        """

        expected_samples = TARGET_SR * WINDOW_SECONDS

        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples "
                f"for a {WINDOW_SECONDS}-second window, "
                f"received {len(audio)}."
            )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

        logmel = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        if logmel.shape != (64, 188):
            raise ValueError(
                f"Unexpected feature shape: {logmel.shape}. "
                f"Expected (64, 188)."
            )

        return logmel.astype(np.float32)

    def predict_audio_window(self, audio):
        """
        Run inference on exactly one 3-second audio window.

        Returns:
          - the model's actual trained probabilities;
          - the production class contract;
          - whether the distribution is complete;
          - explicit uncertainty/review state.

        No probability is invented for classes that the checkpoint
        was not trained to predict.
        """

        logmel = self.audio_to_logmel(audio)

        tensor = torch.from_numpy(logmel)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        probabilities_np = probabilities.cpu().numpy()

        probability_map = {
            label: float(probability)
            for label, probability in zip(
                self.labels,
                probabilities_np,
            )
        }

        mapped_probability_map = {
            BOOTSTRAP_LABEL_MAP.get(label, label): probability
            for label, probability in probability_map.items()
        }

        ordered = sorted(
            probability_map.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_probability = ordered[0]

        if len(ordered) > 1:
            second_probability = ordered[1][1]
        else:
            second_probability = 0.0

        margin = top_probability - second_probability

        mapped_prediction = BOOTSTRAP_LABEL_MAP.get(
            top_label,
            "OTHER",
        )

        # Determine which production classes are actually represented
        # by this checkpoint.
        represented_classes = sorted(
            mapped_probability_map.keys()
        )

        missing_production_classes = [
            label
            for label in PRODUCTION_PATTERN_CLASSES
            if label not in represented_classes
        ]

        distribution_complete = (
            len(missing_production_classes) == 0
            and len(mapped_probability_map)
            == len(PRODUCTION_PATTERN_CLASSES)
        )

        # The bootstrap model is not production-ready.
        uncertainty_reasons = []

        if top_probability < CONFIDENCE_THRESHOLD:
            uncertainty_reasons.append(
                "low_confidence"
            )

        if margin < MARGIN_THRESHOLD:
            uncertainty_reasons.append(
                "low_probability_margin"
            )

        if self.model_status != "PRODUCTION":
            uncertainty_reasons.append(
                "bootstrap_model"
            )

        if not distribution_complete:
            uncertainty_reasons.append(
                "incomplete_pattern_class_set"
            )

        # The current model has no NON-CRY training class/detector.
        uncertainty_reasons.append(
            "non_cry_detector_not_available"
        )

        decision = (
            "CONFIDENT"
            if not uncertainty_reasons
            else "REVIEW"
        )

        return {
            "model_version": self.model_version,
            "model_status": self.model_status,
            "production_ready": (
                self.model_status == "PRODUCTION"
                and distribution_complete
            ),

            "classification_scope": "cry_pattern",

            "production_pattern_classes": (
                PRODUCTION_PATTERN_CLASSES
            ),

            "represented_pattern_classes": (
                represented_classes
            ),

            "missing_pattern_classes": (
                missing_production_classes
            ),

            "distribution_complete": (
                distribution_complete
            ),

            "non_cry_probability": None,
            "non_cry_detector_available": False,

            "prediction_raw": top_label,
            "prediction": mapped_prediction,

            "confidence": float(top_probability),
            "margin": float(margin),

            # These are the actual probabilities produced by the
            # checkpoint. OTHER is deliberately absent because the
            # checkpoint was not trained on OTHER.
            "probabilities": mapped_probability_map,

            "decision": decision,
            "needs_review": decision == "REVIEW",
            "uncertainty_reasons": uncertainty_reasons,

            "feature": {
                "sample_rate": TARGET_SR,
                "window_seconds": WINDOW_SECONDS,
                "hop_seconds": 1,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "shape": [64, 188],
            },
        }


def load_audio_window(audio_path):
    """
    Load one audio file as a mono 16 kHz signal.

    The file must contain exactly one 3-second inference window.
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    expected_samples = TARGET_SR * WINDOW_SECONDS

    if len(audio) != expected_samples:
        raise ValueError(
            f"Audio must contain exactly "
            f"{WINDOW_SECONDS} seconds "
            f"({expected_samples} samples). "
            f"Received {len(audio)} samples."
        )

    return audio.astype(np.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Cradle cry-pattern inference"
    )

    parser.add_argument(
        "audio",
        help="Path to a 3-second audio WAV file",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio = load_audio_window(audio_path)

    inference = CryInference()

    result = inference.predict_audio_window(audio)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
