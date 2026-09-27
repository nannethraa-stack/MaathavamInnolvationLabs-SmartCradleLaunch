from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np


TARGET_SAMPLE_RATE = 16000
WINDOW_SAMPLES = 48000   # 3 seconds
HOP_SAMPLES = 16000      # 1 second


@dataclass
class RollingAudioWindow:
    device_id: str
    window_index: int
    sample_rate: int
    samples: np.ndarray
    source_event_ids: list[str]
    first_event_id: Optional[str] = None
    last_event_id: Optional[str] = None


class RollingAudioBuffer:
    """
    Per-device sample-count based rolling audio buffer.

    Input:
        Arbitrary PCM int16 chunks from MQTT.

    Output:
        Exact 3-second / 48,000-sample windows every 16,000 samples.

    MQTT packet boundaries are irrelevant. A window may span any number
    of incoming packets.
    """

    def __init__(
        self,
        device_id: str,
        sample_rate: int = TARGET_SAMPLE_RATE,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
    ):
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Unsupported sample rate {sample_rate}; "
                f"expected {TARGET_SAMPLE_RATE}"
            )

        if window_samples <= 0:
            raise ValueError("window_samples must be positive")

        if hop_samples <= 0:
            raise ValueError("hop_samples must be positive")

        if hop_samples > window_samples:
            raise ValueError("hop_samples cannot exceed window_samples")

        self.device_id = device_id
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.hop_samples = hop_samples

        self._samples = np.empty(0, dtype=np.int16)
        self._event_ids = deque()
        self._window_index = 0

    @property
    def buffered_samples(self) -> int:
        return int(self._samples.size)

    @property
    def buffered_seconds(self) -> float:
        return self.buffered_samples / self.sample_rate

    def clear(self) -> None:
        self._samples = np.empty(0, dtype=np.int16)
        self._event_ids.clear()

    def add_chunk(
        self,
        samples: np.ndarray,
        event_id: Optional[str] = None,
    ) -> list[RollingAudioWindow]:
        """
        Add one PCM int16 chunk.

        Returns every complete inference window currently available.
        """

        samples = np.asarray(samples)

        if samples.ndim != 1:
            samples = samples.reshape(-1)

        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)

        if samples.size == 0:
            return []

        if event_id:
            self._event_ids.append(
                (event_id, int(samples.size))
            )

        self._samples = np.concatenate(
            (self._samples, samples)
        )

        windows: list[RollingAudioWindow] = []

        while self._samples.size >= self.window_samples:
            window_samples = self._samples[:self.window_samples].copy()

            source_event_ids = self._events_covering_window(
                self.window_samples
            )

            first_event_id = (
                source_event_ids[0]
                if source_event_ids
                else None
            )

            last_event_id = (
                source_event_ids[-1]
                if source_event_ids
                else None
            )

            self._window_index += 1

            windows.append(
                RollingAudioWindow(
                    device_id=self.device_id,
                    window_index=self._window_index,
                    sample_rate=self.sample_rate,
                    samples=window_samples,
                    source_event_ids=source_event_ids,
                    first_event_id=first_event_id,
                    last_event_id=last_event_id,
                )
            )

            # Advance by exactly one second.
            hop = min(self.hop_samples, self._samples.size)

            self._samples = self._samples[hop:]

            self._consume_event_samples(hop)

        return windows

    def _events_covering_window(
        self,
        required_samples: int,
    ) -> list[str]:
        remaining = required_samples
        ids: list[str] = []

        for event_id, count in self._event_ids:
            if remaining <= 0:
                break

            ids.append(event_id)
            remaining -= count

        return ids

    def _consume_event_samples(self, count: int) -> None:
        remaining = count

        while remaining > 0 and self._event_ids:
            event_id, event_samples = self._event_ids[0]

            if event_samples <= remaining:
                remaining -= event_samples
                self._event_ids.popleft()
            else:
                self._event_ids[0] = (
                    event_id,
                    event_samples - remaining,
                )
                remaining = 0


def decode_pcm_s16le_base64(
    audio_base64: str,
) -> np.ndarray:
    """
    Decode base64 PCM signed 16-bit little-endian mono audio.
    """

    import base64

    raw = base64.b64decode(audio_base64)

    if len(raw) % 2 != 0:
        raise ValueError(
            "PCM payload contains an odd number of bytes"
        )

    return np.frombuffer(
        raw,
        dtype="<i2",
    ).copy()
