from __future__ import annotations

import base64
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import CryInference


app = FastAPI(
    title="Smart Cradle Cry Pattern ML Service",
    version="0.1.0",
)

try:
    MODEL = CryInference()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    # Keep the service available for health/diagnostics when the model artifact
    # is intentionally absent from Git. Inference returns 503 until provisioned.
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)


class InferenceRequest(BaseModel):
    audio_base64: str
    format: str
    sample_rate: int
    device_id: str | None = None
    event_id: str | None = None
    window_id: str | None = None


def is_production_ready() -> bool:
    return MODEL is not None and MODEL.model_status == "PRODUCTION"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service": "smart-cradle-cry-pattern-ml",
        "status": "ok",
        "model_version": MODEL.model_version if MODEL else None,
        "model_status": MODEL.model_status if MODEL else "UNAVAILABLE",
        "production_ready": is_production_ready(),
        "model_load_error": MODEL_LOAD_ERROR,
        "window_seconds": 3,
        "hop_seconds": 1,
        "sample_rate": 16000,
    }


@app.post("/infer")
def infer(request: InferenceRequest) -> dict[str, Any]:
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="ML model artifact is not provisioned on this host.",
        )

    if request.sample_rate != 16000:
        raise HTTPException(
            status_code=400,
            detail="Only 16000 Hz audio is currently supported.",
        )

    if request.format.lower() != "pcm_s16le_base64":
        raise HTTPException(
            status_code=400,
            detail="Expected format pcm_s16le_base64.",
        )

    try:
        raw = base64.b64decode(
            request.audio_base64,
            validate=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 audio: {exc}",
        )

    if len(raw) % 2 != 0:
        raise HTTPException(
            status_code=400,
            detail="PCM payload contains an odd number of bytes.",
        )

    audio = np.frombuffer(
        raw,
        dtype="<i2",
    ).copy()

    if audio.size != 48000:
        raise HTTPException(
            status_code=400,
            detail=(
                "Expected exactly 48000 samples for a 3-second "
                f"window; received {audio.size}."
            ),
        )

    started = time.perf_counter()

    try:
        result = MODEL.predict_audio_window(audio)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    elapsed_ms = int(
        (time.perf_counter() - started) * 1000
    )

    result["service_inference_ms"] = elapsed_ms
    result["device_id"] = request.device_id
    result["event_id"] = request.event_id
    result["window_id"] = request.window_id

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inference_service:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )
