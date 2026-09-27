# Smart Cradle — Complete Codebase Handover / Ready Reckoner

**Repository:** `nannethraa-stack/MaathavamInnolvationLabs-SmartCradleLaunch`  
**Default branch:** `main`  
**Document purpose:** factual implementation reference for an engineer taking over the repository.

This document describes the code that exists in `main`. It distinguishes implemented behavior from pending work and records known discrepancies. It does not claim hardware, cloud, security, or clinical validation unless explicitly stated.

---

## 1. System purpose

The repository contains the software for a Smart Cradle consisting of:

1. Portenta H7 embedded firmware.
2. MQTT device-to-backend messaging.
3. Node.js/Express backend.
4. SQLite persistence.
5. Browser caregiver dashboard.
6. Browser engineering/admin observatory.
7. QR/mobile sharing.
8. Live rolling audio-window processing.
9. Python FastAPI/PyTorch cry-pattern ML service.
10. Bootstrap ML training/evaluation utilities.
11. Real Smart Cradle cry/non-cry data collection and human-review infrastructure.

The current repository is **not a completed production clinical ML system**. The current model is explicitly bootstrap/research-only.

---

## 2. Architecture

```
Portenta H7 firmware
    |
    | MQTT
    v
MQTT broker / Mosquitto
    |
    v
Node.js backend
    |-- MQTT ingestion
    |-- SQLite
    |-- rolling audio buffer
    |-- cry-model HTTP adapter
    |-- REST API
    |-- WebSocket
    |-- caregiver dashboard
    |-- admin observatory
    `-- QR sharing
            |
            | HTTP POST /infer
            v
Python FastAPI ML service :8001
            |
            v
PyTorch CNN checkpoint
```

The live MQTT rolling-window implementation is in Node.js. `ml/rolling_buffer.py` is a Python utility and is not the MQTT ingestion component.

---

## 3. Repository structure

Important repository paths:

```
.
├── .gitignore
├── DEPLOYMENT.md
├── TECH_STACK.md
├── Smart Cradle AI System – Technical Handover Document.pdf
├── backend/
│   ├── README.md
│   ├── package.json
│   ├── package-lock.json
│   ├── server.js
│   ├── mqtt-client.js
│   ├── mqtt-watch.js
│   ├── rolling-audio-buffer.js
│   ├── cry-model.js
│   ├── db.js
│   ├── monitoring.js
│   ├── inspect-db.js
│   ├── inspect-device-status.js
│   ├── deploy.sh
│   └── public/
│       ├── index.html
│       ├── mobile.html
│       └── admin/
│           └── index.html
├── firmware/
│   ├── smart_cradle_firmware/
│   │   └── smart_cradle_firmware.ino
│   └── hardware test sketches
└── ml/
    ├── create_grouped_cv.py
    ├── create_source_split.py
    ├── create_stratified_split.py
    ├── extract_logmel.py
    ├── preprocess_audio.py
    ├── train_baseline.py
    ├── train_cnn.py
    ├── inference.py
    ├── inference_service.py
    ├── rolling_buffer.py
    ├── requirements.txt
    ├── models/
    │   └── donateacry_cnn_bootstrap.json
    └── reports/
        ├── baseline_grouped_cv.txt
        └── cnn_grouped_cv.txt
```

The firmware directory contains additional hardware test sketches for camera, HX711, MLX90640, MQ-137, PDM microphone and Wi-Fi work.

---

## 4. Git exclusions and artifacts

The root `.gitignore` excludes:

```
node_modules/
.env
.env.*
*.db
*.db-shm
*.db-wal
__pycache__/
*.py[cod]
.venv/
venv/
ml/data/
ml/datasets/
*.tflite
*.onnx
*.h5
*.keras
*.pt
*.pth
models/generated/
tmp/
temp/
*.tmp
```

Therefore:
- runtime SQLite databases are not versioned;
- datasets are not versioned;
- PyTorch checkpoints are not versioned;
- the bootstrap `.pt` file must be provisioned separately;
- bootstrap JSON metadata and reports are versioned.

---

# 5. Firmware

## 5.1 Main file

`firmware/smart_cradle_firmware/smart_cradle_firmware.ino`

Current firmware version constant:

`fw-2.3.1`

The firmware is C++ for Arduino Portenta H7.

## 5.2 Hardware represented by the firmware

- Arduino Portenta H7
- Portenta Vision Shield
- OV7675 camera
- MLX90640 thermal sensor
- HX711 + 5 kg load cell
- MQ-137 ammonia sensor
- PDM digital microphone
- Wi-Fi
- NTP/UDP time

## 5.3 Presence

Weight is used for cradle presence.

Configured thresholds:
- `>= 0.80 kg`: baby present
- `<= 0.50 kg`: baby absent

Debounce values:
- approximately 3 seconds for present;
- approximately 10 seconds for absent.

MLX90640 thermal information is also used as a fallback signal in presence logic.

## 5.4 MLX90640

- I2C address: `0x33`
- refresh rate: 8 Hz
- center-region temperature is calculated from the frame
- sensor failures and recovery generate alerts.

## 5.5 Respiratory rate

Thermal data is processed over a 15-second window and reported with a respiratory confidence value.

## 5.6 MQ-137

Implemented behavior:
- 25-second heater warm-up;
- 10-sample averaging;
- non-blocking sampling intervals;
- ADC fault detection near 0/4095;
- failure/recovery alerts.

The source currently has `MQ137_CURVE_A=0` and `MQ137_CURVE_B=0`, and the code treats ammonia PPM as invalid while `MQ137_CALIBRATED` is false. This is not a calibrated clinical ammonia measurement.

## 5.7 Camera

Configured:
- QQVGA;
- grayscale;
- 160 x 120.

Current firmware initializes the OV7675 and captures a test frame. Continuous vision inference is not implemented in this firmware version.

## 5.8 Audio

PDM microphone:
- 16 kHz;
- mono.

Edge RMS heuristic:
- RMS threshold: 1500;
- crying duration threshold: 3000 ms.

This RMS detector is an edge heuristic, not the final trained cry classifier.

---

# 6. Firmware MQTT

Topic pattern:

`cradle/{device_uuid}/...`

Topics:
- `cradle/{device_uuid}/telemetry`
- `cradle/{device_uuid}/status`
- `cradle/{device_uuid}/alerts`
- `cradle/{device_uuid}/images`
- `cradle/{device_uuid}/audio`

Configured operational intervals include:
- telemetry: 5 s;
- status: 30 s;
- image: 60 s;
- MQTT retry: 5 s;
- Wi-Fi retry: 10 s;
- NTP retry: 5 min.

MQTT username/password support exists in firmware. The source contains placeholder credentials which must be replaced for deployment.

---

# 7. Real audio data collection

This is the current data-collection implementation.

Firmware constant:

`AUDIO_TRAINING_CAPTURE_ENABLED = true`

When the baby is present, `publishAudioEvent()` publishes microphone buffers to the audio MQTT topic.

The audio payload records:
- 16 kHz;
- 1 channel;
- PCM S16LE;
- base64 data;
- `training_eligible=true`;
- `audio_class_hint`.

The hint is:
- `CRY_CANDIDATE` when the edge RMS detector says crying;
- `NON_CRY_CANDIDATE` otherwise.

**These hints are not ground truth.**

They exist to describe how the recording was captured. The final training label must come from human review.

This means the application now captures both potential cry and potential non-cry audio while a baby is present.

---

# 8. Node.js backend

## 8.1 Dependencies

From `backend/package.json`:

- Express 4.18.x
- mqtt 5.3.x
- better-sqlite3 9.4.x
- ws 8.16.x
- nodemon 3.x for development

Backend runtime documented as Node.js 18+.

Start:

```
cd backend
npm install
npm start
```

Development:

```
npm run dev
```

Port:

`process.env.PORT || 3001`

---

# 9. backend/server.js

This is the HTTP application entry point.

Responsibilities:
- Express setup;
- static browser assets;
- public API;
- admin API;
- QR sharing API;
- WebSocket server;
- MQTT startup;
- broadcasting live backend events.

## Public routes

```
GET /api/telemetry
GET /api/telemetry/history
GET /api/images
GET /api/images/:id
GET /api/alerts
GET /api/presence
```

## Admin routes

```
GET  /admin/health
GET  /admin/devices
GET  /admin/devices/:deviceId
GET  /admin/anomalies
GET  /admin/anomalies/critical
GET  /admin/telemetry
GET  /admin/samples
GET  /admin/events
GET  /admin/audio
GET  /admin/audio/:id
GET  /admin/cry-inferences
GET  /admin/cry-reviews
POST /admin/cry-reviews
GET  /admin/training/stats
GET  /admin/stats
```

## Sharing routes

```
POST /api/share
GET  /api/shared/snapshot
POST /api/share/revoke
```

Admin authentication/authorization is not implemented in the current source. This is a required security task before sensitive/clinical deployment.

---

# 10. backend/mqtt-client.js

The module:
1. connects to the configured MQTT broker;
2. subscribes to `cradle/#`;
3. parses JSON;
4. routes by topic channel;
5. persists data through `db.js`;
6. feeds audio into the rolling buffer;
7. invokes `cry-model.js` for complete windows;
8. stores inference results;
9. broadcasts events.

Default broker:

`process.env.MQTT_BROKER || 'mqtt://broker.hivemq.com:1883'`

The public HiveMQ endpoint is documented as development/testing only.

Production deployment is intended to use authenticated Mosquitto.

Presence-change processing resets the per-device rolling audio buffer so a 3-second ML window does not cross a baby-present/baby-absent boundary.

---

# 11. backend/rolling-audio-buffer.js

This is the **live** rolling-window implementation.

Constants:
- sample rate: 16,000 Hz
- window: 48,000 samples = 3 seconds
- hop: 16,000 samples = 1 second

Incoming MQTT packet boundaries do not define ML windows.

Behavior:
- accepts arbitrary PCM S16LE chunks;
- accumulates them by sample count;
- creates exact 48,000-sample windows;
- advances by 16,000 samples;
- retains the overlapping 2 seconds;
- records source event IDs;
- records window start/end;
- creates per-device window IDs.

Window ID format:

`{device_id}-rolling-{window_index}`

---

# 12. backend/cry-model.js

This is the Node.js-to-Python ML adapter.

Default endpoint:

`http://127.0.0.1:8001/infer`

Override with:

`CRY_MODEL_URL`

The adapter sends:
- `audio_base64`;
- `pcm_s16le_base64`;
- sample rate;
- device ID;
- event ID;
- window ID.

It maps model results into the backend `cry_inferences` structure.

On ML-service failure it records:
- `status=inference_error`;
- `decision=REVIEW`;
- `needs_review=true`;
- uncertainty reason `inference_error`.

It does not invent an inference result.

---

# 13. SQLite database

File:

`backend/cradle.db`

Created automatically by `db.js`.

SQLite WAL mode is enabled.

## Tables

### telemetry
Sensor/telemetry history including:
- temperature;
- respiratory rate/confidence;
- crying/cry detection;
- cry probability/pattern/model;
- audio event ID;
- ammonia;
- diaper state/fusion fields;
- baby weight/height;
- quality;
- validation status.

### alerts
Sensor failure/recovery and other alert records.

### presence_changes
Baby present/absent state transitions.

### images
Base64 camera frames and metadata.

### device_status
Status/heartbeat records.

### sensor_samples
Individual sensor samples including weight, height, temperature, respiratory rate/confidence, ammonia, cry probability and diaper-weight delta.

### audio_events
Original MQTT audio chunks with:
- event ID;
- device;
- timestamp;
- event type;
- sample rate;
- channels;
- sample count;
- format;
- model version;
- training eligibility;
- capture class hint;
- base64 audio.

### cry_inferences
ML results including:
- audio event;
- device/time;
- model status/version;
- probable pattern/probability;
- latency;
- features;
- error;
- window ID;
- window start/end;
- sample count;
- source event IDs;
- probabilities;
- decision;
- needs-review;
- uncertainty reasons;
- service latency.

### cry_reviews
Human review:
- audio event;
- device;
- review status;
- human decision;
- human pattern;
- evidence note;
- training status;
- reviewer;
- review timestamp;
- window ID.

### system_events
Backend/system operational events.

### share_tokens
Hashed QR/share token, device, expiry and revoked state.

---

# 14. Human review contract

Supported decisions:

```
CRY
NON_CRY
UNCERTAIN
```

Legacy decisions also remain accepted:

```
CONFIRM
CORRECT
UNABLE_TO_DETERMINE
```

Supported cry patterns:

```
HUNGER
PAIN
DISCOMFORT
TIRED
BURPING
OTHER
```

Training statuses:

```
CANDIDATE
APPROVED
EXCLUDED
```

Rules:
- CRY requires a pattern;
- NON_CRY must not have a pattern;
- UNCERTAIN is not ground-truth positive/negative data;
- AI predictions are not ground truth.

---

# 15. Automatic review candidate creation

When a complete ML window is recorded, `insertCryInference()` creates a review candidate when a usable window ID and source audio event are available.

Initial values:
- `review_status=NOT_REVIEWED`;
- `training_status=CANDIDATE`;
- human label unset.

This ensures collected windows enter the human-review workflow without treating the model prediction as truth.

---

# 16. Training statistics

Endpoint:

`GET /admin/training/stats`

It groups human-review records and reports counts for:
- CRY;
- NON_CRY;
- UNCERTAIN;
- APPROVED;
- EXCLUDED;
- classification/pattern/training-status combinations.

This is the current visibility mechanism for accumulated training candidates.

---

# 17. Browser applications

## `backend/public/index.html`

Caregiver dashboard.

Uses:
- vanilla HTML/CSS/JavaScript;
- Chart.js;
- WebSocket;
- responsive CSS.

It is intended to show operational monitoring rather than engineering-only ML diagnostics.

## `backend/public/mobile.html`

Mobile QR/share view.

## `backend/public/admin/index.html`

Engineering/admin observatory.

It displays:
- inference history;
- audio review queue;
- retained audio events;
- sensor diagnostics;
- health;
- anomalies;
- engineering data.

The audio review UI now supports CRY/NON_CRY/UNCERTAIN.

---

# 18. Monitoring

`backend/monitoring.js` supplies health/device/anomaly monitoring used by admin routes.

Operational records can be stored in `system_events`.

---

# 19. QR sharing

The backend supports:
- creating share tokens;
- hashing token values for storage;
- expiry;
- revocation;
- retrieving a shared snapshot.

This is a sharing mechanism, not complete application authentication.

---

# 20. Python ML service

Files:
- `ml/inference.py`
- `ml/inference_service.py`
- `ml/rolling_buffer.py`

FastAPI service:
- host: 127.0.0.1
- port: 8001

Endpoints:

```
GET  /health
POST /infer
```

## /infer input contract

Requires:
- sample rate exactly 16000;
- format `pcm_s16le_base64`;
- valid base64;
- even PCM byte length;
- exactly 48,000 samples.

The service returns HTTP 400 for invalid audio contract data.

If the checkpoint is absent:
- service remains available;
- health reports model unavailable;
- inference returns HTTP 503.

---

# 21. CNN architecture

Current `CryCNN`:

```
Conv2D 1 -> 16
ReLU
MaxPool

Conv2D 16 -> 32
ReLU
MaxPool

Conv2D 32 -> 64
ReLU

AdaptiveAvgPool2D(1,1)

Linear 64 -> num_classes
```

Input feature representation:
- 3-second audio;
- 16 kHz;
- 48,000 samples;
- log-mel spectrogram;
- 64 mel bins;
- n_fft=1024;
- hop_length=256;
- current feature shape 64 x 188.

---

# 22. Production cry-pattern contract

The intended production pattern set is:

```
HUNGER
PAIN
DISCOMFORT
TIRED
BURPING
OTHER
```

The current bootstrap model does not contain all of these outputs.

Bootstrap label mapping:

```
hungry      -> HUNGER
belly_pain  -> PAIN
discomfort  -> DISCOMFORT
tired       -> TIRED
burping     -> BURPING
```

There is no trained bootstrap NON_CRY output and no trained bootstrap OTHER output.

---

# 23. Current bootstrap model

Metadata file:

`ml/models/donateacry_cnn_bootstrap.json`

Model version:

`donateacry-cnn-bootstrap-v1`

Status:

`BOOTSTRAP_NOT_PRODUCTION`

Dataset:
- Donate-a-Cry;
- 2,035 windows;
- 221 source recordings;
- 5 folds.

Classes:
- belly_pain;
- burping;
- discomfort;
- hungry;
- tired.

Recorded CNN 5-fold source-grouped Macro-F1:
- fold 1: 0.1213
- fold 2: 0.0293
- fold 3: 0.1692
- fold 4: 0.1743
- fold 5: 0.1885
- mean: 0.1365
- standard deviation: 0.0582

The report explicitly identifies the experiment as not production.

---

# 24. Why the bootstrap model is not production

The current model lacks:
- a trained NON_CRY detector;
- a trained OTHER class;
- real Smart Cradle production recordings;
- human-curated Smart Cradle labels;
- the final production train/validation/test split;
- uncertainty calibration.

The inference layer intentionally keeps the bootstrap status non-production and uses REVIEW behavior rather than fabricating missing model capabilities.

---

# 25. ML preprocessing and training scripts

## create_grouped_cv.py

Input:
`ml/data/processed/donateacry_manifest.csv`

Uses:
`GroupKFold(n_splits=5)`

Group:
`source_id`

Output:
`donateacry_grouped_cv.csv`

Purpose: keep source recordings together during CV.

## create_source_split.py

Input:
`donateacry_manifest.csv`

Uses:
- seed 42;
- 70% source groups train;
- 15% validation;
- 15% test.

Source IDs are shuffled before assignment.

## create_stratified_split.py

Uses `StratifiedGroupKFold` while keeping source IDs indivisible.

It creates an approximate train/validation/test split from source groups. Its source stratification uses one representative sorted class label per source.

## preprocess_audio.py

Configured:
- 16 kHz;
- 3-second window;
- 1-second hop.

Its current main path prints a dry-run message rather than executing preprocessing.

## extract_logmel.py

Input:
`donateacry_grouped_cv.csv`

Outputs:
- `donateacry_logmel.npz`
- `donateacry_logmel_metadata.csv`

Metadata:
- source_id;
- class;
- fold;
- source_file;
- window_index.

## train_baseline.py

RandomForest baseline:
- 300 trees;
- random_state 42;
- balanced class weighting.

Features:
- 64 mel means;
- 64 mel standard deviations.

Recorded mean Macro-F1:
0.1822.

This is a baseline experiment, not a production model.

## train_cnn.py

CNN bootstrap training:
- 15 epochs;
- batch size 32;
- learning rate 0.001;
- seed 42;
- Adam;
- class-balanced CrossEntropyLoss.

It performs 5-fold source-grouped CV and then trains a final bootstrap checkpoint on all available Donate-a-Cry windows.

---

# 26. Python rolling_buffer.py

This utility implements the same fundamental audio contract:
- 16 kHz;
- 48,000-sample window;
- 16,000-sample hop.

It can:
- accept int16 NumPy chunks;
- track source event IDs;
- generate overlapping windows;
- decode base64 S16LE audio.

Again: the live MQTT rolling path is the Node implementation in `backend/rolling-audio-buffer.js`.

---

# 27. ML dependencies

Current `ml/requirements.txt`:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
numpy==2.2.1
librosa==0.10.2.post1
torch==2.5.1
```

A previously inspected development environment used a different PyTorch version and CPU execution. That historical environment is not a repository guarantee. Use the checked-in requirements for a clean ML environment and validate compatibility.

---

# 28. AWS deployment

`backend/deploy.sh` and `DEPLOYMENT.md` describe an EC2 deployment using:
- Ubuntu 22.04;
- Node.js 18;
- Mosquitto;
- PM2;
- backend port 3001;
- MQTT port 1883;
- application path `/opt/smart-cradle`.

PM2 process name:
`smart-cradle`

Configured PM2 memory restart:
512 MB.

Production MQTT intent:
- username/password;
- anonymous access disabled by `deploy.sh`;
- restrict port 1883 to device networks where possible.

The repository does not implement a Docker/Kubernetes/ECS/Lambda deployment.

---

# 29. Deployment-document discrepancy

There is a known inconsistency in `DEPLOYMENT.md`.

The current `backend/deploy.sh` uses authenticated Mosquitto and disables anonymous access.

An older Step 5 example in `DEPLOYMENT.md` still shows:

```
allow_anonymous true
```

That example must not be used for production. The deployment script/security section is the current intended secured configuration.

---

# 30. Security state

Implemented:
- MQTT username/password support;
- secured Mosquitto deployment script;
- hashed share tokens;
- share expiry/revocation;
- local SQLite persistence.

Not implemented as a complete production security layer:
- application authentication/authorization for admin endpoints;
- HTTPS as part of the application deployment;
- complete retention/access-control policy for stored audio;
- full secrets-management workflow;
- complete backup/recovery policy.

Do not describe the current repository as clinically validated.

---

# 31. Data flow

Sensor data:

```
Sensors
  -> Portenta H7
  -> MQTT
  -> Node backend
  -> SQLite
  -> REST/WebSocket
  -> dashboards
```

Audio data:

```
PDM microphone
  -> firmware audio buffer
  -> MQTT audio
  -> audio_events
  -> Node rolling buffer
  -> 3-second window
  -> ML service
  -> cry_inferences
  -> human review
  -> cry_reviews
```

Training-data lifecycle now supported:

```
Real audio
  -> candidate
  -> human review
      -> CRY + pattern
      -> NON_CRY
      -> UNCERTAIN
  -> APPROVED / EXCLUDED
  -> future curated training dataset
```

---

# 32. What is implemented

- Portenta H7 firmware.
- Wi-Fi.
- MQTT.
- MLX90640.
- HX711/load-cell presence.
- MQ-137 acquisition/warm-up/fault handling.
- PDM audio.
- edge RMS cry heuristic.
- OV7675 initialization/test capture.
- MQTT backend ingestion.
- SQLite persistence.
- caregiver dashboard.
- WebSocket updates.
- admin observatory.
- QR sharing.
- rolling 3-second / 1-second-hop audio processing.
- Python FastAPI ML service.
- bootstrap PyTorch CNN.
- uncertainty/review fields.
- CRY/NON_CRY/UNCERTAIN human review.
- automatic review candidates for completed ML windows.
- training statistics endpoint.
- authenticated MQTT deployment configuration.
- continuous candidate capture for both cry and non-cry audio while baby is present.

---

# 33. Not yet production-complete

- trained NON_CRY detector/classifier;
- trained OTHER class;
- complete six-class production model;
- complete curated-dataset generator;
- final training manifest;
- final production train/validation/test split pipeline;
- production retraining automation;
- full model registry;
- calibrated production uncertainty thresholds;
- production checkpoint in Git;
- complete audio retention lifecycle;
- admin authentication/authorization;
- HTTPS deployment integration;
- clinical validation.

---

# 34. Critical ML rules for future maintainers

1. Do not use the RMS edge detector as ground truth.
2. Do not use bootstrap CNN predictions as ground truth.
3. Human review is the current source of truth.
4. Do not assign a cry pattern to NON_CRY.
5. Do not silently turn UNCERTAIN into CRY or NON_CRY.
6. Preserve source event IDs and window IDs.
7. Keep MQTT packet boundaries separate from ML window boundaries.
8. Keep the 3-second/48,000-sample/16-kHz contract.
9. Keep the 1-second hop.
10. Prevent correlated recordings/windows from leaking across training and test sets.
11. Store dataset/model provenance with production models.
12. Never change a model status to production solely because inference executes.

---

# 35. Exact next ML phase

Once sufficient real Smart Cradle recordings have been collected and human-reviewed, implement:

```
APPROVED cry + non-cry windows
            |
            v
curated dataset
            |
            v
training manifest
            |
            v
source/device/baby-grouped split
            |
            +--> train
            +--> validation
            `--> test
            |
            v
production training
            |
            v
held-out evaluation
            |
            v
uncertainty calibration
            |
            v
model registry
            |
            v
validated production checkpoint
            |
            v
ML service deployment
```

The exact grouping key for the final production dataset must be based on the identity/provenance metadata actually available from the collected Smart Cradle recordings. Do not invent a grouping key after collection has lost that information.

---

# 36. Recommended takeover reading order

1. This document.
2. `TECH_STACK.md`.
3. `DEPLOYMENT.md`.
4. `backend/server.js`.
5. `backend/mqtt-client.js`.
6. `backend/db.js`.
7. `backend/rolling-audio-buffer.js`.
8. `backend/cry-model.js`.
9. `ml/inference_service.py`.
10. `ml/inference.py`.
11. `ml/rolling_buffer.py`.
12. `firmware/smart_cradle_firmware/smart_cradle_firmware.ino`.
13. `ml/models/donateacry_cnn_bootstrap.json`.
14. `ml/reports/cnn_grouped_cv.txt`.
15. `ml/reports/baseline_grouped_cv.txt`.
16. `ml/train_cnn.py`.

Then inspect the hardware test sketches relevant to the hardware being commissioned.

---

# 37. Local startup

Backend:

```
cd backend
npm install
npm start
```

Dashboard:

`http://localhost:3001`

ML service:

```
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn inference_service:app --host 127.0.0.1 --port 8001
```

A checkpoint must be provisioned separately because `*.pt` is excluded from Git.

---

# 38. Current source-of-truth rules

For implementation behavior, use source code over stale prose:

- Backend port: `backend/server.js`
- MQTT: `backend/mqtt-client.js`
- DB schema/migrations: `backend/db.js`
- live rolling window: `backend/rolling-audio-buffer.js`
- Node ML adapter: `backend/cry-model.js`
- Python inference contract: `ml/inference_service.py`
- model architecture/status: `ml/inference.py` + model metadata
- hardware behavior: `firmware/smart_cradle_firmware/smart_cradle_firmware.ino`
- deployment automation: `backend/deploy.sh`

If documentation conflicts with source, verify the source before changing or deploying the system.

---

# 39. Validation status

The modified backend JavaScript files were syntax-checked after the recent audio/review changes:
- `backend/db.js`: syntax OK;
- `backend/server.js`: syntax OK;
- `backend/mqtt-client.js`: syntax OK.

The firmware changes have not been represented here as a claim of successful physical Portenta compilation/flashing or hardware/MQTT end-to-end validation. Those require the actual hardware and deployment environment.

---

# 40. Final state

The repository is currently a **Smart Cradle telemetry, monitoring, audio-data collection, human-review and bootstrap-ML platform**.

The current strategic boundary is:

```
CURRENT
Real Smart Cradle data collection
        +
Human CRY/NON_CRY review
        +
Bootstrap ML for experimental inference

NEXT
Curated production dataset
        ->
training manifest
        ->
leakage-safe split
        ->
production training
        ->
held-out evaluation
        ->
uncertainty calibration
        ->
model registry
        ->
validated production model
```

A maintainer should preserve this separation. The bootstrap model is not the final production model, and collected audio should be curated through human review before it becomes training ground truth.
