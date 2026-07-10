# Sign Language FE

React/Vite frontend + FastAPI backend for testing sign-language recognition with MediaPipe landmarks and the local `v2` PyTorch model.

## Current Model

- Checkpoint: `backend/models/v2/final_trainval_model.pt`
- Config: `backend/models/v2/config.json`
- Input shape: `[60, 291]` or `[batch, 60, 291]`
- Features: demo-compatible preprocessed world pose + world hands only
- Face and mouth features are not used by the v2 model

Feature dimension:

```text
pose world: 33 landmarks * 5 values = 165
hands world: 2 hands * 21 landmarks * 3 values = 126
total = 291
```

## Requirements

- Node.js 18+ and npm
- Python 3.11 or 3.12 recommended
- Python packages from `backend/requirements.txt`
- A webcam or a local video file for testing

PyTorch wheels may not be available for every Python version. If `torch` does not install or the backend reports that PyTorch is missing, create a fresh Python 3.11/3.12 environment.

## Setup

Install frontend dependencies:

```powershell
cd path\to\Sign-Language-FE
npm install
```

Create and install backend dependencies:

```powershell
cd path\to\Sign-Language-FE\backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.11` is not available, install Python 3.11 or 3.12 first, then rerun the commands above.

## Run Backend

Open terminal 1:

```powershell
cd path\to\Sign-Language-FE\backend
.\.venv\Scripts\Activate.ps1
python run_server.py
```

Expected output:

```text
Python executable: ...\backend\.venv\Scripts\python.exe
PyTorch: ...
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open.

Check backend health from another terminal:

```powershell
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

Expected:

```json
{"status":"ok","checkpoint_exists":true,"torch_available":true}
```

## Run Frontend

Open terminal 2:

```powershell
cd path\to\Sign-Language-FE
npm run dev -- --host 127.0.0.1
```

Open the Vite URL, usually:

```text
http://127.0.0.1:5173/
```

## Test Flow

1. Start the backend first.
2. Start the frontend second.
3. Open the app in the browser.
4. Click `Start Camera` or `Upload Video`.
5. Confirm that hand/pose overlays are visible.
6. Watch `Frame buffer` in the prediction panel.
7. It should count from `0/60` to `60/60`.
8. When 60 processed frames are ready, the frontend calls `POST /model/predict`.
9. If the backend is healthy, the prediction label should update.

Uploaded videos autoplay and loop to make testing easier.

## Build

```powershell
cd path\to\Sign-Language-FE
npm run build
```

## Backend API

- `GET /health`: checks backend, checkpoint, and PyTorch availability.
- `GET /model/metadata`: returns model metadata.
- `POST /model/predict`: runs inference for `[60, 291]` or `[batch, 60, 291]`.
- `POST /model/benchmark`: measures synthetic inference latency.

Synthetic predict test:

```powershell
python -c "import json, urllib.request; data=json.dumps({'sequence': [[[0]*291]*60], 'top_k': 3}).encode(); req=urllib.request.Request('http://127.0.0.1:8000/model/predict', data=data, headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req).read().decode())"
```

## Troubleshooting

### `POST /model/predict 503`

The backend is running, but the Python environment does not have PyTorch.

Fix:

```powershell
cd path\to\Sign-Language-FE\backend
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__)"
```

If this fails, reinstall backend dependencies in a Python 3.11/3.12 virtual environment:

```powershell
python -m pip install -r requirements.txt
```

### `ERR_CONNECTION_REFUSED`

The frontend cannot reach the backend. Make sure this is running:

```text
http://127.0.0.1:8000
```

Start it with:

```powershell
cd path\to\Sign-Language-FE\backend
.\.venv\Scripts\Activate.ps1
python run_server.py
```

### `ERR_NO_BUFFER_SPACE`

This usually happens after many repeated failed backend requests. Fix the backend error first, then reload the browser page.

### `Frame buffer` does not increase

Check:

- The camera or uploaded video is actually playing.
- MediaPipe status shows hand/pose counts.
- Browser camera permission is allowed.
- For uploaded video, press play manually if autoplay is blocked.

### `Frame buffer` reaches 60 but the label does not change

Open DevTools -> Network -> filter by `predict`.

- `200`: check the response body.
- `503`: backend Python environment is missing PyTorch.
- No request: recognition is not active or frames are not reaching the prediction hook.

### MediaPipe `landmark_projection_calculator` warning

This warning is usually safe to ignore if hand/pose overlays are visible.

### `favicon.ico 404`

This does not affect recognition.

## Notes

The v2 model was trained with `seq_len=60`. The frontend keeps a live 90-frame window, starts prediction after 30 frames, trims inactive frames, resamples to 60 frames, normalizes pose/hands like the Python demo extractor, and then sends `[60, 291]` to the backend.

60 processed frames are not always 2 seconds. Actual wait time depends on MediaPipe pipeline FPS:

```text
wait time = 60 / completed_pipeline_fps
```

Examples:

- 30 FPS -> about 2 seconds
- 20 FPS -> about 3 seconds
- 10 FPS -> about 6 seconds
