# Backend Model Test API

FastAPI backend for testing the trained `asl_tcn_baseline/best.pt` model.

## Run

Create a Python 3.11 or 3.12 environment, then install dependencies:

```bash
pip install -r backend/requirements.txt
```

Start the API from the `backend` folder:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Endpoints

- `GET /health`: checks checkpoint and PyTorch availability.
- `GET /model/metadata`: returns `config.json`, `history.json` summary, `latency.json`, and checkpoint size.
- `POST /model/predict`: runs inference for keypoints shaped `[60, 634]` or `[batch, 60, 634]`.
- `POST /model/benchmark`: measures model inference latency with synthetic keypoints.

Example predict payload shape:

```json
{
  "sequence": "[60 frames x 634 floats]",
  "top_k": 5
}
```

Replace `sequence` with a real 60-frame keypoint sequence where each frame has 634 values.

## Notes

- The current local backend venv uses Python 3.14 and does not have PyTorch installed, so `/model/predict` returns HTTP 503 until a PyTorch-compatible env is used.
- The TCN class is reconstructed from `config.json` and checkpoint tensor names/shapes. If the original training source is available, use that exact model class for the strongest reproducibility.
