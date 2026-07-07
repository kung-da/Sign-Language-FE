# Backend Model Test API

FastAPI backend for testing the trained `models/v2/final_trainval_model.pt` model.

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
- `GET /model/metadata`: returns `config.json`, `history.csv` summary, `train_summary.json`, and checkpoint size.
- `POST /model/predict`: runs inference for v2 world pose+hands keypoints shaped `[60, 291]` or `[batch, 60, 291]`.
- `POST /model/benchmark`: measures model inference latency with synthetic keypoints.

Example predict payload shape:

```json
{
  "sequence": "[60 frames x 291 floats]",
  "top_k": 5
}
```

Replace `sequence` with a real 60-frame keypoint sequence where each frame has 291 values.

## Notes

- The v2 model uses only world pose and hand landmarks. Face and normalized landmark columns are not part of the backend input.
- The Transformer class is reconstructed from `config.json` and checkpoint tensor names/shapes.
