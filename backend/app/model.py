from __future__ import annotations

import json
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - depends on local runtime
    torch = None
    nn = None


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models" / "v2"
CHECKPOINT_PATH = MODEL_DIR / "final_trainval_model.pt"
CONFIG_PATH = MODEL_DIR / "config.json"
HISTORY_PATH = MODEL_DIR / "history.csv"
SUMMARY_PATH = MODEL_DIR / "train_summary.json"


class ModelUnavailableError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_history_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"epochs_recorded": None, "best_epoch_by_val_acc": None, "best_val_acc": None}

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    best_epoch = None
    best_val_acc = None
    for row in rows:
        try:
            val_acc = float(row.get("val_acc", "nan"))
        except ValueError:
            continue
        if not np.isnan(val_acc) and (best_val_acc is None or val_acc > best_val_acc):
            best_val_acc = val_acc
            best_epoch = row.get("epoch")

    return {
        "best_epoch_by_val_acc": best_epoch,
        "best_val_acc": best_val_acc,
        "epochs_recorded": len(rows),
    }


def load_model_metadata() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    summary = read_json(SUMMARY_PATH)
    checkpoint_size_mb = (
        CHECKPOINT_PATH.stat().st_size / (1024 * 1024) if CHECKPOINT_PATH.exists() else None
    )

    return {
        "model_dir": str(MODEL_DIR),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "checkpoint_size_mb": checkpoint_size_mb,
        "config": config,
        "train_summary": summary,
        "history_summary": read_history_summary(HISTORY_PATH),
        "runtime": {
            "torch_available": torch is not None,
            "standardization_available": True,
            "labels_available": True,
        },
    }


if nn is not None:

    class TransformerSequenceClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            d_model: int,
            num_heads: int,
            num_layers: int,
            dim_feedforward: int,
            seq_len: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
            self.input_proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            x = self.input_proj(x)
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)
            x = x + self.pos_embedding[:, : x.shape[1], :]
            x = self.encoder(x)
            x = self.norm(x[:, 0])
            return self.classifier(x)


@dataclass
class PredictionResult:
    top_k: list[list[dict[str, float | int | str]]]
    inference_ms: float
    batch_size: int


class V2Predictor:
    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:
            raise ModelUnavailableError(
                "PyTorch is not installed. Install torch in a Python 3.11/3.12 backend env "
                "to run v2 inference."
            )
        if not CHECKPOINT_PATH.exists():
            raise ModelUnavailableError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        self.config = read_json(CONFIG_PATH)
        self.seq_len = int(self.config.get("seq_len", 60))
        self.input_dim = int(self.config.get("input_dim", 291))
        self.num_classes = int(self.config.get("num_classes", 2000))
        self.device = torch.device(device)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        self.id_to_label = self._load_id_to_label(checkpoint.get("id_to_label", {}))
        self.feature_mean = self._load_feature_stat(checkpoint.get("feature_mean"), default=0.0)
        self.feature_std = self._load_feature_stat(checkpoint.get("feature_std"), default=1.0)

        self.model = TransformerSequenceClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            d_model=int(self.config.get("d_model", 448)),
            num_heads=int(self.config.get("num_heads", 8)),
            num_layers=int(self.config.get("num_layers", 2)),
            dim_feedforward=int(self.config.get("dim_feedforward", 896)),
            seq_len=self.seq_len,
            dropout=float(self.config.get("dropout", 0.5)),
        ).to(self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _load_feature_stat(self, value: Any, default: float) -> torch.Tensor:
        if value is None:
            arr = np.full((self.input_dim,), default, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.input_dim,):
            raise ModelUnavailableError(
                f"feature statistic must have shape [{self.input_dim}], got {list(arr.shape)}"
            )
        if default == 1.0:
            arr = np.where(np.abs(arr) < 1e-8, 1.0, arr)
        return torch.from_numpy(arr).to(self.device).view(1, 1, self.input_dim)

    def _load_id_to_label(self, value: Any) -> dict[int, str]:
        if not isinstance(value, dict):
            return {}
        labels: dict[int, str] = {}
        for key, label in value.items():
            try:
                labels[int(key)] = str(label)
            except (TypeError, ValueError):
                continue
        return labels

    def validate_sequence(self, sequence: Any) -> np.ndarray:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"sequence must have shape [{self.seq_len}, {self.input_dim}] or [batch, {self.seq_len}, {self.input_dim}]")
        expected = (self.seq_len, self.input_dim)
        if tuple(arr.shape[1:]) != expected:
            raise ValueError(f"expected sequence shape [batch, {expected[0]}, {expected[1]}], got {list(arr.shape)}")
        return arr

    def predict(self, sequence: Any, top_k: int = 5) -> PredictionResult:
        arr = self.validate_sequence(sequence)
        top_k = max(1, min(int(top_k), self.num_classes))

        with torch.inference_mode():
            tensor = torch.from_numpy(arr).to(self.device)
            tensor = (tensor - self.feature_mean) / self.feature_std
            start = time.perf_counter()
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            values, indices = torch.topk(probs, k=top_k, dim=1)
            inference_ms = (time.perf_counter() - start) * 1000

        batch_results: list[list[dict[str, float | int | str]]] = []
        for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
            batch_results.append(
                [
                    {
                        "class_index": int(index),
                        "label": self.id_to_label.get(int(index), f"class_{int(index)}"),
                        "probability": float(probability),
                    }
                    for probability, index in zip(row_values, row_indices)
                ]
            )

        return PredictionResult(
            top_k=batch_results,
            inference_ms=inference_ms,
            batch_size=int(arr.shape[0]),
        )

    def benchmark(self, runs: int = 30, warmup: int = 5) -> dict[str, Any]:
        runs = max(1, min(int(runs), 500))
        warmup = max(0, min(int(warmup), 100))
        sample = np.zeros((1, self.seq_len, self.input_dim), dtype=np.float32)

        for _ in range(warmup):
            self.predict(sample, top_k=1)

        timings = [self.predict(sample, top_k=1).inference_ms for _ in range(runs)]
        sorted_timings = sorted(timings)
        p95_index = min(len(sorted_timings) - 1, int(round(0.95 * (len(sorted_timings) - 1))))
        return {
            "runs": runs,
            "warmup": warmup,
            "mean_ms": statistics.fmean(timings),
            "median_ms": statistics.median(timings),
            "p95_ms": sorted_timings[p95_index],
            "min_ms": min(timings),
            "max_ms": max(timings),
        }


_predictor: V2Predictor | None = None


def get_predictor() -> V2Predictor:
    global _predictor
    if _predictor is None:
        _predictor = V2Predictor()
    return _predictor


load_baseline_metadata = load_model_metadata
