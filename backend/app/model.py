from __future__ import annotations

import json
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


ROOT_DIR = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT_DIR / "asl_tcn_baseline"
CHECKPOINT_PATH = BASELINE_DIR / "best.pt"
CONFIG_PATH = BASELINE_DIR / "config.json"
HISTORY_PATH = BASELINE_DIR / "history.json"
LATENCY_PATH = BASELINE_DIR / "latency.json"


class ModelUnavailableError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_metadata() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    history = read_json(HISTORY_PATH)
    latency = read_json(LATENCY_PATH)
    checkpoint_size_mb = (
        CHECKPOINT_PATH.stat().st_size / (1024 * 1024) if CHECKPOINT_PATH.exists() else None
    )

    best_epoch = None
    best_val_acc = None
    rows = history.get("history") if isinstance(history, dict) else None
    if isinstance(rows, list):
        for row in rows:
            val_acc = row.get("val_acc") if isinstance(row, dict) else None
            if isinstance(val_acc, (int, float)) and (
                best_val_acc is None or val_acc > best_val_acc
            ):
                best_val_acc = val_acc
                best_epoch = row.get("epoch")

    return {
        "baseline_dir": str(BASELINE_DIR),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "checkpoint_size_mb": checkpoint_size_mb,
        "config": config,
        "latency": latency,
        "history_summary": {
            "best_epoch_by_val_acc": best_epoch,
            "best_val_acc": best_val_acc,
            "epochs_recorded": len(rows) if isinstance(rows, list) else None,
        },
        "runtime": {
            "torch_available": torch is not None,
            "normalization_available": False,
            "labels_available": False,
        },
    }


if nn is not None:

    class TemporalBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation // 2
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.res = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.res(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout(out)
            return self.relu(out + residual)


    class TCNClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                TemporalBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
                TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
                TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
            )
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            x = self.net(x)
            x = x.mean(dim=2)
            return self.classifier(x)


@dataclass
class PredictionResult:
    top_k: list[list[dict[str, float | int | str]]]
    inference_ms: float
    batch_size: int


class BaselinePredictor:
    def __init__(self, device: str = "cpu") -> None:
        if torch is None or nn is None:
            raise ModelUnavailableError(
                "PyTorch is not installed. Install torch in a Python 3.11/3.12 backend env "
                "to run best.pt inference."
            )
        if not CHECKPOINT_PATH.exists():
            raise ModelUnavailableError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        self.config = read_json(CONFIG_PATH)
        self.seq_len = int(self.config.get("seq_len", 60))
        self.input_dim = int(self.config.get("input_dim", 634))
        self.num_classes = int(self.config.get("num_classes", 2000))
        self.device = torch.device(device)

        self.model = TCNClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            dropout=float(self.config.get("dropout", 0.5)),
        ).to(self.device)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def validate_sequence(self, sequence: Any) -> np.ndarray:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError("sequence must have shape [60, 634] or [batch, 60, 634]")
        expected = (self.seq_len, self.input_dim)
        if tuple(arr.shape[1:]) != expected:
            raise ValueError(f"expected sequence shape [batch, {expected[0]}, {expected[1]}], got {list(arr.shape)}")
        return arr

    def predict(self, sequence: Any, top_k: int = 5) -> PredictionResult:
        arr = self.validate_sequence(sequence)
        top_k = max(1, min(int(top_k), self.num_classes))

        with torch.inference_mode():
            tensor = torch.from_numpy(arr).to(self.device)
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
                        "label": f"class_{int(index)}",
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


_predictor: BaselinePredictor | None = None


def get_predictor() -> BaselinePredictor:
    global _predictor
    if _predictor is None:
        _predictor = BaselinePredictor()
    return _predictor
