from __future__ import annotations

import importlib.util
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .model import BACKEND_DIR, PredictionResult, V2Predictor


DEFAULT_EXTRACTOR_PATH = Path(os.environ.get("ASL_EXTRACTOR_PATH", r"D:\Project\DBML\src\ASL\extract_asl_landmarks.py"))
UPLOAD_TMP_DIR = BACKEND_DIR / "tmp" / "demo_uploads"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DemoUploadUnavailableError(RuntimeError):
    pass


class DemoUploadPredictor:
    def __init__(self, predictor: V2Predictor, extractor_path: Path = DEFAULT_EXTRACTOR_PATH) -> None:
        self.predictor = predictor
        self.extractor_path = extractor_path
        self.extractor = self._load_extractor()
        self._configure_extractor()
        self.landmarkers = self._create_landmarkers()
        self.timestamp_offset_ms = 0
        self.lock = threading.Lock()

    def close(self) -> None:
        for landmarker in self.landmarkers:
            if landmarker is not None:
                landmarker.close()

    def predict_video(self, filename: str, file_bytes: bytes, top_k: int = 5) -> dict[str, Any]:
        safe_name = safe_file_name(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension '{suffix}'. Expected one of: {', '.join(sorted(VIDEO_EXTENSIONS))}")

        request_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        upload_dir = UPLOAD_TMP_DIR / "uploads"
        npz_dir = UPLOAD_TMP_DIR / "npz"
        upload_dir.mkdir(parents=True, exist_ok=True)
        npz_dir.mkdir(parents=True, exist_ok=True)

        video_path = upload_dir / f"{request_id}_{safe_name}"
        output_npz_path = npz_dir / f"{request_id}_{Path(safe_name).stem}.npz"
        video_path.write_bytes(file_bytes)

        started = time.perf_counter()
        with self.lock:
            result, self.timestamp_offset_ms = self.extractor.process_one_video(
                video_path=video_path,
                output_npz_path=output_npz_path,
                hand_landmarker=self.landmarkers[0],
                pose_landmarker=self.landmarkers[1],
                face_landmarker=self.landmarkers[2],
                timestamp_offset_ms=self.timestamp_offset_ms,
                target_label=Path(safe_name).stem,
            )

        if result is None:
            raise RuntimeError("Video extraction failed.")

        sequence, meta = self._sequence_from_npz(output_npz_path)
        prediction = self.predictor.predict(sequence, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "filename": safe_name,
            "npz_path": str(output_npz_path),
            "extract": result,
            "meta": meta,
            "top_k": prediction.top_k,
            "batch_size": prediction.batch_size,
            "model_inference_ms": prediction.inference_ms,
            "end_to_end_ms": elapsed_ms,
        }

    def _load_extractor(self):
        if not self.extractor_path.exists():
            raise DemoUploadUnavailableError(f"ASL extractor not found: {self.extractor_path}")

        spec = importlib.util.spec_from_file_location("asl_demo_upload_extractor", self.extractor_path)
        if spec is None or spec.loader is None:
            raise DemoUploadUnavailableError(f"Cannot import ASL extractor from: {self.extractor_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _configure_extractor(self) -> None:
        feature_parts = self.predictor.config.get("feature_parts") or ["pose", "hands", "face", "mouth"]
        self.extractor.USE_POSE = "pose" in feature_parts
        self.extractor.USE_HAND = "hands" in feature_parts
        self.extractor.USE_FACE = any(part in feature_parts for part in ("face", "mouth"))
        self.extractor.MEDIAPIPE_DELEGATE = "auto"
        self.extractor.TARGET_FRAMES = self.predictor.seq_len
        self.extractor.MIN_TARGET_FRAMES = self.predictor.seq_len
        self.extractor.MAX_TARGET_FRAMES = None
        self.extractor.ENABLE_PREVIEW = False
        self.extractor.SKIP_EXISTING = False

    def _create_landmarkers(self):
        try:
            self.extractor.download_models()
            if getattr(self.extractor, "MEDIAPIPE_DELEGATE", "auto") == "auto":
                try:
                    self.extractor.MEDIAPIPE_DELEGATE = "gpu"
                    return (
                        self.extractor.create_hand_landmarker() if self.extractor.USE_HAND else None,
                        self.extractor.create_pose_landmarker() if self.extractor.USE_POSE else None,
                        self.extractor.create_face_landmarker() if self.extractor.USE_FACE else None,
                    )
                except Exception:
                    self.extractor.MEDIAPIPE_DELEGATE = "cpu"

            return (
                self.extractor.create_hand_landmarker() if self.extractor.USE_HAND else None,
                self.extractor.create_pose_landmarker() if self.extractor.USE_POSE else None,
                self.extractor.create_face_landmarker() if self.extractor.USE_FACE else None,
            )
        except Exception as exc:
            raise DemoUploadUnavailableError(f"Cannot create MediaPipe landmarkers: {type(exc).__name__}: {exc}") from exc

    def _sequence_from_npz(self, npz_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        with np.load(npz_path, allow_pickle=False) as data:
            parts = [data[key].astype(np.float32, copy=False) for key in ("pose", "hands", "face", "mouth")]
            full_x = np.concatenate(parts, axis=1)
            meta = {
                "label": str(data["label"]) if "label" in data else npz_path.stem,
                "video_name": str(data["video_name"]) if "video_name" in data else npz_path.name,
                "source_total_frames": int(data["source_total_frames"]) if "source_total_frames" in data else None,
                "target_frames": int(data["target_frames"]) if "target_frames" in data else full_x.shape[0],
            }

        feature_indices = np.asarray(self.predictor.config["feature_indices"], dtype=np.int64)
        sequence = np.nan_to_num(full_x[:, feature_indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return sequence, meta


def safe_file_name(name: str) -> str:
    base = Path(name).name or "uploaded.mp4"
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .") or "uploaded.mp4"


_demo_upload_predictor: DemoUploadPredictor | None = None


def get_demo_upload_predictor(predictor: V2Predictor) -> DemoUploadPredictor:
    global _demo_upload_predictor
    if _demo_upload_predictor is None:
        _demo_upload_predictor = DemoUploadPredictor(predictor)
    return _demo_upload_predictor
