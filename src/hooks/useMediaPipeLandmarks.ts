import {
  DrawingUtils,
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { useEffect, useState, type Dispatch, type RefObject, type SetStateAction } from "react";

const DETECTION_INTERVAL_MS = 16;
const INFERENCE_MAX_WIDTH = 480;
const UI_UPDATE_INTERVAL_MS = 250;
const FPS_WINDOW_MS = 1000;

interface LandmarkCounts {
  hands: number;
  face: number;
  pose: number;
}

interface WorkerLandmarks {
  hands: NormalizedLandmark[][];
  face: NormalizedLandmark[][];
  pose: NormalizedLandmark[][];
}

export interface PipelinePerformanceMetrics {
  endToEndLatencyMs: number | null;
  extractionTimeMs: number | null;
  fps: number;
  inferenceSize: {
    totalBytes: number;
    items: Array<{ label: string; bytes: number }>;
  } | null;
  memory: {
    usedJSHeapSize: number;
    totalJSHeapSize: number;
  } | null;
  modelInferenceTimeMs: number | null;
  smoothingWindowSize: number;
  taskTimesMs: {
    hand: number | null;
    face: number | null;
    pose: number | null;
  };
}

interface UseMediaPipeLandmarksOptions {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  isActive: boolean;
}

interface VideoFrameMetadata {
  mediaTime: number;
}

type VideoFrameCallback = (now: DOMHighResTimeStamp, metadata: VideoFrameMetadata) => void;
type VideoElementWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: VideoFrameCallback) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

type LandmarkStatus = "idle" | "loading" | "ready" | "error";
type InferenceDelegate = "CPU" | "GPU";
type LandmarkTask = "hand" | "face" | "pose";
type WorkerToMainMessage =
  | { type: "ready"; delegate: InferenceDelegate; task: LandmarkTask }
  | { type: "result"; landmarks: NormalizedLandmark[][]; processingMs: number; task: LandmarkTask }
  | { type: "error"; message: string; task?: LandmarkTask };

type PerformanceWithMemory = Performance & {
  memory?: {
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  };
};

const emptyMetrics: PipelinePerformanceMetrics = {
  endToEndLatencyMs: null,
  extractionTimeMs: null,
  fps: 0,
  inferenceSize: null,
  memory: null,
  modelInferenceTimeMs: null,
  smoothingWindowSize: 5,
  taskTimesMs: {
    hand: null,
    face: null,
    pose: null,
  },
};

export function useMediaPipeLandmarks({ videoRef, canvasRef, isActive }: UseMediaPipeLandmarksOptions) {
  const [status, setStatus] = useState<LandmarkStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState<LandmarkCounts>({ hands: 0, face: 0, pose: 0 });
  const [delegate, setDelegate] = useState<InferenceDelegate | null>(null);
  const [metrics, setMetrics] = useState<PipelinePerformanceMetrics>(emptyMetrics);

  useEffect(() => {
    if (!isActive) {
      setStatus("idle");
      setDelegate(null);
      setCounts({ hands: 0, face: 0, pose: 0 });
      setMetrics((currentMetrics) => ({
        ...emptyMetrics,
        inferenceSize: currentMetrics.inferenceSize,
      }));
      clearCanvas(canvasRef.current);
      return;
    }

    let isCancelled = false;
    let readyWorkers = 0;
    let isFrameInFlight = false;
    let pendingResults = 0;
    let scheduledFrame = 0;
    let scheduledWithVideoFrameCallback = false;
    let lastDetectionMs = 0;
    let lastUiUpdateMs = 0;
    let lastCounts: LandmarkCounts = { hands: 0, face: 0, pose: 0 };
    let activeFrameStartedAt = 0;
    let drawing: DrawingUtils | null = null;
    let drawingContext: CanvasRenderingContext2D | null = null;
    const completedFrameTimes: number[] = [];
    const inferenceCanvas = document.createElement("canvas");
    const inferenceContext = inferenceCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    });
    const workers = createWorkers();
    const latestLandmarks: WorkerLandmarks = { hands: [], face: [], pose: [] };
    const latestTaskTimes: PipelinePerformanceMetrics["taskTimesMs"] = {
      hand: null,
      face: null,
      pose: null,
    };

    setStatus("loading");
    setError(null);
    void loadModelSizes().then((inferenceSize) => {
      if (!isCancelled) {
        setMetrics((currentMetrics) => ({ ...currentMetrics, inferenceSize }));
      }
    });

    const handleWorkerFailure = (message: string) => {
      if (isCancelled) return;
      isFrameInFlight = false;
      pendingResults = 0;
      setStatus("error");
      setError(message);
    };

    for (const [task, worker] of Object.entries(workers) as Array<[LandmarkTask, Worker]>) {
      worker.onerror = (event) => {
        handleWorkerFailure(event.message || `MediaPipe ${task} worker crashed while running landmark detection.`);
      };

      worker.onmessageerror = () => {
        handleWorkerFailure(`MediaPipe ${task} worker could not transfer landmark data.`);
      };

      worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
        if (isCancelled) return;

        if (event.data.type === "ready") {
          readyWorkers += 1;
          setDelegate(event.data.delegate);
          if (readyWorkers === 3) setStatus("ready");
          return;
        }

        if (event.data.type === "error") {
          handleWorkerFailure(event.data.message);
          return;
        }

        assignLandmarks(latestLandmarks, event.data.task, event.data.landmarks);
        latestTaskTimes[event.data.task] = event.data.processingMs;
        pendingResults -= 1;

        if (pendingResults === 0) {
          isFrameInFlight = false;
          drawLandmarks(latestLandmarks);
          updateCounts(latestLandmarks);
          updatePerformanceMetrics(activeFrameStartedAt, latestTaskTimes, completedFrameTimes, setMetrics);
        }
      };

      worker.postMessage({ type: "init", task });
    }

    const drawLandmarks = (landmarks: WorkerLandmarks) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      if (!drawingContext) {
        drawingContext = canvas.getContext("2d", {
          alpha: true,
          desynchronized: true,
        });
        drawing = drawingContext ? new DrawingUtils(drawingContext) : null;
      }

      if (!drawingContext || !drawing) return;

      drawingContext.clearRect(0, 0, canvas.width, canvas.height);

      for (const handLandmarks of landmarks.hands) {
        drawing.drawConnectors(handLandmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: "#22c55e",
          lineWidth: 3,
        });
        drawing.drawLandmarks(handLandmarks, { color: "#ecfeff", fillColor: "#22c55e", radius: 3 });
      }

      for (const faceLandmarks of landmarks.face) {
        drawing.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_CONTOURS, {
          color: "#38bdf8",
          lineWidth: 1,
        });
      }

      for (const poseLandmarks of landmarks.pose) {
        drawing.drawConnectors(poseLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
          color: "#fb7185",
          lineWidth: 3,
        });
        drawing.drawLandmarks(poseLandmarks, { color: "#fff7ed", fillColor: "#fb7185", radius: 3 });
      }
    };

    const updateCounts = (landmarks: WorkerLandmarks) => {
      const now = performance.now();
      const nextCounts = {
        hands: landmarks.hands.length,
        face: landmarks.face.length,
        pose: landmarks.pose.length,
      };

      if (now - lastUiUpdateMs > UI_UPDATE_INTERVAL_MS && hasCountsChanged(lastCounts, nextCounts)) {
        lastCounts = nextCounts;
        lastUiUpdateMs = now;
        setCounts(nextCounts);
      }
    };

    const scheduleNextDetection = () => {
      const video = videoRef.current as VideoElementWithFrameCallback | null;

      if (isCancelled || !video) return;

      if (video.requestVideoFrameCallback) {
        scheduledWithVideoFrameCallback = true;
        scheduledFrame = video.requestVideoFrameCallback(detect);
        return;
      }

      scheduledWithVideoFrameCallback = false;
      scheduledFrame = window.requestAnimationFrame((now) => {
        detect(now, { mediaTime: video.currentTime });
      });
    };

    const detect: VideoFrameCallback = (now, metadata) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!isCancelled) scheduleNextDetection();
      if (
        isCancelled ||
        readyWorkers < 3 ||
        isFrameInFlight ||
        !video ||
        !canvas ||
        !inferenceContext ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        !video.videoWidth ||
        !video.videoHeight
      ) {
        return;
      }

      if (now - lastDetectionMs < DETECTION_INTERVAL_MS) return;
      lastDetectionMs = now;

      syncCanvasToVideo(canvas, video);
      syncInferenceCanvas(inferenceCanvas, video);
      inferenceContext.drawImage(video, 0, 0, inferenceCanvas.width, inferenceCanvas.height);
      isFrameInFlight = true;
      pendingResults = 3;
      activeFrameStartedAt = performance.now();

      void Promise.all([
        createImageBitmap(inferenceCanvas),
        createImageBitmap(inferenceCanvas),
        createImageBitmap(inferenceCanvas),
      ])
        .then(([handFrame, faceFrame, poseFrame]) => {
          const frames = { hand: handFrame, face: faceFrame, pose: poseFrame };

          if (isCancelled) {
            handFrame.close();
            faceFrame.close();
            poseFrame.close();
            return;
          }

          for (const [task, frame] of Object.entries(frames) as Array<[LandmarkTask, ImageBitmap]>) {
            workers[task].postMessage(
              {
                type: "detect",
                frame,
                timestamp: metadata.mediaTime * 1000,
              },
              [frame],
            );
          }
        })
        .catch(() => {
          isFrameInFlight = false;
          pendingResults = 0;
        });
    };

    scheduleNextDetection();

    return () => {
      isCancelled = true;
      const video = videoRef.current as VideoElementWithFrameCallback | null;
      if (scheduledWithVideoFrameCallback && video?.cancelVideoFrameCallback) {
        video.cancelVideoFrameCallback(scheduledFrame);
      } else {
        window.cancelAnimationFrame(scheduledFrame);
      }
      Object.values(workers).forEach((worker) => worker.terminate());
      clearCanvas(canvasRef.current);
    };
  }, [canvasRef, isActive, videoRef]);

  return { counts, delegate, error, metrics, status };
}

function createWorkers() {
  return {
    hand: createWorker(),
    face: createWorker(),
    pose: createWorker(),
  };
}

function createWorker() {
  return new Worker(new URL("../workers/mediaPipeLandmarks.worker.ts", import.meta.url), {
    type: "module",
  });
}

function assignLandmarks(landmarks: WorkerLandmarks, task: LandmarkTask, nextLandmarks: NormalizedLandmark[][]) {
  if (task === "hand") landmarks.hands = nextLandmarks;
  if (task === "face") landmarks.face = nextLandmarks;
  if (task === "pose") landmarks.pose = nextLandmarks;
}

function updatePerformanceMetrics(
  frameStartedAt: number,
  taskTimesMs: PipelinePerformanceMetrics["taskTimesMs"],
  completedFrameTimes: number[],
  setMetrics: Dispatch<SetStateAction<PipelinePerformanceMetrics>>,
) {
  const now = performance.now();
  const extractionTimeMs = Math.max(taskTimesMs.hand ?? 0, taskTimesMs.face ?? 0, taskTimesMs.pose ?? 0);
  const cutoff = now - FPS_WINDOW_MS;
  completedFrameTimes.push(now);

  while (completedFrameTimes.length && completedFrameTimes[0] < cutoff) {
    completedFrameTimes.shift();
  }

  setMetrics((currentMetrics) => ({
    ...currentMetrics,
    endToEndLatencyMs: Math.round(now - frameStartedAt),
    extractionTimeMs: Math.round(extractionTimeMs),
    fps: completedFrameTimes.length,
    memory: getMemorySnapshot(),
    taskTimesMs: {
      hand: roundNullable(taskTimesMs.hand),
      face: roundNullable(taskTimesMs.face),
      pose: roundNullable(taskTimesMs.pose),
    },
  }));
}

function hasCountsChanged(previous: LandmarkCounts, next: LandmarkCounts) {
  return previous.hands !== next.hands || previous.face !== next.face || previous.pose !== next.pose;
}

function syncCanvasToVideo(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
  if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
  if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
}

function syncInferenceCanvas(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
  const width = Math.min(INFERENCE_MAX_WIDTH, video.videoWidth);
  const height = Math.round(width / (video.videoWidth / video.videoHeight));

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
}

function clearCanvas(canvas: HTMLCanvasElement | null) {
  const context = canvas?.getContext("2d");
  if (!canvas || !context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
}

async function loadModelSizes() {
  const items = await Promise.all([
    getAssetSize("Hand landmarker", "/mediapipe/models/hand_landmarker.task"),
    getAssetSize("Face landmarker", "/mediapipe/models/face_landmarker.task"),
    getAssetSize("Pose landmarker lite", "/mediapipe/models/pose_landmarker_lite.task"),
  ]);

  return {
    items,
    totalBytes: items.reduce((total, item) => total + item.bytes, 0),
  };
}

async function getAssetSize(label: string, url: string) {
  const response = await fetch(url, { method: "HEAD" });
  const contentLength = response.headers.get("content-length");

  if (contentLength) {
    return { label, bytes: Number(contentLength) };
  }

  const fallbackResponse = await fetch(url);
  const blob = await fallbackResponse.blob();
  return { label, bytes: blob.size };
}

function getMemorySnapshot() {
  const memory = (performance as PerformanceWithMemory).memory;
  if (!memory) return null;

  return {
    totalJSHeapSize: memory.totalJSHeapSize,
    usedJSHeapSize: memory.usedJSHeapSize,
  };
}

function roundNullable(value: number | null) {
  return value === null ? null : Math.round(value);
}
