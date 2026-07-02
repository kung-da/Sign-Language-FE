import {
  DrawingUtils,
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { useEffect, useState, type Dispatch, type RefObject, type SetStateAction } from "react";
import {
  extractFrameFeatures,
  handPreviewToLandmarks,
  preprocessFrameFeatures,
  type FrameFeatures,
} from "../utils/landmarkPreprocessing";

const TARGET_DETECTION_FPS = 60;
const DETECTION_INTERVAL_MS = 1000 / TARGET_DETECTION_FPS;
const INFERENCE_MAX_WIDTH = 480;
const PREVIEW_INTERPOLATION_FRAMES = 60;
const UI_UPDATE_INTERVAL_MS = 250;
const FPS_WINDOW_MS = 1000;

interface LandmarkCounts {
  hands: number;
  face: number;
  pose: number;
}

export type LandmarkLike = {
  x: number;
  y: number;
  z: number;
  visibility: number;
  presence?: number;
};
export type HandednessLabel = "Left" | "Right" | null;

export interface WorkerLandmarks {
  face: LandmarkLike[][];
  faceBlendshapes: number[];
  handedness: HandednessLabel[];
  hands: LandmarkLike[][];
  handWorldLandmarks: LandmarkLike[][];
  pose: LandmarkLike[][];
  poseWorldLandmarks: LandmarkLike[][];
}

export interface PipelinePerformanceMetrics {
  cameraFps: number;
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
  onLandmarks?: (landmarks: WorkerLandmarks) => void;
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
type DisplayDelegate = InferenceDelegate | "Mixed";
type LandmarkTask = "hand" | "face" | "pose";
type WorkerToMainMessage =
  | { type: "ready"; delegate: InferenceDelegate; task: LandmarkTask }
  | {
      type: "result";
      blendshapes?: number[];
      handedness?: HandednessLabel[];
      landmarks: LandmarkLike[][];
      processingMs: number;
      task: LandmarkTask;
      worldLandmarks?: LandmarkLike[][];
    }
  | { type: "error"; message: string; task?: LandmarkTask };

type PerformanceWithMemory = Performance & {
  memory?: {
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  };
};

const emptyMetrics: PipelinePerformanceMetrics = {
  cameraFps: 0,
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

export function useMediaPipeLandmarks({ videoRef, canvasRef, isActive, onLandmarks }: UseMediaPipeLandmarksOptions) {
  const [status, setStatus] = useState<LandmarkStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState<LandmarkCounts>({ hands: 0, face: 0, pose: 0 });
  const [delegate, setDelegate] = useState<DisplayDelegate | null>(null);
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
    let lastMetricsUpdateMs = 0;
    let lastCameraMediaTime = -1;
    let lastCounts: LandmarkCounts = { hands: 0, face: 0, pose: 0 };
    let activeFrameStartedAt = 0;
    let drawing: DrawingUtils | null = null;
    let drawingContext: CanvasRenderingContext2D | null = null;
    const completedFrameTimes: number[] = [];
    const cameraFrameTimes: number[] = [];
    const previewFrameBuffer: FrameFeatures[] = [];
    const workerDelegates: Partial<Record<LandmarkTask, InferenceDelegate>> = {};
    const inferenceCanvas = document.createElement("canvas");
    const inferenceContext = inferenceCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    });
    const workers = createWorkers();
    const latestLandmarks: WorkerLandmarks = {
      face: [],
      faceBlendshapes: [],
      handedness: [],
      hands: [],
      handWorldLandmarks: [],
      pose: [],
      poseWorldLandmarks: [],
    };
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
          workerDelegates[event.data.task] = event.data.delegate;
          if (readyWorkers === 3) {
            const delegates = Object.values(workerDelegates);
            setDelegate(delegates.every((value) => value === "GPU") ? "GPU" : delegates.every((value) => value === "CPU") ? "CPU" : "Mixed");
            setStatus("ready");
          }
          return;
        }

        if (event.data.type === "error") {
          handleWorkerFailure(event.data.message);
          return;
        }

        assignLandmarks(latestLandmarks, event.data.task, event.data.landmarks, event.data);
        latestTaskTimes[event.data.task] = event.data.processingMs;
        pendingResults -= 1;

        if (pendingResults === 0) {
          isFrameInFlight = false;
          const displayLandmarks = buildDisplayLandmarks(latestLandmarks, previewFrameBuffer);
          drawLandmarks(displayLandmarks);
          updateCounts(displayLandmarks);
          const completedAt = performance.now();
          completedFrameTimes.push(completedAt);
          trimFpsWindow(completedFrameTimes, completedAt);
          if (completedAt - lastMetricsUpdateMs >= UI_UPDATE_INTERVAL_MS) {
            lastMetricsUpdateMs = completedAt;
            updatePerformanceMetrics(activeFrameStartedAt, latestTaskTimes, completedFrameTimes, cameraFrameTimes, setMetrics);
          }
          onLandmarks?.({
            face: [...latestLandmarks.face],
            faceBlendshapes: [...latestLandmarks.faceBlendshapes],
            handedness: [...latestLandmarks.handedness],
            hands: [...latestLandmarks.hands],
            handWorldLandmarks: [...latestLandmarks.handWorldLandmarks],
            pose: [...latestLandmarks.pose],
            poseWorldLandmarks: [...latestLandmarks.poseWorldLandmarks],
          });
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
      if (metadata.mediaTime !== lastCameraMediaTime) {
        lastCameraMediaTime = metadata.mediaTime;
        cameraFrameTimes.push(now);
        trimFpsWindow(cameraFrameTimes, now);
      }
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

      // requestVideoFrameCallback already fires once per decoded camera frame.
      // Applying a second 16.67 ms gate can turn a jittery 60 FPS stream into ~30 FPS.
      if (!scheduledWithVideoFrameCallback && now - lastDetectionMs < DETECTION_INTERVAL_MS) return;
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

function assignLandmarks(landmarks: WorkerLandmarks, task: LandmarkTask, nextLandmarks: LandmarkLike[][], message: Extract<WorkerToMainMessage, { type: "result" }>) {
  if (task === "hand") {
    landmarks.hands = nextLandmarks;
    landmarks.handWorldLandmarks = message.worldLandmarks ?? [];
    landmarks.handedness = message.handedness ?? [];
  }
  if (task === "face") {
    landmarks.face = nextLandmarks;
    landmarks.faceBlendshapes = message.blendshapes ?? [];
  }
  if (task === "pose") {
    landmarks.pose = nextLandmarks;
    landmarks.poseWorldLandmarks = message.worldLandmarks ?? [];
  }
}

function buildDisplayLandmarks(landmarks: WorkerLandmarks, previewFrameBuffer: FrameFeatures[]): WorkerLandmarks {
  previewFrameBuffer.push(extractFrameFeatures(landmarks));
  while (previewFrameBuffer.length > PREVIEW_INTERPOLATION_FRAMES) previewFrameBuffer.shift();

  const currentIndex = previewFrameBuffer.length - 1;
  const preprocessed = preprocessFrameFeatures(previewFrameBuffer);

  return {
    ...landmarks,
    hands: handPreviewToLandmarks(preprocessed.previewHands[currentIndex] ?? [], preprocessed.validMask[currentIndex] ?? []),
  };
}

function updatePerformanceMetrics(
  frameStartedAt: number,
  taskTimesMs: PipelinePerformanceMetrics["taskTimesMs"],
  completedFrameTimes: number[],
  cameraFrameTimes: number[],
  setMetrics: Dispatch<SetStateAction<PipelinePerformanceMetrics>>,
) {
  const now = performance.now();
  const extractionTimeMs = Math.max(taskTimesMs.hand ?? 0, taskTimesMs.face ?? 0, taskTimesMs.pose ?? 0);
  setMetrics((currentMetrics) => ({
    ...currentMetrics,
    cameraFps: cameraFrameTimes.length,
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

function trimFpsWindow(frameTimes: number[], now: number) {
  const cutoff = now - FPS_WINDOW_MS;
  while (frameTimes.length && frameTimes[0] < cutoff) frameTimes.shift();
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
