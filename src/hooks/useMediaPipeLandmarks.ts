import {
  DrawingUtils,
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import { useEffect, useState, type RefObject } from "react";

const DETECTION_INTERVAL_MS = 85;
const INFERENCE_MAX_WIDTH = 480;
const UI_UPDATE_INTERVAL_MS = 250;

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
  | { type: "result"; landmarks: NormalizedLandmark[][]; task: LandmarkTask }
  | { type: "error"; message: string; task?: LandmarkTask };

export function useMediaPipeLandmarks({ videoRef, canvasRef, isActive }: UseMediaPipeLandmarksOptions) {
  const [status, setStatus] = useState<LandmarkStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState<LandmarkCounts>({ hands: 0, face: 0, pose: 0 });
  const [delegate, setDelegate] = useState<InferenceDelegate | null>(null);

  useEffect(() => {
    if (!isActive) {
      setStatus("idle");
      setDelegate(null);
      setCounts({ hands: 0, face: 0, pose: 0 });
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
    let drawing: DrawingUtils | null = null;
    let drawingContext: CanvasRenderingContext2D | null = null;
    const inferenceCanvas = document.createElement("canvas");
    const inferenceContext = inferenceCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    });
    const workers = createWorkers();
    const latestLandmarks: WorkerLandmarks = { hands: [], face: [], pose: [] };

    setStatus("loading");
    setError(null);

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
        pendingResults -= 1;

        if (pendingResults === 0) {
          isFrameInFlight = false;
          drawLandmarks(latestLandmarks);
          updateCounts(latestLandmarks);
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

  return { counts, delegate, error, status };
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
