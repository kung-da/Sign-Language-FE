import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { useEffect, useState, type RefObject } from "react";

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_ROOT = "https://storage.googleapis.com/mediapipe-models";
const DETECTION_INTERVAL_MS = 85;
const INFERENCE_MAX_WIDTH = 480;
const UI_UPDATE_INTERVAL_MS = 250;

interface LandmarkCounts {
  hands: number;
  face: number;
  pose: number;
}

interface UseMediaPipeLandmarksOptions {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  isActive: boolean;
}

type LandmarkStatus = "idle" | "loading" | "ready" | "error";
type InferenceDelegate = "CPU" | "GPU";
type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

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
    let animationFrame = 0;
    let handLandmarker: HandLandmarker | null = null;
    let faceLandmarker: FaceLandmarker | null = null;
    let poseLandmarker: PoseLandmarker | null = null;
    let lastDetectionMs = 0;
    let lastUiUpdateMs = 0;
    let lastCounts: LandmarkCounts = { hands: 0, face: 0, pose: 0 };
    const inferenceCanvas = document.createElement("canvas");
    const inferenceContext = inferenceCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    });

    const loadAndRun = async () => {
      try {
        setStatus("loading");
        setError(null);

        const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
        let selectedDelegate: InferenceDelegate = "GPU";
        let landmarkers = await createLandmarkers(vision, selectedDelegate).catch(() => null);

        if (!landmarkers) {
          selectedDelegate = "CPU";
          landmarkers = await createLandmarkers(vision, selectedDelegate);
        }

        if (isCancelled) {
          landmarkers.hands.close();
          landmarkers.face.close();
          landmarkers.pose.close();
          return;
        }

        handLandmarker = landmarkers.hands;
        faceLandmarker = landmarkers.face;
        poseLandmarker = landmarkers.pose;
        setDelegate(selectedDelegate);
        setStatus("ready");

        const detect = () => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          const context = canvas?.getContext("2d");

          if (
            isCancelled ||
            !video ||
            !canvas ||
            !context ||
            !inferenceContext ||
            !handLandmarker ||
            !faceLandmarker ||
            !poseLandmarker
          ) {
            return;
          }

          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
            animationFrame = window.requestAnimationFrame(detect);
            return;
          }

          const now = performance.now();
          if (now - lastDetectionMs < DETECTION_INTERVAL_MS) {
            animationFrame = window.requestAnimationFrame(detect);
            return;
          }
          lastDetectionMs = now;

          syncCanvasToVideo(canvas, video);
          context.clearRect(0, 0, canvas.width, canvas.height);
          syncInferenceCanvas(inferenceCanvas, video);
          inferenceContext.drawImage(video, 0, 0, inferenceCanvas.width, inferenceCanvas.height);

          const timestamp = video.currentTime * 1000;
          const handResult = handLandmarker.detectForVideo(inferenceCanvas, timestamp);
          const faceResult = faceLandmarker.detectForVideo(inferenceCanvas, timestamp);
          const poseResult = poseLandmarker.detectForVideo(inferenceCanvas, timestamp);
          const drawing = new DrawingUtils(context);

          for (const handLandmarks of handResult.landmarks) {
            drawing.drawConnectors(handLandmarks, HandLandmarker.HAND_CONNECTIONS, {
              color: "#22c55e",
              lineWidth: 3,
            });
            drawing.drawLandmarks(handLandmarks, { color: "#ecfeff", fillColor: "#22c55e", radius: 3 });
          }

          for (const faceLandmarks of faceResult.faceLandmarks) {
            drawing.drawConnectors(faceLandmarks, FaceLandmarker.FACE_LANDMARKS_CONTOURS, {
              color: "#38bdf8",
              lineWidth: 1,
            });
          }

          for (const poseLandmarks of poseResult.landmarks) {
            drawing.drawConnectors(poseLandmarks, PoseLandmarker.POSE_CONNECTIONS, {
              color: "#fb7185",
              lineWidth: 3,
            });
            drawing.drawLandmarks(poseLandmarks, { color: "#fff7ed", fillColor: "#fb7185", radius: 3 });
          }

          const nextCounts = {
            hands: handResult.landmarks.length,
            face: faceResult.faceLandmarks.length,
            pose: poseResult.landmarks.length,
          };

          if (now - lastUiUpdateMs > UI_UPDATE_INTERVAL_MS && hasCountsChanged(lastCounts, nextCounts)) {
            lastCounts = nextCounts;
            lastUiUpdateMs = now;
            setCounts(nextCounts);
          }

          animationFrame = window.requestAnimationFrame(detect);
        };

        animationFrame = window.requestAnimationFrame(detect);
      } catch {
        if (!isCancelled) {
          setStatus("error");
          setError("MediaPipe landmarks could not be loaded. Check your network connection.");
        }
      }
    };

    void loadAndRun();

    return () => {
      isCancelled = true;
      window.cancelAnimationFrame(animationFrame);
      handLandmarker?.close();
      faceLandmarker?.close();
      poseLandmarker?.close();
      clearCanvas(canvasRef.current);
    };
  }, [canvasRef, isActive, videoRef]);

  return { counts, delegate, error, status };
}

async function createLandmarkers(vision: VisionFileset, delegate: InferenceDelegate) {
  const canvasOptions =
    delegate === "GPU"
      ? {
          canvas: [createGpuCanvas(), createGpuCanvas(), createGpuCanvas()],
        }
      : null;
  let hands: HandLandmarker | null = null;
  let face: FaceLandmarker | null = null;
  let pose: PoseLandmarker | null = null;

  try {
    hands = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        delegate,
        modelAssetPath: `${MODEL_ROOT}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
      },
      canvas: canvasOptions?.canvas[0],
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    face = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        delegate,
        modelAssetPath: `${MODEL_ROOT}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
      },
      canvas: canvasOptions?.canvas[1],
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });

    pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        delegate,
        modelAssetPath: `${MODEL_ROOT}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`,
      },
      canvas: canvasOptions?.canvas[2],
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
    });

    return { face, hands, pose };
  } catch (error) {
    hands?.close();
    face?.close();
    pose?.close();
    throw error;
  }
}

function createGpuCanvas() {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(1, 1);
  }

  return document.createElement("canvas");
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
