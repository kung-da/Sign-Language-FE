import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

const WASM_ROOT = import.meta.env.DEV ? "/mediapipe-dev-wasm" : "/mediapipe/wasm";
const MODEL_ROOT = "/mediapipe/models";

type InferenceDelegate = "CPU" | "GPU";
type LandmarkTask = "hand" | "face" | "pose";
type LandmarkLike = {
  x: number;
  y: number;
  z: number;
  visibility: number;
  presence?: number;
};
type HandednessLabel = "Left" | "Right" | null;

interface InitMessage {
  type: "init";
  task: LandmarkTask;
}

interface DetectMessage {
  type: "detect";
  frame: ImageBitmap;
  timestamp: number;
}

type MainToWorkerMessage = InitMessage | DetectMessage;
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

let task: LandmarkTask | null = null;
let landmarker: HandLandmarker | FaceLandmarker | PoseLandmarker | null = null;
let lastTimestamp = 0;

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<MainToWorkerMessage>) => void) | null;
  postMessage: (message: WorkerToMainMessage) => void;
};

workerScope.onmessage = (event) => {
  if (event.data.type === "init") {
    void init(event.data.task);
    return;
  }

  if (!task || !landmarker) {
    event.data.frame.close();
    return;
  }

  try {
    const timestamp = Math.max(event.data.timestamp, lastTimestamp + 1);
    lastTimestamp = timestamp;
    const startedAt = performance.now();
    const result = detectLandmarks(landmarker, task, event.data.frame, timestamp);

    workerScope.postMessage({
      type: "result",
      task,
      landmarks: cloneLandmarks(result.landmarks),
      worldLandmarks: result.worldLandmarks ? cloneLandmarks(result.worldLandmarks) : undefined,
      handedness: result.handedness,
      blendshapes: result.blendshapes,
      processingMs: performance.now() - startedAt,
    });
  } catch (error) {
    workerScope.postMessage({
      type: "error",
      task,
      message: `MediaPipe ${task} worker failed while processing a camera frame: ${getErrorMessage(error)}`,
    });
  } finally {
    event.data.frame.close();
  }
};

async function init(nextTask: LandmarkTask) {
  try {
    task = nextTask;
    let selectedDelegate: InferenceDelegate = "GPU";
    try {
      landmarker = await createLandmarker(nextTask, selectedDelegate);
    } catch {
      selectedDelegate = "CPU";
      landmarker = await createLandmarker(nextTask, selectedDelegate);
    }
    workerScope.postMessage({ type: "ready", delegate: selectedDelegate, task: nextTask });
  } catch (error) {
    workerScope.postMessage({
      type: "error",
      task: nextTask,
      message: `MediaPipe ${nextTask} landmarks could not be loaded: ${getErrorMessage(error)}`,
    });
  }
}

async function createLandmarker(nextTask: LandmarkTask, delegate: InferenceDelegate) {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT, true);

  if (nextTask === "hand") {
    return HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        delegate,
        modelAssetPath: `${MODEL_ROOT}/hand_landmarker.task`,
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  }

  if (nextTask === "face") {
    return FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        delegate,
        modelAssetPath: `${MODEL_ROOT}/face_landmarker.task`,
      },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    });
  }

  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      delegate,
      modelAssetPath: `${MODEL_ROOT}/pose_landmarker_lite.task`,
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  });
}

function detectLandmarks(
  nextLandmarker: HandLandmarker | FaceLandmarker | PoseLandmarker,
  nextTask: LandmarkTask,
  frame: ImageBitmap,
  timestamp: number,
) {
  if (nextTask === "hand") {
    const result = (nextLandmarker as HandLandmarker).detectForVideo(frame, timestamp);
    return {
      handedness: result.handedness.map((item) => getHandednessLabel(item)),
      landmarks: result.landmarks,
      worldLandmarks: result.worldLandmarks,
    };
  }

  if (nextTask === "face") {
    const result = (nextLandmarker as FaceLandmarker).detectForVideo(frame, timestamp);
    return {
      blendshapes: cloneBlendshapes(result.faceBlendshapes),
      landmarks: result.faceLandmarks,
    };
  }

  const result = (nextLandmarker as PoseLandmarker).detectForVideo(frame, timestamp);
  return {
    landmarks: result.landmarks,
    worldLandmarks: result.worldLandmarks,
  };
}

function cloneLandmarks(landmarks: LandmarkLike[][]) {
  return landmarks.map((landmarkGroup) =>
    landmarkGroup.map((landmark) => ({
      x: landmark.x,
      y: landmark.y,
      z: landmark.z,
      visibility: landmark.visibility ?? 0,
      presence: landmark.presence,
    })),
  );
}

function cloneBlendshapes(faceBlendshapes: Array<{ categories: Array<{ score: number }> }>) {
  return faceBlendshapes[0]?.categories.map((category) => category.score) ?? [];
}

function getHandednessLabel(handedness: Array<{ categoryName?: string; category_name?: string }>): HandednessLabel {
  const label = handedness[0]?.categoryName ?? handedness[0]?.category_name;
  if (label === "Left" || label === "Right") return label;
  return null;
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "Unknown error";
}

export {};
