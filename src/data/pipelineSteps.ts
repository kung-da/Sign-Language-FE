import type { PipelineStep } from "../types/recognition";

export const pipelineSteps: PipelineStep[] = [
  {
    id: "input",
    title: "Video/Webcam Input",
    description: "Accept live camera frames or uploaded demo video as the visual source.",
    icon: "Camera",
  },
  {
    id: "preprocess",
    title: "Preprocessing",
    description: "Normalize frame size, crop regions of interest, and prepare frames for pose estimation.",
    icon: "SlidersHorizontal",
  },
  {
    id: "keypoints",
    title: "MediaPipe Keypoints",
    description: "Extract hands, pose, and face landmarks as structured keypoint tensors.",
    icon: "ScanSearch",
  },
  {
    id: "buffer",
    title: "Sequence Buffer",
    description: "Group consecutive frames into temporal windows for sign-level context.",
    icon: "Layers3",
  },
  {
    id: "encoder",
    title: "Encoder Embedding",
    description: "Run LSTM, GRU, or Transformer encoder logic to produce a compact embedding.",
    icon: "BrainCircuit",
  },
  {
    id: "vector",
    title: "Vector Search",
    description: "Search nearest known sign embeddings in a vector database.",
    icon: "Database",
  },
  {
    id: "smooth",
    title: "Result Smoothing",
    description: "Apply confidence thresholding and voting to reduce flicker.",
    icon: "Activity",
  },
  {
    id: "output",
    title: "Vietnamese Text Output",
    description: "Display the best matching Vietnamese phrase and supporting confidence data.",
    icon: "MessageSquareText",
  },
];
