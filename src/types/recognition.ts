export type RecognitionStatus = "idle" | "running" | "paused" | "unknown" | "error";

export interface TopPrediction {
  label: string;
  gloss: string;
  confidence: number;
}

export interface RealtimeStats {
  fps: number;
  latencyMs: number;
  modelStatus: RecognitionStatus;
}

export interface PredictionResult {
  label: string;
  text: string;
  confidence: number;
  status: RecognitionStatus;
  topPredictions: TopPrediction[];
  stats: RealtimeStats;
  updatedAt: string;
}

export interface PipelineStep {
  id: string;
  title: string;
  description: string;
  icon: string;
}

export interface MetricItem {
  label: string;
  value: number;
  suffix: string;
  description: string;
}

export interface NewSignPayload {
  label: string;
  description: string;
  region?: string;
  embedding: number[];
  confidence: number;
  source: "user-demo" | "backend";
  createdAt: string;
}
