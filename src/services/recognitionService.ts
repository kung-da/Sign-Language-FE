import { mockPredictions } from "../data/mockPredictions";
import type { PredictionResult } from "../types/recognition";

export interface RecognizeFrameRequest {
  frame?: ImageBitmap | HTMLVideoElement | null;
  sequenceId: string;
}

const jitter = (value: number, range: number) => {
  const next = value + (Math.random() - 0.5) * range;
  return Number(Math.max(0, Math.min(1, next)).toFixed(2));
};

export const recognitionService = {
  async recognizeFrame(_request: RecognizeFrameRequest): Promise<PredictionResult> {
    return this.getPrediction();
  },

  async getPrediction(): Promise<PredictionResult> {
    const sample = mockPredictions[Math.floor(Math.random() * mockPredictions.length)];
    const confidence = jitter(sample.confidence, 0.14);
    const status = confidence < 0.5 ? "unknown" : "running";

    return {
      ...sample,
      confidence,
      status,
      stats: {
        fps: Math.round(24 + Math.random() * 8),
        latencyMs: Math.round(34 + Math.random() * 32),
        modelStatus: status,
      },
      topPredictions: sample.topPredictions.map((item, index) => ({
        ...item,
        confidence: index === 0 ? confidence : jitter(item.confidence, 0.08),
      })),
      updatedAt: new Date().toISOString(),
    };
  },
};
