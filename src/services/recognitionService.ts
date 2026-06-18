import { apiClient } from "./apiClient";
import type { PredictionResult } from "../types/recognition";

export interface RecognizeFrameRequest {
  frame?: ImageBitmap | HTMLVideoElement | null;
  sequenceId: string;
}

export const recognitionService = {
  async predictFromSequence(
    sequence: number[][],
    topK = 5,
  ): Promise<PredictionResult> {
    const response = await apiClient.postPredict(sequence, topK);
    const row = response.top_k[0] ?? [];

    const best = row[0];
    const confidence = best?.probability ?? 0;
    const label = best?.label ?? "Unknown";
    const status = confidence < 0.3 ? "unknown" : "running";

    return {
      label,
      text: label,
      confidence,
      status,
      topPredictions: row.slice(0, 3).map((item) => ({
        label: item.label,
        gloss: `class ${item.class_index}`,
        confidence: item.probability,
      })),
      stats: {
        fps: 0,
        latencyMs: Math.round(response.model_inference_ms),
        modelStatus: status,
      },
      updatedAt: new Date().toISOString(),
    };
  },
};
