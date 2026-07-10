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
    return predictionFromResponse(response.top_k[0] ?? [], Math.round(response.model_inference_ms));
  },

  async predictFromVideo(file: File, topK = 5): Promise<PredictionResult> {
    const response = await apiClient.postPredictVideo(file, topK);
    return predictionFromResponse(response.top_k[0] ?? [], Math.round(response.end_to_end_ms));
  },
};

function predictionFromResponse(
  row: Array<{ class_index: number; label: string; probability: number }>,
  latencyMs: number,
): PredictionResult {
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
      latencyMs,
      modelStatus: status,
    },
    updatedAt: new Date().toISOString(),
  };
}
