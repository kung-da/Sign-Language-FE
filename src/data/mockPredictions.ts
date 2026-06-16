import type { PredictionResult } from "../types/recognition";

export const mockPredictions: PredictionResult[] = [
  {
    label: "Xin chao",
    text: "Xin chao",
    confidence: 0.92,
    status: "running",
    topPredictions: [
      { label: "Xin chao", gloss: "Greeting", confidence: 0.92 },
      { label: "Cam on", gloss: "Thanks", confidence: 0.05 },
      { label: "Tam biet", gloss: "Goodbye", confidence: 0.03 },
    ],
    stats: { fps: 28, latencyMs: 44, modelStatus: "running" },
    updatedAt: new Date().toISOString(),
  },
  {
    label: "Cam on",
    text: "Cam on",
    confidence: 0.87,
    status: "running",
    topPredictions: [
      { label: "Cam on", gloss: "Thanks", confidence: 0.87 },
      { label: "Xin loi", gloss: "Sorry", confidence: 0.08 },
      { label: "Xin chao", gloss: "Greeting", confidence: 0.05 },
    ],
    stats: { fps: 31, latencyMs: 38, modelStatus: "running" },
    updatedAt: new Date().toISOString(),
  },
  {
    label: "Unknown sign",
    text: "Chua nhan dien",
    confidence: 0.41,
    status: "unknown",
    topPredictions: [
      { label: "Unknown sign", gloss: "Low confidence", confidence: 0.41 },
      { label: "Can giup do", gloss: "Need help", confidence: 0.28 },
      { label: "Benh vien", gloss: "Hospital", confidence: 0.16 },
    ],
    stats: { fps: 24, latencyMs: 62, modelStatus: "unknown" },
    updatedAt: new Date().toISOString(),
  },
];
