import type { MetricItem } from "../types/recognition";

export const metrics: MetricItem[] = [
  { label: "Accuracy", value: 91.8, suffix: "%", description: "Mock validation accuracy" },
  { label: "Precision", value: 89.4, suffix: "%", description: "Demo class precision" },
  { label: "Recall", value: 88.7, suffix: "%", description: "Demo class recall" },
  { label: "F1-score", value: 89.0, suffix: "%", description: "Balanced mock score" },
  { label: "FPS", value: 30, suffix: "", description: "Target realtime throughput" },
  { label: "Latency", value: 42, suffix: "ms", description: "Mock end-to-end latency" },
  { label: "Confidence", value: 86, suffix: "%", description: "Average mock confidence" },
  { label: "Unknown sign rate", value: 7.5, suffix: "%", description: "Low-confidence demo cases" },
];
