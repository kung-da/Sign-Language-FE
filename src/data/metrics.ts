import type { MetricItem } from "../types/recognition";

export const metrics: MetricItem[] = [
  { label: "Accuracy", value: 24.5, suffix: "%", description: "Best validation accuracy (top-1, epoch 45)" },
  { label: "Top-3 Acc", value: 46.9, suffix: "%", description: "Validation top-3 accuracy" },
  { label: "Top-5 Acc", value: 58.0, suffix: "%", description: "Validation top-5 accuracy" },
  { label: "F1-score", value: 19.2, suffix: "%", description: "Macro F1 on validation set" },
  { label: "FPS", value: 30, suffix: "", description: "Target realtime throughput" },
  { label: "Latency", value: 9.8, suffix: "ms", description: "Mean inference latency (CPU)" },
  { label: "Classes", value: 2000, suffix: "", description: "ASL sign classes in vocabulary" },
  { label: "Parameters", value: 2.15, suffix: "M", description: "TCN model parameter count" },
];
