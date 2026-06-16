import type { TopPrediction } from "../../types/recognition";

export function TopPredictions({ predictions }: { predictions: TopPrediction[] }) {
  return (
    <div className="space-y-3">
      {predictions.map((prediction) => (
        <div key={prediction.label}>
          <div className="mb-1 flex items-center justify-between gap-3 text-sm">
            <span className="font-medium text-text">{prediction.label}</span>
            <span className="text-muted">{Math.round(prediction.confidence * 100)}%</span>
          </div>
          <div className="h-2 rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan to-violet"
              style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
