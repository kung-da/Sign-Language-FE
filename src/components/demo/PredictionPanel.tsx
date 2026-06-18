import { Plus } from "lucide-react";
import type { PredictionResult } from "../../types/recognition";
import { Button } from "../ui/Button";
import { GlassCard } from "../ui/GlassCard";
import { StatusPill } from "../ui/StatusPill";
import { RealtimeStats } from "./RealtimeStats";
import { TopPredictions } from "./TopPredictions";

interface PredictionPanelProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
  onAddSign: () => void;
}

export function PredictionPanel({ prediction, isLoading, onAddSign }: PredictionPanelProps) {
  const confidence = prediction ? Math.round(prediction.confidence * 100) : 0;
  const showAdd = prediction?.status === "unknown" || (prediction?.confidence ?? 1) < 0.55;

  return (
    <GlassCard className="p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-muted">Prediction text</p>
          <h3 className="mt-2 text-3xl font-bold text-text">
            {prediction ? prediction.text : isLoading ? "Analyzing..." : "Waiting for input"}
          </h3>
        </div>
        <StatusPill status={prediction?.status ?? "idle"} />
      </div>
      <div className="mt-6">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="text-muted">Confidence</span>
          <span className="font-semibold text-text">{confidence}%</span>
        </div>
        <div className="h-3 rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan via-blue to-violet transition-all duration-500"
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>
      <div className="mt-6">
        <p className="mb-3 text-sm font-semibold text-text">Top 3 predictions</p>
        {prediction ? (
          <TopPredictions predictions={prediction.topPredictions.slice(0, 3)} />
        ) : (
          <p className="rounded-lg border border-white/10 bg-white/5 p-4 text-sm text-muted">
            Predictions will appear after 60 frames are buffered (~2 seconds).
          </p>
        )}
      </div>
      {prediction && (
        <div className="mt-6">
          <RealtimeStats stats={prediction.stats} />
        </div>
      )}
      {showAdd && (
        <Button className="mt-6 w-full" variant="secondary" icon={<Plus size={18} />} onClick={onAddSign}>
          Add New Sign
        </Button>
      )}
    </GlassCard>
  );
}
