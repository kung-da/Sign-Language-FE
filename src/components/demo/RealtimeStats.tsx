import type { RealtimeStats as RealtimeStatsType } from "../../types/recognition";
import { StatusPill } from "../ui/StatusPill";

export function RealtimeStats({ stats }: { stats: RealtimeStatsType }) {
  return (
    <div className="grid grid-cols-3 gap-3">
      <div className="rounded-lg border border-white/10 bg-white/5 p-3">
        <p className="text-xs text-muted">FPS</p>
        <p className="mt-1 text-lg font-bold text-text">{stats.fps}</p>
      </div>
      <div className="rounded-lg border border-white/10 bg-white/5 p-3">
        <p className="text-xs text-muted">Latency</p>
        <p className="mt-1 text-lg font-bold text-text">{stats.latencyMs}ms</p>
      </div>
      <div className="rounded-lg border border-white/10 bg-white/5 p-3">
        <p className="text-xs text-muted">Model</p>
        <div className="mt-1">
          <StatusPill status={stats.modelStatus} />
        </div>
      </div>
    </div>
  );
}
