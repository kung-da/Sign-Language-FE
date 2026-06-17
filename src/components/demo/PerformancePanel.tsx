import type { PipelinePerformanceMetrics } from "../../hooks/useMediaPipeLandmarks";
import { GlassCard } from "../ui/GlassCard";

interface PerformancePanelProps {
  metrics: PipelinePerformanceMetrics;
}

export function PerformancePanel({ metrics }: PerformancePanelProps) {
  return (
    <GlassCard className="p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="font-semibold text-text">Pipeline Performance</p>
          <p className="text-sm text-muted">Measured in browser while webcam is running.</p>
        </div>
        <span className="rounded-md border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold text-muted">
          MediaPipe real-time
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <Metric label="MediaPipe extraction" value={formatMs(metrics.extractionTimeMs)} note="Max hand/face/pose worker time" />
        <Metric label="Model inference" value={formatMs(metrics.modelInferenceTimeMs)} note="Waiting for real sign model" />
        <Metric label="End-to-end latency" value={formatMs(metrics.endToEndLatencyMs)} note="Frame capture to landmark overlay" />
        <Metric label="Pipeline FPS" value={metrics.fps ? `${metrics.fps} fps` : "Waiting"} note="Completed landmark frames/sec" />
        <Metric label="Model size" value={formatBytes(metrics.inferenceSize?.totalBytes ?? null)} note="Local MediaPipe task assets" />
        <Metric label="JS heap RAM" value={formatMemory(metrics.memory)} note="Available in Chromium browsers" />
        <Metric label="CPU usage" value="Not available" note="Use DevTools or Task Manager" />
        <Metric label="GPU usage" value="Not available" note="Use browser/GPU profiler" />
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
          <p className="text-sm font-semibold text-text">Per-task extraction</p>
          <div className="mt-3 grid grid-cols-3 gap-2 text-sm">
            <MiniMetric label="Hand" value={formatMs(metrics.taskTimesMs.hand)} />
            <MiniMetric label="Face" value={formatMs(metrics.taskTimesMs.face)} />
            <MiniMetric label="Pose" value={formatMs(metrics.taskTimesMs.pose)} />
          </div>
        </div>

        <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
          <p className="text-sm font-semibold text-text">Smoothing / voting readiness</p>
          <p className="mt-2 text-sm text-muted">
            Recommended majority vote window: {metrics.smoothingWindowSize} predictions. Real evaluation starts when
            the sign classifier replaces mock predictions.
          </p>
        </div>
      </div>

      {metrics.inferenceSize?.items.length ? (
        <div className="mt-4 rounded-md border border-white/10 bg-white/[0.04] p-3">
          <p className="text-sm font-semibold text-text">Model asset breakdown</p>
          <div className="mt-2 grid gap-2 text-sm text-muted sm:grid-cols-3">
            {metrics.inferenceSize.items.map((item) => (
              <span key={item.label}>
                {item.label}: <span className="font-semibold text-text">{formatBytes(item.bytes)}</span>
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </GlassCard>
  );
}

function Metric({ label, note, value }: { label: string; note: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
      <p className="text-xs font-semibold uppercase tracking-normal text-muted">{label}</p>
      <p className="mt-2 text-xl font-bold text-text">{value}</p>
      <p className="mt-1 text-xs text-muted">{note}</p>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted">{label}</p>
      <p className="mt-1 font-semibold text-text">{value}</p>
    </div>
  );
}

function formatMs(value: number | null) {
  return value === null ? "Not measured" : `${value} ms`;
}

function formatBytes(value: number | null) {
  if (value === null) return "Not measured";
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}

function formatMemory(memory: PipelinePerformanceMetrics["memory"]) {
  if (!memory) return "Not available";
  return `${formatBytes(memory.usedJSHeapSize)} / ${formatBytes(memory.totalJSHeapSize)}`;
}
