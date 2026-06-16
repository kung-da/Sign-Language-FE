import { cn } from "../../utils/cn";
import type { RecognitionStatus } from "../../types/recognition";

const styles: Record<RecognitionStatus, string> = {
  idle: "border-white/10 bg-white/10 text-muted",
  running: "border-success/30 bg-success/10 text-success",
  paused: "border-blue/30 bg-blue/10 text-blue",
  unknown: "border-danger/30 bg-danger/10 text-danger",
  error: "border-danger/30 bg-danger/10 text-danger",
};

export function StatusPill({ status }: { status: RecognitionStatus }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold capitalize",
        styles[status],
      )}
    >
      {status}
    </span>
  );
}
