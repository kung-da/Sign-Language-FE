import { motion, useInView } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { MetricItem } from "../../types/recognition";
import { GlassCard } from "./GlassCard";

export function MetricCard({ metric }: { metric: MetricItem }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!inView) return;
    const duration = 900;
    const startedAt = performance.now();
    let frame = 0;

    const tick = (now: number) => {
      const progress = Math.min((now - startedAt) / duration, 1);
      setDisplay(Number((metric.value * progress).toFixed(metric.value % 1 ? 1 : 0)));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [inView, metric.value]);

  return (
    <GlassCard ref={ref} className="p-5">
      <p className="text-sm text-muted">{metric.label}</p>
      <motion.p className="mt-3 text-3xl font-bold text-text">
        {display}
        {metric.suffix}
      </motion.p>
      <p className="mt-2 text-sm text-muted">{metric.description}</p>
    </GlassCard>
  );
}
