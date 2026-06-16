import { metrics } from "../../data/metrics";
import { MetricCard } from "../ui/MetricCard";
import { SectionHeading } from "../ui/SectionHeading";

export function MetricsSection() {
  return (
    <section id="metrics" className="section-container">
      <SectionHeading
        eyebrow="Demo Metrics"
        title="Readable model-performance dashboard"
        description="Numbers are generated for presentation only and can later be replaced by real evaluation results."
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <MetricCard key={metric.label} metric={metric} />
        ))}
      </div>
    </section>
  );
}
