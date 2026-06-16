import { CheckCircle2 } from "lucide-react";
import { features } from "../../data/features";
import { GlassCard } from "../ui/GlassCard";
import { SectionHeading } from "../ui/SectionHeading";

export function FeaturesSection() {
  return (
    <section id="features" className="section-container">
      <SectionHeading
        eyebrow="Features"
        title="Built for a convincing technical demo"
        description="The interface highlights what exists today as mock frontend behavior and what can become real with backend integration."
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {features.map((feature) => (
          <GlassCard key={feature} className="flex items-start gap-3 p-4">
            <CheckCircle2 className="mt-0.5 shrink-0 text-success" size={20} />
            <p className="text-sm font-medium text-text">{feature}</p>
          </GlassCard>
        ))}
      </div>
    </section>
  );
}
