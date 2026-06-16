import {
  Activity,
  BrainCircuit,
  Camera,
  Database,
  Layers3,
  MessageSquareText,
  ScanSearch,
  SlidersHorizontal,
} from "lucide-react";
import { pipelineSteps } from "../../data/pipelineSteps";
import { GlassCard } from "../ui/GlassCard";
import { SectionHeading } from "../ui/SectionHeading";

const iconMap = {
  Camera,
  SlidersHorizontal,
  ScanSearch,
  Layers3,
  BrainCircuit,
  Database,
  Activity,
  MessageSquareText,
};

export function PipelineSection() {
  return (
    <section id="pipeline" className="section-container">
      <SectionHeading
        eyebrow="AI Pipeline"
        title="From motion frames to Vietnamese text"
        description="The frontend mirrors the real AI flow so the system can be explained clearly during an academic demo."
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {pipelineSteps.map((step) => {
          const Icon = iconMap[step.icon as keyof typeof iconMap];
          return (
            <GlassCard key={step.id} className="group p-5 transition hover:-translate-y-1 hover:border-cyan/40">
              <div className="mb-4 grid h-11 w-11 place-items-center rounded-lg bg-cyan/10 text-cyan group-hover:bg-cyan group-hover:text-slate-950">
                <Icon size={22} />
              </div>
              <h3 className="font-bold text-text">{step.title}</h3>
              <p className="mt-2 text-sm leading-6 text-muted">{step.description}</p>
            </GlassCard>
          );
        })}
      </div>
    </section>
  );
}
