import { ArrowRight } from "lucide-react";
import { GlassCard } from "../ui/GlassCard";
import { SectionHeading } from "../ui/SectionHeading";

const nodes = [
  "Webcam / Video Input",
  "MediaPipe",
  "Sequence Buffer",
  "LSTM / GRU / Transformer",
  "Embedding Vector",
  "Vector Database",
  "Similarity Search",
  "NLP / Output",
];

export function ArchitectureSection() {
  return (
    <section id="architecture" className="section-container">
      <SectionHeading
        eyebrow="Architecture"
        title="Backend-ready recognition flow"
        description="These nodes represent where the future Python/FastAPI backend, model inference, and vector storage will connect."
      />
      <div className="grid items-center gap-3 lg:flex">
        {nodes.map((node, index) => (
          <div key={node} className="contents">
            <GlassCard className="p-4 text-center lg:flex-1">
              <p className="text-sm font-semibold text-text">{node}</p>
            </GlassCard>
            {index < nodes.length - 1 && (
              <div className="grid place-items-center text-cyan lg:flex-none">
                <ArrowRight className="hidden lg:block" size={20} />
                <div className="h-6 w-px bg-cyan/40 lg:hidden" />
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
