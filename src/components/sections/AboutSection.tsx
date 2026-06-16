import { HeartHandshake } from "lucide-react";
import { GlassCard } from "../ui/GlassCard";
import { SectionHeading } from "../ui/SectionHeading";

export function AboutSection() {
  return (
    <section id="about" className="section-container">
      <SectionHeading
        eyebrow="Social Impact"
        title="Accessible communication for everyday settings"
        description="This frontend is designed for learning, presentation, and future research iteration."
      />
      <GlassCard className="mx-auto max-w-4xl p-6 sm:p-8">
        <div className="flex flex-col gap-5 sm:flex-row">
          <div className="grid h-14 w-14 shrink-0 place-items-center rounded-lg bg-violet/15 text-violet">
            <HeartHandshake size={28} />
          </div>
          <div>
            <h3 className="text-2xl font-bold text-text">Vietnamese Sign Language support</h3>
            <p className="mt-4 leading-8 text-muted">
              The project aims to support communication between deaf or hard-of-hearing people and people who do not
              know sign language. Future versions could serve learning, healthcare, public services, and daily
              conversations by combining realtime recognition, Vietnamese language output, and assistive interfaces.
            </p>
          </div>
        </div>
      </GlassCard>
    </section>
  );
}
