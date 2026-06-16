interface SectionHeadingProps {
  eyebrow: string;
  title: string;
  description: string;
}

export function SectionHeading({ eyebrow, title, description }: SectionHeadingProps) {
  return (
    <div className="mx-auto mb-10 max-w-3xl text-center">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-cyan">{eyebrow}</p>
      <h2 className="mt-3 text-3xl font-bold text-text sm:text-4xl">{title}</h2>
      <p className="mt-4 text-base leading-7 text-muted">{description}</p>
    </div>
  );
}
