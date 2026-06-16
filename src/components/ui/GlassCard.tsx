import { forwardRef, type HTMLAttributes } from "react";
import { cn } from "../../utils/cn";

export const GlassCard = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("glass-card rounded-lg", className)} {...props} />
  ),
);

GlassCard.displayName = "GlassCard";
