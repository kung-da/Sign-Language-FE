import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        card: "hsl(var(--card))",
        border: "hsl(var(--border))",
        text: "hsl(var(--text))",
        muted: "hsl(var(--muted))",
        cyan: "hsl(var(--cyan))",
        blue: "hsl(var(--blue))",
        violet: "hsl(var(--violet))",
        success: "hsl(var(--success))",
        danger: "hsl(var(--danger))",
      },
      boxShadow: {
        glow: "0 0 32px rgba(34, 211, 238, 0.16)",
      },
      animation: {
        scan: "scan 2.5s linear infinite",
        pulseDot: "pulseDot 1.7s ease-in-out infinite",
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-20%)", opacity: "0" },
          "15%": { opacity: "1" },
          "100%": { transform: "translateY(1050%)", opacity: "0" },
        },
        pulseDot: {
          "0%, 100%": { transform: "scale(0.9)", opacity: "0.65" },
          "50%": { transform: "scale(1.3)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
