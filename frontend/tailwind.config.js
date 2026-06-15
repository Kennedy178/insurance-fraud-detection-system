/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Base backgrounds
        base: {
          DEFAULT: "#080b14",
          surface: "#0f1623",
          elevated: "#162032",
          border: "rgba(255,255,255,0.06)",
          "border-active": "rgba(255,255,255,0.14)",
        },
        // Accent - trust blue
        accent: {
          DEFAULT: "#4f7ef7",
          hover: "#6b93f8",
          muted: "rgba(79,126,247,0.15)",
          dim: "rgba(79,126,247,0.08)",
        },
        // Semantic risk colors
        fraud: {
          DEFAULT: "#f04f57",
          muted: "rgba(240,79,87,0.15)",
          dim: "rgba(240,79,87,0.08)",
        },
        warn: {
          DEFAULT: "#f5a623",
          muted: "rgba(245,166,35,0.15)",
          dim: "rgba(245,166,35,0.08)",
        },
        safe: {
          DEFAULT: "#22c55e",
          muted: "rgba(34,197,94,0.15)",
          dim: "rgba(34,197,94,0.08)",
        },
        // Text
        ink: {
          primary: "#f0f4ff",
          secondary: "#7c8fa6",
          muted: "#3d4e63",
        },
        // Light mode overrides applied via dark: prefix inversion
        light: {
          base: "#f5f7ff",
          surface: "#ffffff",
          elevated: "#edf0fb",
          border: "rgba(0,0,0,0.07)",
          "border-active": "rgba(0,0,0,0.15)",
          "ink-primary": "#0d1526",
          "ink-secondary": "#4a5568",
          "ink-muted": "#94a3b8",
        },
      },
      fontFamily: {
        display: ["Sora", "sans-serif"],
        body: ["IBM Plex Sans", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "1rem" }],
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
        "card-hover": "0 4px 16px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.4)",
        accent: "0 0 24px rgba(79,126,247,0.2)",
        fraud: "0 0 24px rgba(240,79,87,0.2)",
        safe: "0 0 24px rgba(34,197,94,0.2)",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-in": "slideIn 0.3s ease-out",
        "pulse-dot": "pulseDot 2s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideIn: {
          "0%": { opacity: "0", transform: "translateX(-12px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        pulseDot: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.5", transform: "scale(0.85)" },
        },
      },
      transitionDuration: {
        "200": "200ms",
        "300": "300ms",
      },
    },
  },
  plugins: [],
};