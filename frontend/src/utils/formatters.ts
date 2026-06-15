import type { RiskLevel, ConfidenceLevel } from "../api/types";

// Returns Tailwind color class strings for a given risk level
export function getRiskColor(level: RiskLevel): {
  text: string;
  bg: string;
  border: string;
  badge: string;
} {
  switch (level) {
    case "HIGH":
      return {
        text: "text-fraud",
        bg: "bg-fraud-dim",
        border: "border-fraud/20",
        badge: "badge-high",
      };
    case "MEDIUM":
      return {
        text: "text-warn",
        bg: "bg-warn-dim",
        border: "border-warn/20",
        badge: "badge-medium",
      };
    case "LOW":
      return {
        text: "text-safe",
        bg: "bg-safe-dim",
        border: "border-safe/20",
        badge: "badge-low",
      };
  }
}

// Returns hex color string for use in Chart.js (cannot use Tailwind classes there)
export function getRiskHex(level: RiskLevel): string {
  switch (level) {
    case "HIGH":
      return "#f04f57";
    case "MEDIUM":
      return "#f5a623";
    case "LOW":
      return "#22c55e";
  }
}

// Returns a color based on a 0-100 risk score (for gradient progress bars)
export function getScoreColor(score: number): string {
  if (score >= 70) return "#f04f57";
  if (score >= 35) return "#f5a623";
  return "#22c55e";
}

// Format a fraud probability (0-1) to a percentage string
export function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

// Format a risk score (0-100) integer
export function formatScore(score: number): string {
  return score.toString().padStart(2, "0");
}

// Format fraud rate (0-1) to display percentage with 2 decimals
export function formatRate(rate: number): string {
  return `${(rate * 100).toFixed(2)}%`;
}

// Format ISO date string to readable format: "Jun 6, 2026"
export function formatDate(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

// Format ISO date string to short format: "Jun 6"
export function formatDateShort(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

// Format ISO datetime to time: "14:32"
export function formatTime(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

// Format inference time in ms
export function formatInferenceMs(ms: number | undefined | null): string {
  if (ms == null || isNaN(ms)) return "N/A";
  return `${ms.toFixed(0)}ms`;
}

// Format confidence level to title case
export function formatConfidence(level: ConfidenceLevel): string {
  return level.charAt(0).toUpperCase() + level.slice(1);
}

// Truncate a string to a max length with ellipsis
export function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max) + "...";
}

// Format a feature name from snake_case or colon-separated to readable
export function formatFeatureName(name: string): string {
  return name
    .replace(/Days:Policy-/g, "Days - Policy ")
    .replace(/AddressChange-/g, "Address Change - ")
    .replace(/_/g, " ")
    .replace(/([A-Z])/g, " $1")
    .trim()
    .replace(/\s+/g, " ");
}

// Compute percentage width for importance bars (0-1 -> 0-100)
export function importanceToWidth(importance: number): string {
  const pct = Math.min(Math.max(importance * 100, 2), 100);
  return `${pct.toFixed(1)}%`;
}