import { useLocation } from "react-router-dom";
import { useTheme } from "../../context/ThemeContext";

const ROUTE_LABELS: Record<string, string> = {
  "/": "Dashboard",
  "/analyze": "Analyze Claim",
  "/history": "Claims History",
};

const ROUTE_DESCRIPTIONS: Record<string, string> = {
  "/": "Fraud analytics overview and recent activity",
  "/analyze": "Submit a claim for real-time fraud analysis",
  "/history": "Browse and search all analyzed claims",
};

interface TopbarProps {
  apiOnline: boolean;
}

export default function Topbar({ apiOnline }: TopbarProps) {
  const location = useLocation();
  const { isDark } = useTheme();
  const label = ROUTE_LABELS[location.pathname] ?? "FraudGuard";
  const description = ROUTE_DESCRIPTIONS[location.pathname] ?? "";

  // Explicit color bypasses any CSS class layering issues on mobile dark mode
  const titleColor = isDark ? "#f0f4ff" : "#0d1526";

  return (
    <header className="topbar-bg sticky top-0 z-20 px-4 md:px-6 py-3 md:py-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex-1 min-w-0">

          {/* App name - desktop only, small muted label above title */}
          <p className="hidden md:block text-xs font-body text-secondary mb-1">
            FraudGuard
          </p>

          {/* Page title - inline color so it is never absorbed by backdrop */}
          <h1
            className="text-base md:text-lg font-display font-semibold leading-tight truncate"
            style={{ color: titleColor }}
          >
            {label}
          </h1>

          {/* Description - desktop only */}
          {description && (
            <p className="hidden md:block text-xs font-body text-secondary mt-0.5 truncate">
              {description}
            </p>
          )}
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3 flex-shrink-0">
          {/* Mobile API dot */}
          <span
            className="md:hidden w-2 h-2 rounded-full flex-shrink-0"
            style={{ background: apiOnline ? "#22c55e" : "#f04f57" }}
            title={apiOnline ? "API Online" : "API Offline"}
          />

          {/* Desktop model badge */}
          <div className="hidden md:flex items-center gap-2 px-3 py-2 rounded-xl flex-shrink-0 card">
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0 bg-accent" />
            <span className="text-xs font-mono font-medium text-accent">
              XGBoost v1.0.0
            </span>
            <span className="text-xs font-mono text-secondary">
              ROC-AUC 0.781
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}