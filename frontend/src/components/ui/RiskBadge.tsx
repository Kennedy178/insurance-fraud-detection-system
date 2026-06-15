import type { RiskLevel } from "../../api/types";
import { getRiskColor } from "../../utils/formatters";

interface RiskBadgeProps {
  level: RiskLevel;
  size?: "sm" | "md";
}

export default function RiskBadge({ level, size = "md" }: RiskBadgeProps) {
  const colors = getRiskColor(level);

  const sizeClass = size === "sm"
    ? "px-2 py-0.5 text-2xs"
    : "px-2.5 py-1 text-xs";

  return (
    <span className={`${colors.badge} ${sizeClass} font-mono font-semibold`}>
      {level}
    </span>
  );
}