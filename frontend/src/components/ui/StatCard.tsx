import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string;
  subValue?: string;
  icon: ReactNode;
  iconColor: string;
  iconBg: string;
  trend?: {
    value: string;
    positive: boolean;
  };
  loading?: boolean;
}

export default function StatCard({
  label,
  value,
  subValue,
  icon,
  iconColor,
  iconBg,
  trend,
  loading = false,
}: StatCardProps) {
  if (loading) {
    return (
      <div className="card p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 space-y-2">
            <div className="skeleton h-3 w-24 rounded" />
            <div className="skeleton h-7 w-16 rounded" />
            <div className="skeleton h-2.5 w-20 rounded" />
          </div>
          <div className="skeleton w-10 h-10 rounded-xl" />
        </div>
      </div>
    );
  }

  return (
    <div className="card p-5 hover:shadow-card-hover transition-all duration-200 group">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-body font-medium text-secondary uppercase tracking-wide mb-2">
            {label}
          </p>
          <p className="text-2xl font-display font-bold text-primary leading-none">
            {value}
          </p>
          {subValue && (
            <p className="text-xs font-mono text-secondary mt-1.5">{subValue}</p>
          )}
          {trend && (
            <div className="flex items-center gap-1 mt-2">
              <span
                className={`text-xs font-mono font-medium ${
                  trend.positive ? "text-safe" : "text-fraud"
                }`}
              >
                {trend.positive ? "+" : ""}
                {trend.value}
              </span>
              <span className="text-xs text-muted">vs last period</span>
            </div>
          )}
        </div>
        {/* Icon */}
        <div
          className="flex items-center justify-center w-10 h-10 rounded-xl flex-shrink-0 transition-transform duration-200 group-hover:scale-110"
          style={{ background: iconBg }}
        >
          <span style={{ color: iconColor }}>{icon}</span>
        </div>
      </div>
    </div>
  );
}