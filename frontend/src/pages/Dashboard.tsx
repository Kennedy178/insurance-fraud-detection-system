import { useEffect, useRef } from "react";
import {
  Chart,
  LineController,
  DoughnutController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
  Legend,
  ArcElement,
  type ChartConfiguration,
} from "chart.js";
import {
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  BarChart3,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import { Link } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { getStats, getPredictions, getModelInfo } from "../api/client";
import StatCard from "../components/ui/StatCard";
import RiskBadge from "../components/ui/RiskBadge";
import ErrorMessage from "../components/ui/ErrorMessage";
import Footer from "../components/ui/Footer";
import { useTheme } from "../context/ThemeContext";
import {
  formatRate,
  formatProbability,
  formatDate,
  formatTime,
  formatInferenceMs,
} from "../utils/formatters";

Chart.register(
  LineController,
  DoughnutController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
  Legend,
  ArcElement
);

const CHART_COLORS = {
  accent: "#4f7ef7",
  fraud: "#f04f57",
  warn: "#f5a623",
  safe: "#22c55e",
  gridDark: "rgba(255,255,255,0.05)",
  gridLight: "rgba(0,0,0,0.06)",
  tickDark: "#3d4e63",
  tickLight: "#94a3b8",
};

export default function Dashboard() {
  const { isDark } = useTheme();

  const {
    data: stats,
    loading: statsLoading,
    error: statsError,
    refetch: refetchStats,
  } = useApi(getStats);

  const {
    data: predictions,
    loading: predsLoading,
    error: predsError,
    refetch: refetchPreds,
  } = useApi(() => getPredictions(10, 0));

  const { data: modelInfo, loading: modelLoading } = useApi(getModelInfo);

  const trendChartRef = useRef<HTMLCanvasElement>(null);
  const distChartRef = useRef<HTMLCanvasElement>(null);
  const trendChartInstance = useRef<Chart | null>(null);
  const distChartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!trendChartRef.current || !stats) return;
    if (trendChartInstance.current) trendChartInstance.current.destroy();

    const trend = stats.daily_trend;
    const labels = trend.map((d) => {
      const date = new Date(d.date);
      return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    });

    const legendColor = isDark ? "#f0f4ff" : "#0d1526";
    const gridColor = isDark ? CHART_COLORS.gridDark : CHART_COLORS.gridLight;
    const tickColor = isDark ? CHART_COLORS.tickDark : CHART_COLORS.tickLight;

    const config: ChartConfiguration = {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Fraud Flagged",
            data: trend.map((d) => d.fraud),
            borderColor: CHART_COLORS.fraud,
            backgroundColor: isDark ? "rgba(240,79,87,0.08)" : "rgba(240,79,87,0.06)",
            borderWidth: 2,
            pointBackgroundColor: CHART_COLORS.fraud,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.4,
            fill: true,
          },
          {
            label: "Total Claims",
            data: trend.map((d) => d.total),
            borderColor: CHART_COLORS.accent,
            backgroundColor: isDark ? "rgba(79,126,247,0.06)" : "rgba(79,126,247,0.04)",
            borderWidth: 2,
            pointBackgroundColor: CHART_COLORS.accent,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.4,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            align: "end",
            labels: {
              color: legendColor,
              font: { family: "IBM Plex Sans", size: 11 },
              boxWidth: 10,
              boxHeight: 10,
              usePointStyle: true,
              pointStyle: "circle",
              padding: 16,
            },
          },
          tooltip: {
            backgroundColor: isDark ? "#162032" : "#ffffff",
            titleColor: isDark ? "#f0f4ff" : "#0d1526",
            bodyColor: isDark ? "#7c8fa6" : "#4a5568",
            borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
            borderWidth: 1,
            padding: 10,
            titleFont: { family: "IBM Plex Sans", size: 12, weight: 600 },
            bodyFont: { family: "IBM Plex Mono", size: 11 },
          },
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: {
              color: tickColor,
              font: { family: "IBM Plex Mono", size: 10 },
              maxTicksLimit: 7,
            },
            border: { color: "transparent" },
          },
          y: {
            grid: { color: gridColor },
            ticks: {
              color: tickColor,
              font: { family: "IBM Plex Mono", size: 10 },
              precision: 0,
            },
            border: { color: "transparent" },
            beginAtZero: true,
          },
        },
      },
    };

    trendChartInstance.current = new Chart(trendChartRef.current, config);
    return () => { trendChartInstance.current?.destroy(); };
  }, [stats, isDark]);

  useEffect(() => {
    if (!distChartRef.current || !stats) return;
    if (distChartInstance.current) distChartInstance.current.destroy();

    const summary = stats.summary;
    const high = summary.high_risk_count;
    const medium = Math.max(0, summary.fraud_count - high);
    const low = summary.legitimate_count;
    const legendColor = isDark ? "#f0f4ff" : "#0d1526";

    const config: ChartConfiguration<"doughnut"> = {
      type: "doughnut",
      data: {
        labels: ["High Risk", "Medium Risk", "Low Risk"],
        datasets: [
          {
            data: [high, medium, low],
            backgroundColor: [
              "rgba(240,79,87,0.85)",
              "rgba(245,166,35,0.85)",
              "rgba(34,197,94,0.85)",
            ],
            borderColor: isDark ? "#0f1623" : "#ffffff",
            borderWidth: 3,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: legendColor,
              font: { family: "IBM Plex Sans", size: 11 },
              boxWidth: 10,
              boxHeight: 10,
              usePointStyle: true,
              pointStyle: "circle",
              padding: 14,
            },
          },
          tooltip: {
            backgroundColor: isDark ? "#162032" : "#ffffff",
            titleColor: isDark ? "#f0f4ff" : "#0d1526",
            bodyColor: isDark ? "#7c8fa6" : "#4a5568",
            borderColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
            borderWidth: 1,
            padding: 10,
            titleFont: { family: "IBM Plex Sans", size: 12, weight: 600 },
            bodyFont: { family: "IBM Plex Mono", size: 11 },
          },
        },
      },
    };

    distChartInstance.current = new Chart(distChartRef.current, config);
    return () => { distChartInstance.current?.destroy(); };
  }, [stats, isDark]);

  const summary = stats?.summary;

  return (
    <div className="space-y-4 animate-fade-in w-full min-w-0">

      {/* Stat cards - 2 cols mobile, 4 cols desktop */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
        <StatCard
          label="Total Analyzed"
          value={summary ? summary.total_predictions.toLocaleString() : "0"}
          subValue={`Last ${summary?.period_days ?? 30} days`}
          icon={<BarChart3 size={16} />}
          iconColor={CHART_COLORS.accent}
          iconBg="rgba(79,126,247,0.12)"
          loading={statsLoading}
        />
        <StatCard
          label="Fraud Rate"
          value={summary ? formatRate(summary.fraud_rate) : "0.00%"}
          subValue={`${summary?.fraud_count ?? 0} flagged`}
          icon={<TrendingUp size={16} />}
          iconColor={CHART_COLORS.fraud}
          iconBg="rgba(240,79,87,0.12)"
          loading={statsLoading}
        />
        <StatCard
          label="Fraud Flagged"
          value={summary ? summary.fraud_count.toString() : "0"}
          subValue={`Avg ${summary ? formatProbability(summary.avg_fraud_probability) : "0%"}`}
          icon={<ShieldAlert size={16} />}
          iconColor={CHART_COLORS.warn}
          iconBg="rgba(245,166,35,0.12)"
          loading={statsLoading}
        />
        <StatCard
          label="High Risk"
          value={summary ? summary.high_risk_count.toString() : "0"}
          subValue="Investigate"
          icon={<ShieldCheck size={16} />}
          iconColor={CHART_COLORS.safe}
          iconBg="rgba(34,197,94,0.12)"
          loading={statsLoading}
        />
      </div>

      {statsError && <ErrorMessage message={statsError} onRetry={refetchStats} />}

      {/* Charts - stack on mobile, side by side on desktop */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 md:gap-4">
        <div className="card p-4 md:p-5 lg:col-span-2 min-w-0">
          <div className="flex items-center justify-between mb-4">
            <div className="min-w-0">
              <h2 className="text-sm font-display font-semibold text-primary">
                Daily Fraud Trend
              </h2>
              <p className="text-xs font-body text-secondary mt-0.5 hidden sm:block">
                Claims analyzed vs fraud flagged per day
              </p>
            </div>
            <button
              onClick={refetchStats}
              className="btn-ghost text-xs px-2.5 py-1.5 flex-shrink-0"
            >
              <RefreshCw size={12} />
            </button>
          </div>
          <div className="h-48 md:h-56 relative">
            {statsLoading && (
              <div className="absolute inset-0">
                <div className="skeleton w-full h-full rounded-xl" />
              </div>
            )}
            <canvas ref={trendChartRef} />
          </div>
        </div>

        <div className="card p-4 md:p-5 min-w-0">
          <div className="mb-4">
            <h2 className="text-sm font-display font-semibold text-primary">
              Risk Distribution
            </h2>
            <p className="text-xs font-body text-secondary mt-0.5 hidden sm:block">
              Claims by risk level
            </p>
          </div>
          <div className="h-48 md:h-56 relative">
            {statsLoading && (
              <div className="absolute inset-0">
                <div className="skeleton w-full h-full rounded-xl" />
              </div>
            )}
            <canvas ref={distChartRef} />
          </div>
        </div>
      </div>

      {/* Recent predictions table */}
      <div className="card p-4 md:p-5 min-w-0">
        <div className="flex items-center justify-between mb-4 gap-3">
          <div className="min-w-0">
            <h2 className="text-sm font-display font-semibold text-primary">
              Recent Predictions
            </h2>
            <p className="text-xs font-body text-secondary mt-0.5 hidden sm:block">
              Last 10 claims analyzed
            </p>
          </div>
          <Link to="/history" className="btn-ghost text-xs px-3 py-1.5 flex-shrink-0">
            View all
            <ArrowRight size={12} />
          </Link>
        </div>

        {predsError && <ErrorMessage message={predsError} onRetry={refetchPreds} />}

        <div className="table-scroll">
          <table className="w-full text-left" style={{ minWidth: "480px" }}>
            <thead>
              <tr className="border-b divider">
                {["Date", "Risk", "Score", "Probability", "Verdict", "Time"].map((col) => (
                  <th
                    key={col}
                    className="pb-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide pr-3 last:pr-0 whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {predsLoading &&
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i}>
                    {Array.from({ length: 6 }).map((_, j) => (
                      <td key={j} className="py-3 pr-3">
                        <div className="skeleton h-4 w-16 rounded" />
                      </td>
                    ))}
                  </tr>
                ))}

              {!predsLoading && predictions && predictions.length === 0 && (
                <tr>
                  <td
                    colSpan={6}
                    className="py-10 text-center text-sm font-body text-muted"
                  >
                    No predictions yet. Analyze a claim to get started.
                  </td>
                </tr>
              )}

              {!predsLoading &&
                predictions?.map((pred) => (
                  <tr
                    key={pred.id}
                    className="hover:bg-white/[0.02] transition-colors duration-150"
                  >
                    <td className="py-3 pr-3 whitespace-nowrap">
                      <p className="text-xs font-mono text-primary">
                        {formatDate(pred.created_at)}
                      </p>
                      <p className="text-xs font-mono text-muted mt-0.5">
                        {formatTime(pred.created_at)}
                      </p>
                    </td>
                    <td className="py-3 pr-3">
                      <RiskBadge level={pred.risk_level} size="sm" />
                    </td>
                    <td className="py-3 pr-3 whitespace-nowrap">
                      <span className="text-sm font-mono font-semibold text-primary">
                        {pred.risk_score}
                      </span>
                      <span className="text-xs font-mono text-muted">/100</span>
                    </td>
                    <td className="py-3 pr-3 whitespace-nowrap">
                      <span
                        className="text-sm font-mono font-medium"
                        style={{
                          color:
                            pred.fraud_probability >= 0.7
                              ? "#f04f57"
                              : pred.fraud_probability >= 0.35
                              ? "#f5a623"
                              : "#22c55e",
                        }}
                      >
                        {formatProbability(pred.fraud_probability)}
                      </span>
                    </td>
                    <td className="py-3 pr-3 whitespace-nowrap">
                      <span
                        className={`text-xs font-mono font-semibold ${
                          pred.is_fraud ? "text-fraud" : "text-safe"
                        }`}
                      >
                        {pred.is_fraud ? "Fraud" : "Legit"}
                      </span>
                    </td>
                    <td className="py-3 whitespace-nowrap">
                      <span className="text-xs font-mono text-muted">
                        {formatInferenceMs(pred.inference_ms ?? 0)}
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model info strip */}
      {!modelLoading && modelInfo && (
        <div className="card px-4 md:px-5 py-3 min-w-0">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            {[
              { label: "Model", value: modelInfo.model_name },
              { label: "Algorithm", value: modelInfo.algorithm },
              { label: "ROC-AUC", value: modelInfo.roc_auc?.toFixed(3) ?? "0.781", accent: true },
              { label: "Threshold", value: modelInfo.deployed_threshold?.toFixed(4) ?? "0.3517" },
              { label: "Features", value: String(modelInfo.features_count ?? 76) },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-accent/50 flex-shrink-0" />
                <span className="text-xs font-mono text-secondary whitespace-nowrap">
                  {item.label}:{" "}
                  <span
                    className={
                      item.accent
                        ? "text-accent font-semibold"
                        : "text-primary font-medium"
                    }
                  >
                    {item.value}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer with real social links */}
      <Footer />
    </div>
  );
}