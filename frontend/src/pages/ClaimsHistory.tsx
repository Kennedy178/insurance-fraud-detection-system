import { useState, useMemo, useCallback, Fragment } from "react";
import {
  Search,
  ChevronUp,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsUpDown,
  RefreshCw,
  X,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import { getPredictions } from "../api/client";
import RiskBadge from "../components/ui/RiskBadge";
import ErrorMessage from "../components/ui/ErrorMessage";
import Footer from "../components/ui/Footer";
import type { PredictionRecord, RiskLevel } from "../api/types";
import {
  formatDate,
  formatTime,
  formatProbability,
  formatInferenceMs,
  formatFeatureName,
  importanceToWidth,
} from "../utils/formatters";

const PAGE_SIZE = 20;

type SortKey = "created_at" | "risk_score" | "fraud_probability" | "risk_level";
type SortDir = "asc" | "desc";

const RISK_FILTER_OPTIONS: { label: string; value: string }[] = [
  { label: "All Levels", value: "" },
  { label: "High", value: "HIGH" },
  { label: "Medium", value: "MEDIUM" },
  { label: "Low", value: "LOW" },
];

const VERDICT_FILTER_OPTIONS: { label: string; value: string }[] = [
  { label: "All Verdicts", value: "" },
  { label: "Fraud", value: "fraud" },
  { label: "Legitimate", value: "legitimate" },
];

function SortIcon({
  column,
  sortKey,
  sortDir,
}: {
  column: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
}) {
  if (sortKey !== column) {
    return <ChevronsUpDown size={12} className="text-muted" />;
  }
  return sortDir === "asc" ? (
    <ChevronUp size={12} className="text-accent" />
  ) : (
    <ChevronDown size={12} className="text-accent" />
  );
}

function RowDetail({ pred }: { pred: PredictionRecord }) {
  return (
    <tr>
      <td colSpan={7} className="px-5 pb-4 pt-0">
        <div
          className="rounded-xl p-4 space-y-4"
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          {/* Verdict + confidence row */}
          <div className="flex items-center gap-3 flex-wrap">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-mono font-semibold ${
                pred.is_fraud
                  ? "bg-fraud-muted text-fraud border border-fraud/20"
                  : "bg-safe-muted text-safe border border-safe/20"
              }`}
            >
              {pred.is_fraud ? (
                <ShieldAlert size={12} />
              ) : (
                <ShieldCheck size={12} />
              )}
              {pred.is_fraud ? "Fraud Detected" : "Likely Legitimate"}
            </span>
            <span className="text-xs font-mono text-secondary">
              Confidence:{" "}
              <span className="text-primary font-medium capitalize">
                {pred.confidence}
              </span>
            </span>
            <span className="text-xs font-mono text-secondary">
              Inference:{" "}
              <span className="text-primary font-medium">
                {formatInferenceMs(pred.inference_ms)}
              </span>
            </span>
          </div>

          {/* Recommendation */}
          {pred.recommendation && (
            <div>
              <p className="text-xs font-body font-semibold text-secondary uppercase tracking-wide mb-1">
                Recommendation
              </p>
              <p className="text-xs font-body text-primary leading-relaxed">
                {pred.recommendation}
              </p>
            </div>
          )}

          {/* Risk factors */}
          {pred.risk_factors && pred.risk_factors.length > 0 && (
            <div>
              <p className="text-xs font-body font-semibold text-secondary uppercase tracking-wide mb-2">
                Top Risk Factors
              </p>
              <div className="space-y-2">
                {pred.risk_factors.slice(0, 5).map((factor, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between gap-4 mb-1">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-xs font-mono text-muted w-4 flex-shrink-0">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span className="text-xs font-body text-primary truncate">
                          {formatFeatureName(factor.feature)}
                        </span>
                      </div>
                      <span className="text-xs font-mono font-semibold text-accent flex-shrink-0">
                        {(factor.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="ml-6 h-1 rounded-full bg-white/[0.05] overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: importanceToWidth(factor.importance),
                          background:
                            i === 0
                              ? "#f04f57"
                              : i <= 2
                              ? "#f5a623"
                              : "#4f7ef7",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </td>
    </tr>
  );
}

export default function ClaimsHistory() {
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState<string>("");
  const [verdictFilter, setVerdictFilter] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const { data: allPredictions, loading, error, refetch } = useApi(
    () => getPredictions(200, 0)
  );

  const handleSort = useCallback((key: SortKey) => {
    setSortKey((prev) => {
      if (prev === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        return key;
      }
      setSortDir("desc");
      return key;
    });
    setPage(1);
  }, []);

  const handleSearch = (val: string) => {
    setSearch(val);
    setPage(1);
  };

  const handleRiskFilter = (val: string) => {
    setRiskFilter(val);
    setPage(1);
  };

  const handleVerdictFilter = (val: string) => {
    setVerdictFilter(val);
    setPage(1);
  };

  const clearFilters = () => {
    setSearch("");
    setRiskFilter("");
    setVerdictFilter("");
    setPage(1);
  };

  const hasActiveFilters = search || riskFilter || verdictFilter;

  const filtered = useMemo(() => {
    if (!allPredictions) return [];

    let rows = [...allPredictions];

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter(
        (p) =>
          String(p.id).includes(q) ||
          p.risk_level.toLowerCase().includes(q) ||
          (p.is_fraud ? "fraud" : "legitimate").includes(q) ||
          p.recommendation?.toLowerCase().includes(q)
      );
    }

    if (riskFilter) {
      rows = rows.filter((p) => p.risk_level === riskFilter);
    }

    if (verdictFilter) {
      rows = rows.filter((p) =>
        verdictFilter === "fraud" ? p.is_fraud : !p.is_fraud
      );
    }

    rows.sort((a, b) => {
      let aVal: string | number;
      let bVal: string | number;

      switch (sortKey) {
        case "created_at":
          aVal = new Date(a.created_at).getTime();
          bVal = new Date(b.created_at).getTime();
          break;
        case "risk_score":
          aVal = a.risk_score;
          bVal = b.risk_score;
          break;
        case "fraud_probability":
          aVal = a.fraud_probability;
          bVal = b.fraud_probability;
          break;
        case "risk_level": {
          const order: Record<RiskLevel, number> = {
            HIGH: 3,
            MEDIUM: 2,
            LOW: 1,
          };
          aVal = order[a.risk_level];
          bVal = order[b.risk_level];
          break;
        }
      }

      if (aVal < bVal) return sortDir === "asc" ? -1 : 1;
      if (aVal > bVal) return sortDir === "asc" ? 1 : -1;
      return 0;
    });

    return rows;
  }, [allPredictions, search, riskFilter, verdictFilter, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageRows = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const toggleExpand = (id: number) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Filters row */}
      <div className="card p-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 flex-wrap">
          {/* Search */}
          <div className="relative flex-1 min-w-48">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
            />
            <input
              type="text"
              placeholder="Search by ID, risk level, verdict..."
              value={search}
              onChange={(e) => handleSearch(e.target.value)}
              className="input-field pl-9 text-xs"
            />
            {search && (
              <button
                onClick={() => handleSearch("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-primary transition-colors"
              >
                <X size={12} />
              </button>
            )}
          </div>

          {/* Risk level filter chips */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {RISK_FILTER_OPTIONS.map((opt) => (
              <button
                key={opt.value || "all-risk"}
                onClick={() => handleRiskFilter(opt.value)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all duration-150 ${
                  riskFilter === opt.value ? "bg-accent text-white" : "btn-ghost"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Verdict filter chips */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {VERDICT_FILTER_OPTIONS.map((opt) => (
              <button
                key={opt.value || "all-verdict"}
                onClick={() => handleVerdictFilter(opt.value)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all duration-150 ${
                  verdictFilter === opt.value
                    ? opt.value === "fraud"
                      ? "bg-fraud text-white"
                      : opt.value === "legitimate"
                      ? "bg-safe text-white"
                      : "bg-accent text-white"
                    : "btn-ghost"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Clear + refresh */}
          <div className="flex items-center gap-2 ml-auto">
            {hasActiveFilters && (
              <button onClick={clearFilters} className="btn-ghost text-xs">
                <X size={12} />
                Clear
              </button>
            )}
            <button
              onClick={refetch}
              className="btn-ghost text-xs"
              title="Refresh"
            >
              <RefreshCw size={12} />
            </button>
          </div>
        </div>

        {/* Result count */}
        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs font-mono text-muted">
            {loading
              ? "Loading..."
              : `${filtered.length} claim${filtered.length !== 1 ? "s" : ""} found`}
          </span>
          {hasActiveFilters && !loading && (
            <span className="text-xs font-mono text-secondary">
              (filtered from {allPredictions?.length ?? 0} total)
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && <ErrorMessage message={error} onRetry={refetch} />}

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b divider">
                <th className="px-5 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide w-8" />
                <th className="px-3 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide">
                  ID
                </th>
                <th
                  className="px-3 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide cursor-pointer hover:text-primary transition-colors select-none"
                  onClick={() => handleSort("created_at")}
                >
                  <div className="flex items-center gap-1.5">
                    Date
                    <SortIcon
                      column="created_at"
                      sortKey={sortKey}
                      sortDir={sortDir}
                    />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide cursor-pointer hover:text-primary transition-colors select-none"
                  onClick={() => handleSort("risk_level")}
                >
                  <div className="flex items-center gap-1.5">
                    Risk Level
                    <SortIcon
                      column="risk_level"
                      sortKey={sortKey}
                      sortDir={sortDir}
                    />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide cursor-pointer hover:text-primary transition-colors select-none"
                  onClick={() => handleSort("risk_score")}
                >
                  <div className="flex items-center gap-1.5">
                    Score
                    <SortIcon
                      column="risk_score"
                      sortKey={sortKey}
                      sortDir={sortDir}
                    />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide cursor-pointer hover:text-primary transition-colors select-none"
                  onClick={() => handleSort("fraud_probability")}
                >
                  <div className="flex items-center gap-1.5">
                    Probability
                    <SortIcon
                      column="fraud_probability"
                      sortKey={sortKey}
                      sortDir={sortDir}
                    />
                  </div>
                </th>
                <th className="px-3 py-3 text-xs font-body font-semibold text-secondary uppercase tracking-wide">
                  Verdict
                </th>
              </tr>
            </thead>
            <tbody>
              {/* Loading skeletons */}
              {loading &&
                Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i} className="border-b divider last:border-0">
                    <td className="px-5 py-3">
                      <div className="skeleton h-4 w-4 rounded" />
                    </td>
                    {Array.from({ length: 6 }).map((_, j) => (
                      <td key={j} className="px-3 py-3">
                        <div className="skeleton h-4 w-20 rounded" />
                      </td>
                    ))}
                  </tr>
                ))}

              {/* Empty state */}
              {!loading && pageRows.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-5 py-14 text-center">
                    <div className="flex flex-col items-center gap-3">
                      {hasActiveFilters ? (
                        <>
                          <AlertTriangle size={20} className="text-muted" />
                          <p className="text-sm font-body text-secondary">
                            No claims match your filters
                          </p>
                          <button
                            onClick={clearFilters}
                            className="btn-ghost text-xs"
                          >
                            <X size={12} />
                            Clear filters
                          </button>
                        </>
                      ) : (
                        <>
                          <ShieldCheck size={20} className="text-muted" />
                          <p className="text-sm font-body text-secondary">
                            No predictions yet
                          </p>
                          <p className="text-xs font-body text-muted">
                            Analyze a claim to see it appear here
                          </p>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              )}

              {/* Data rows - Fragment with key fixes the console warning without nesting tbody */}
              {!loading &&
                pageRows.map((pred) => {
                  const isExpanded = expandedId === pred.id;
                  return (
                    <Fragment key={pred.id}>
                      <tr
                        className={`border-b divider last:border-0 cursor-pointer transition-colors duration-150 ${
                          isExpanded
                            ? "bg-accent-dim"
                            : "hover:bg-white/[0.02]"
                        }`}
                        onClick={() => toggleExpand(pred.id)}
                      >
                        {/* Expand chevron */}
                        <td className="px-5 py-3">
                          <span className="text-muted">
                            {isExpanded ? (
                              <ChevronUp size={13} />
                            ) : (
                              <ChevronDown size={13} />
                            )}
                          </span>
                        </td>
                        {/* ID */}
                        <td className="px-3 py-3">
                          <span className="text-xs font-mono text-muted">
                            #{String(pred.id).padStart(4, "0")}
                          </span>
                        </td>
                        {/* Date */}
                        <td className="px-3 py-3">
                          <div>
                            <p className="text-xs font-mono text-primary">
                              {formatDate(pred.created_at)}
                            </p>
                            <p className="text-xs font-mono text-muted mt-0.5">
                              {formatTime(pred.created_at)}
                            </p>
                          </div>
                        </td>
                        {/* Risk level */}
                        <td className="px-3 py-3">
                          <RiskBadge level={pred.risk_level} size="sm" />
                        </td>
                        {/* Score */}
                        <td className="px-3 py-3">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-mono font-semibold text-primary">
                              {pred.risk_score}
                            </span>
                            <div className="w-14 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${pred.risk_score}%`,
                                  background:
                                    pred.risk_level === "HIGH"
                                      ? "#f04f57"
                                      : pred.risk_level === "MEDIUM"
                                      ? "#f5a623"
                                      : "#22c55e",
                                }}
                              />
                            </div>
                          </div>
                        </td>
                        {/* Probability */}
                        <td className="px-3 py-3">
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
                        {/* Verdict */}
                        <td className="px-3 py-3">
                          <span
                            className={`text-xs font-mono font-semibold ${
                              pred.is_fraud ? "text-fraud" : "text-safe"
                            }`}
                          >
                            {pred.is_fraud ? "Fraud" : "Legitimate"}
                          </span>
                        </td>
                      </tr>

                      {/* Expanded detail row */}
                      {isExpanded && <RowDetail pred={pred} />}
                    </Fragment>
                  );
                })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {!loading && filtered.length > PAGE_SIZE && (
          <div className="px-5 py-3 border-t divider flex items-center justify-between gap-4 flex-wrap">
            <span className="text-xs font-mono text-muted">
              Page {page} of {totalPages} - {filtered.length} results
            </span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setPage(1)}
                disabled={page === 1}
                className="btn-ghost text-xs px-2.5 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                First
              </button>
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="btn-ghost text-xs px-2.5 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronLeft size={12} />
              </button>

              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                const pageNum =
                  totalPages <= 5
                    ? i + 1
                    : page <= 3
                    ? i + 1
                    : page >= totalPages - 2
                    ? totalPages - 4 + i
                    : page - 2 + i;
                return (
                  <button
                    key={pageNum}
                    onClick={() => setPage(pageNum)}
                    className={`w-8 h-8 rounded-lg text-xs font-mono transition-all duration-150 ${
                      page === pageNum ? "bg-accent text-white" : "btn-ghost"
                    }`}
                  >
                    {pageNum}
                  </button>
                );
              })}

              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="btn-ghost text-xs px-2.5 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronRight size={12} />
              </button>
              <button
                onClick={() => setPage(totalPages)}
                disabled={page === totalPages}
                className="btn-ghost text-xs px-2.5 py-1.5 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Last
              </button>
            </div>
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
}