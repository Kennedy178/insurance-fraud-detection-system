import { useState } from "react";
import {
  ScanSearch,
  Sparkles,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Loader2,
  RotateCcw,
  Info,
} from "lucide-react";
import { predictClaim } from "../api/client";
import type { PredictionResponse, ClaimInput, RiskLevel } from "../api/types";
import { getRiskColor, formatProbability, importanceToWidth, formatFeatureName } from "../utils/formatters";
import Footer from "../components/ui/Footer";

// Demo case that returns is_fraud=true, probability=0.41, risk_level=MEDIUM
const DEMO_CASE: ClaimInput = {
  Month: "Jun",
  WeekOfMonth: 1,
  DayOfWeek: "Thursday",
  MonthClaimed: "Jun",
  WeekOfMonthClaimed: 1,
  DayOfWeekClaimed: "Thursday",
  Sex: "Male",
  MaritalStatus: "Married",
  Age: 28,
  DriverRating: 2,
  Make: "Honda",
  VehicleCategory: "Sedan",
  VehiclePrice: "20,000 to 29,000",
  AgeOfVehicle: "3 years",
  AgeOfPolicyHolder: "26 to 30",
  PolicyType: "Sedan - Collision",
  BasePolicy: "Collision",
  Deductible: 400,
  AccidentArea: "Rural",
  Fault: "Policy Holder",
  AgentType: "External",
  PoliceReportFiled: "No",
  WitnessPresent: "No",
  PastNumberOfClaims: "none",
  NumberOfSuppliments: "more than 5",
  NumberOfCars: "1 vehicle",
  "Days:Policy-Accident": "8 to 15",
  "Days:Policy-Claim": "8 to 15",
  "AddressChange-Claim": "1 year",
};

// Dropdown option lists
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const DAYS_OF_WEEK = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
const MAKES = ["Accura","BMW","Chevrolet","Dodge","Ferrari","Ford","Honda","Jaguar","Lexus","Mazda","Mecedes","Mercury","Nisson","Pontiac","Porche","Saab","Saturn","Toyota","VW"];
const VEHICLE_CATEGORIES = ["Sedan","Sport","Utility"];
const VEHICLE_PRICES = ["less than 20,000","20,000 to 29,000","30,000 to 39,000","40,000 to 59,000","60,000 to 69,000","more than 69,000"];
const AGE_OF_VEHICLE = ["new","2 years","3 years","4 years","5 years","6 years","7 years","more than 7"];
const AGE_OF_POLICY_HOLDER = ["16 to 17","18 to 20","21 to 25","26 to 30","31 to 35","36 to 40","41 to 50","51 to 65","over 65"];
const POLICY_TYPES = ["Sedan - All Perils","Sedan - Collision","Sedan - Liability","Sport - All Perils","Sport - Collision","Sport - Liability","Utility - All Perils","Utility - Collision","Utility - Liability"];
const BASE_POLICIES = ["All Perils","Collision","Liability"];
const ACCIDENT_AREAS = ["Rural","Urban"];
const FAULTS = ["Policy Holder","Third Party"];
const AGENT_TYPES = ["External","Internal"];
const YES_NO = ["Yes","No"];
const PAST_CLAIMS = ["none","1","2","3","more than 3"];
const NUM_SUPPLIMENTS = ["none","1 to 2","3 to 5","more than 5"];
const NUM_CARS = ["1 vehicle","2 vehicles","3 to 4","more than 4"];
const DAYS_RANGES = ["none","1 to 7","8 to 15","15 to 30","more than 30"];
const ADDRESS_CHANGE = ["no change","under 6 months","1 year","2 to 3 years","4 to 8 years"];
const DRIVER_RATINGS = [1, 2, 3, 4];
const DEDUCTIBLES = [300, 400, 500, 700];
const WEEK_NUMS = [1, 2, 3, 4, 5];

// Section toggle state type
type SectionKey = "policy" | "vehicle" | "incident" | "claimant";

// Risk score gauge rendered as SVG arc
function RiskGauge({ score, level }: { score: number; level: RiskLevel }) {
  const colors = getRiskColor(level);
  const radius = 54;
  const cx = 70;
  const cy = 70;
  const strokeWidth = 10;
  // Arc goes from 210deg to 330deg (240deg sweep) - bottom-left to bottom-right
  const startAngle = 210;
  const sweepAngle = 240;
  const endAngle = startAngle + (sweepAngle * score) / 100;

  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const arcX = (deg: number) => cx + radius * Math.cos(toRad(deg));
  const arcY = (deg: number) => cy + radius * Math.sin(toRad(deg));

  const trackEnd = startAngle + sweepAngle;
  const trackPath = `M ${arcX(startAngle)} ${arcY(startAngle)} A ${radius} ${radius} 0 1 1 ${arcX(trackEnd)} ${arcY(trackEnd)}`;
  const fillPath = score > 0
    ? `M ${arcX(startAngle)} ${arcY(startAngle)} A ${radius} ${radius} 0 ${endAngle - startAngle > 180 ? 1 : 0} 1 ${arcX(endAngle)} ${arcY(endAngle)}`
    : "";

  const gaugeColor =
    level === "HIGH" ? "#f04f57" : level === "MEDIUM" ? "#f5a623" : "#22c55e";

  return (
    <div className="flex flex-col items-center">
      <svg width="140" height="110" viewBox="0 0 140 110">
        {/* Track */}
        <path
          d={trackPath}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Fill arc */}
        {fillPath && (
          <path
            d={fillPath}
            fill="none"
            stroke={gaugeColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${gaugeColor}80)` }}
          />
        )}
        {/* Score text */}
        <text
          x={cx}
          y={cy + 6}
          textAnchor="middle"
          fontSize="22"
          fontWeight="700"
          fontFamily="IBM Plex Mono"
          fill={gaugeColor}
        >
          {score}
        </text>
        <text
          x={cx}
          y={cy + 22}
          textAnchor="middle"
          fontSize="9"
          fontFamily="IBM Plex Sans"
          fill="#7c8fa6"
        >
          / 100
        </text>
      </svg>
      <span className={`text-xs font-mono font-semibold mt-1 ${colors.text}`}>
        {level} RISK
      </span>
    </div>
  );
}

// Reusable select field
function SelectField({
  label,
  value,
  onChange,
  options,
  required,
}: {
  label: string;
  value: string | number;
  onChange: (v: string) => void;
  options: (string | number)[];
  required?: boolean;
}) {
  return (
    <div>
      <label className="field-label">
        {label}
        {required && <span className="text-fraud ml-0.5">*</span>}
      </label>
      <select
        className="select-field"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
      >
        <option value="">Select...</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

// Reusable number input
function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  required,
}: {
  label: string;
  value: number | string;
  onChange: (v: string) => void;
  min?: number;
  max?: number;
  required?: boolean;
}) {
  return (
    <div>
      <label className="field-label">
        {label}
        {required && <span className="text-fraud ml-0.5">*</span>}
      </label>
      <input
        type="number"
        className="input-field"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        min={min}
        max={max}
        required={required}
      />
    </div>
  );
}

// Collapsible form section
function FormSection({
  title,
  description,
  isOpen,
  onToggle,
  children,
}: {
  title: string;
  description: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="card overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-white/[0.02] transition-colors duration-150"
      >
        <div>
          <p className="text-sm font-display font-semibold text-primary">{title}</p>
          <p className="text-xs font-body text-secondary mt-0.5">{description}</p>
        </div>
        <span className="text-secondary flex-shrink-0 ml-4">
          {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>
      {isOpen && (
        <div className="px-5 pb-5 border-t divider pt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {children}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalyzeClaim() {
  // Form state - initialized with empty/default values
  const [form, setForm] = useState<Partial<ClaimInput>>({});
  const [sections, setSections] = useState<Record<SectionKey, boolean>>({
    policy: true,
    vehicle: true,
    incident: true,
    claimant: true,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const set = (key: keyof ClaimInput, value: string | number) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const toggleSection = (key: SectionKey) => {
    setSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const loadDemo = () => {
    setForm(DEMO_CASE);
    setResult(null);
    setError(null);
    // Open all sections so user can see the pre-filled data
    setSections({ policy: true, vehicle: true, incident: true, claimant: true });
  };

  const resetForm = () => {
    setForm({});
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = form as ClaimInput;
      const response = await predictClaim(payload);
      setResult(response);
      // Scroll result panel into view smoothly
      setTimeout(() => {
        document.getElementById("result-panel")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err: unknown) {
      const message =
        (err as { message?: string }).message ?? "Prediction request failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const riskColors = result ? getRiskColor(result.risk_level) : null;

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl">
      {/* Page header row */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Info size={14} className="text-secondary" />
          <p className="text-xs font-body text-secondary">
            All fields marked <span className="text-fraud">*</span> are required
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={loadDemo}
            className="btn-ghost text-xs"
          >
            <Sparkles size={13} />
            Load Demo Case
          </button>
          <button
            type="button"
            onClick={resetForm}
            className="btn-ghost text-xs"
          >
            <RotateCcw size={13} />
            Reset
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Section 1 - Policy Info */}
        <FormSection
          title="Policy Information"
          description="Claim timing, policy type, and coverage details"
          isOpen={sections.policy}
          onToggle={() => toggleSection("policy")}
        >
          <SelectField
            label="Month of Incident"
            value={form.Month ?? ""}
            onChange={(v) => set("Month", v)}
            options={MONTHS}
            required
          />
          <SelectField
            label="Week of Month"
            value={form.WeekOfMonth ?? ""}
            onChange={(v) => set("WeekOfMonth", Number(v))}
            options={WEEK_NUMS}
            required
          />
          <SelectField
            label="Day of Week"
            value={form.DayOfWeek ?? ""}
            onChange={(v) => set("DayOfWeek", v)}
            options={DAYS_OF_WEEK}
            required
          />
          <SelectField
            label="Month Claimed"
            value={form.MonthClaimed ?? ""}
            onChange={(v) => set("MonthClaimed", v)}
            options={MONTHS}
            required
          />
          <SelectField
            label="Week of Month Claimed"
            value={form.WeekOfMonthClaimed ?? ""}
            onChange={(v) => set("WeekOfMonthClaimed", Number(v))}
            options={WEEK_NUMS}
            required
          />
          <SelectField
            label="Day of Week Claimed"
            value={form.DayOfWeekClaimed ?? ""}
            onChange={(v) => set("DayOfWeekClaimed", v)}
            options={DAYS_OF_WEEK}
            required
          />
          <SelectField
            label="Policy Type"
            value={form.PolicyType ?? ""}
            onChange={(v) => set("PolicyType", v)}
            options={POLICY_TYPES}
            required
          />
          <SelectField
            label="Base Policy"
            value={form.BasePolicy ?? ""}
            onChange={(v) => set("BasePolicy", v)}
            options={BASE_POLICIES}
            required
          />
          <SelectField
            label="Deductible"
            value={form.Deductible ?? ""}
            onChange={(v) => set("Deductible", Number(v))}
            options={DEDUCTIBLES}
            required
          />
          <SelectField
            label="Days - Policy to Accident"
            value={form["Days:Policy-Accident"] ?? ""}
            onChange={(v) => set("Days:Policy-Accident", v)}
            options={DAYS_RANGES}
            required
          />
          <SelectField
            label="Days - Policy to Claim"
            value={form["Days:Policy-Claim"] ?? ""}
            onChange={(v) => set("Days:Policy-Claim", v)}
            options={DAYS_RANGES}
            required
          />
          <SelectField
            label="Agent Type"
            value={form.AgentType ?? ""}
            onChange={(v) => set("AgentType", v)}
            options={AGENT_TYPES}
            required
          />
        </FormSection>

        {/* Section 2 - Vehicle Info */}
        <FormSection
          title="Vehicle Details"
          description="Vehicle make, category, price, and age"
          isOpen={sections.vehicle}
          onToggle={() => toggleSection("vehicle")}
        >
          <SelectField
            label="Make"
            value={form.Make ?? ""}
            onChange={(v) => set("Make", v)}
            options={MAKES}
            required
          />
          <SelectField
            label="Vehicle Category"
            value={form.VehicleCategory ?? ""}
            onChange={(v) => set("VehicleCategory", v)}
            options={VEHICLE_CATEGORIES}
            required
          />
          <SelectField
            label="Vehicle Price Range ($)"
            value={form.VehiclePrice ?? ""}
            onChange={(v) => set("VehiclePrice", v)}
            options={VEHICLE_PRICES}
            required
          />
          <SelectField
            label="Age of Vehicle"
            value={form.AgeOfVehicle ?? ""}
            onChange={(v) => set("AgeOfVehicle", v)}
            options={AGE_OF_VEHICLE}
            required
          />
          <SelectField
            label="Number of Cars"
            value={form.NumberOfCars ?? ""}
            onChange={(v) => set("NumberOfCars", v)}
            options={NUM_CARS}
            required
          />
        </FormSection>

        {/* Section 3 - Incident Info */}
        <FormSection
          title="Incident Details"
          description="Accident area, fault, police report, and witnesses"
          isOpen={sections.incident}
          onToggle={() => toggleSection("incident")}
        >
          <SelectField
            label="Accident Area"
            value={form.AccidentArea ?? ""}
            onChange={(v) => set("AccidentArea", v)}
            options={ACCIDENT_AREAS}
            required
          />
          <SelectField
            label="Fault"
            value={form.Fault ?? ""}
            onChange={(v) => set("Fault", v)}
            options={FAULTS}
            required
          />
          <SelectField
            label="Police Report Filed"
            value={form.PoliceReportFiled ?? ""}
            onChange={(v) => set("PoliceReportFiled", v)}
            options={YES_NO}
            required
          />
          <SelectField
            label="Witness Present"
            value={form.WitnessPresent ?? ""}
            onChange={(v) => set("WitnessPresent", v)}
            options={YES_NO}
            required
          />
          <SelectField
            label="Number of Supplements"
            value={form.NumberOfSuppliments ?? ""}
            onChange={(v) => set("NumberOfSuppliments", v)}
            options={NUM_SUPPLIMENTS}
            required
          />
          <SelectField
            label="Address Change to Claim"
            value={form["AddressChange-Claim"] ?? ""}
            onChange={(v) => set("AddressChange-Claim", v)}
            options={ADDRESS_CHANGE}
            required
          />
        </FormSection>

        {/* Section 4 - Claimant Info */}
        <FormSection
          title="Claimant Profile"
          description="Personal details and claim history of the policy holder"
          isOpen={sections.claimant}
          onToggle={() => toggleSection("claimant")}
        >
          <SelectField
            label="Sex"
            value={form.Sex ?? ""}
            onChange={(v) => set("Sex", v)}
            options={["Male", "Female"]}
            required
          />
          <SelectField
            label="Marital Status"
            value={form.MaritalStatus ?? ""}
            onChange={(v) => set("MaritalStatus", v)}
            options={["Single", "Married", "Divorced", "Widow"]}
            required
          />
          <NumberField
            label="Age"
            value={form.Age ?? ""}
            onChange={(v) => set("Age", Number(v))}
            min={16}
            max={100}
            required
          />
          <SelectField
            label="Driver Rating"
            value={form.DriverRating ?? ""}
            onChange={(v) => set("DriverRating", Number(v))}
            options={DRIVER_RATINGS}
            required
          />
          <SelectField
            label="Age of Policy Holder"
            value={form.AgeOfPolicyHolder ?? ""}
            onChange={(v) => set("AgeOfPolicyHolder", v)}
            options={AGE_OF_POLICY_HOLDER}
            required
          />
          <SelectField
            label="Past Number of Claims"
            value={form.PastNumberOfClaims ?? ""}
            onChange={(v) => set("PastNumberOfClaims", v)}
            options={PAST_CLAIMS}
            required
          />
        </FormSection>

        {/* Error display */}
        {error && (
          <div className="card p-4 border border-fraud/20 flex items-start gap-3">
            <AlertTriangle size={15} className="text-fraud flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-body font-medium text-primary">
                Prediction failed
              </p>
              <p className="text-xs font-mono text-secondary mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Submit button */}
        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full justify-center py-3 text-sm font-semibold disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 size={15} className="animate-spin" />
              Analyzing claim...
            </>
          ) : (
            <>
              <ScanSearch size={15} />
              Analyze Claim
            </>
          )}
        </button>
      </form>

      {/* Result panel */}
      {result && (
        <div id="result-panel" className="space-y-4 animate-fade-in">
          {/* Result header */}
          <div
            className="card p-5"
            style={{
              borderColor: result.is_fraud
                ? "rgba(240,79,87,0.25)"
                : "rgba(34,197,94,0.25)",
            }}
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6">
              {/* Gauge */}
              <RiskGauge
                score={result.risk_score}
                level={result.risk_level}
              />

              {/* Main result info */}
              <div className="flex-1 min-w-0 space-y-3">
                {/* Verdict badge */}
                <div className="flex items-center gap-3 flex-wrap">
                  <span
                    className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-sm font-mono font-semibold ${
                      result.is_fraud
                        ? "bg-fraud-muted text-fraud border border-fraud/20"
                        : "bg-safe-muted text-safe border border-safe/20"
                    }`}
                  >
                    {result.is_fraud ? (
                      <ShieldAlert size={14} />
                    ) : (
                      <ShieldCheck size={14} />
                    )}
                    {result.is_fraud ? "Fraud Detected" : "Likely Legitimate"}
                  </span>
                  <span
                    className={`text-xs font-mono px-2.5 py-1 rounded-lg border ${
                      result.confidence === "high"
                        ? "text-safe bg-safe-dim border-safe/15"
                        : result.confidence === "medium"
                        ? "text-warn bg-warn-dim border-warn/15"
                        : "text-secondary bg-white/5 border-white/10"
                    }`}
                  >
                    {result.confidence} confidence
                  </span>
                </div>

                {/* Probability bar */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-body text-secondary">
                      Fraud Probability
                    </span>
                    <span
                      className={`text-sm font-mono font-bold ${riskColors?.text}`}
                    >
                      {formatProbability(result.fraud_probability)}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: formatProbability(result.fraud_probability),
                        background:
                          result.risk_level === "HIGH"
                            ? "#f04f57"
                            : result.risk_level === "MEDIUM"
                            ? "#f5a623"
                            : "#22c55e",
                      }}
                    />
                  </div>
                </div>

                {/* Recommendation */}
                <div className="card-elevated rounded-xl px-4 py-3">
                  <p className="text-xs font-body font-semibold text-secondary uppercase tracking-wide mb-1">
                    Recommendation
                  </p>
                  <p className="text-sm font-body text-primary leading-relaxed">
                    {result.recommendation}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Risk factors */}
          {result.risk_factors && result.risk_factors.length > 0 && (
            <div className="card p-5">
              <div className="mb-4">
                <h3 className="text-sm font-display font-semibold text-primary">
                  Top Risk Factors
                </h3>
                <p className="text-xs font-body text-secondary mt-0.5">
                  Features driving this prediction, by model importance
                </p>
              </div>
              <div className="space-y-3">
                {result.risk_factors.slice(0, 7).map((factor, i) => (
                  <div key={i}>
                    <div className="flex items-start justify-between gap-4 mb-1.5">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono text-muted w-4 flex-shrink-0">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <span className="text-xs font-body font-medium text-primary truncate">
                            {formatFeatureName(factor.feature)}
                          </span>
                        </div>
                        {factor.description && (
                          <p className="text-xs font-body text-secondary mt-0.5 ml-6">
                            {factor.description}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-3 flex-shrink-0">
                        <span className="text-xs font-mono text-secondary">
                          val: {factor.value}
                        </span>
                        <span className="text-xs font-mono font-semibold text-accent w-10 text-right">
                          {(factor.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    {/* Importance bar */}
                    <div className="ml-6 h-1.5 rounded-full bg-white/[0.05] overflow-hidden">
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

          {/* Model metadata */}
          <div className="card px-5 py-3">
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                <span className="text-xs font-mono text-secondary">
                  Model:{" "}
                  <span className="text-primary font-medium">
                    {result.model_info.model_name}
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent/50" />
                <span className="text-xs font-mono text-secondary">
                  Threshold:{" "}
                  <span className="text-primary font-medium">
                    {result.model_info.deployed_threshold.toFixed(4)}
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent/50" />
                <span className="text-xs font-mono text-secondary">
                  Inference:{" "}
                  <span className="text-primary font-medium">
                    {(result.inference_ms ?? 0).toFixed(0)}ms
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent/50" />
                <span className="text-xs font-mono text-secondary">
                  Algorithm:{" "}
                  <span className="text-primary font-medium">
                    {result.model_info.algorithm}
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
      <Footer />
    </div>
  );
}