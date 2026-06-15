// All TypeScript interfaces matching the FastAPI backend response shapes exactly

// Risk level values
export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

// Confidence values
export type ConfidenceLevel = "low" | "medium" | "high";

// A single risk factor from the model
export interface RiskFactor {
  feature: string;
  importance: number;
  value: number;
  description: string;
}

// Model metadata embedded in prediction response
export interface ModelInfo {
  model_name: string;
  model_version: string;
  deployed_threshold: number;
  algorithm: string;
}

// Response from POST /api/v1/predict
export interface PredictionResponse {
  is_fraud: boolean;
  fraud_probability: number;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: ConfidenceLevel;
  recommendation: string;
  risk_factors: RiskFactor[];
  model_info: ModelInfo;
  inference_ms: number | null;
}

// A record returned from GET /api/v1/predictions
export interface PredictionRecord {
  id: number;
  created_at: string;
  is_fraud: boolean;
  fraud_probability: number;
  risk_score: number;
  risk_level: RiskLevel;
  confidence: ConfidenceLevel;
  recommendation: string;
  risk_factors: RiskFactor[];
  inference_ms: number | null;
}

// Wrapper returned by GET /api/v1/predictions
export interface PredictionsResponse {
  count: number;
  predictions: PredictionRecord[];
}

// Daily trend data point
export interface DailyTrend {
  date: string;
  total: number;
  fraud: number;
}

// Summary block from GET /api/v1/stats
export interface StatsSummary {
  period_days: number;
  total_predictions: number;
  fraud_count: number;
  legitimate_count: number;
  fraud_rate: number;
  avg_fraud_probability: number;
  high_risk_count: number;
}

// Full response from GET /api/v1/stats
export interface StatsResponse {
  summary: StatsSummary;
  daily_trend: DailyTrend[];
}

// Response from GET /api/v1/model/info
export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  algorithm: string;
  deployed_threshold: number;
  roc_auc: number;
  features_count: number;
  trained_at?: string;
}

// Response from GET /health
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  database: string;
  version: string;
}

// Claim input payload sent to POST /api/v1/predict
// Field names match the backend exactly - special chars preserved
export interface ClaimInput {
  Month: string;
  WeekOfMonth: number;
  DayOfWeek: string;
  MonthClaimed: string;
  WeekOfMonthClaimed: number;
  DayOfWeekClaimed: string;
  Sex: string;
  MaritalStatus: string;
  Age: number;
  DriverRating: number;
  Make: string;
  VehicleCategory: string;
  VehiclePrice: string;
  AgeOfVehicle: string;
  AgeOfPolicyHolder: string;
  PolicyType: string;
  BasePolicy: string;
  Deductible: number;
  AccidentArea: string;
  Fault: string;
  AgentType: string;
  PoliceReportFiled: string;
  WitnessPresent: string;
  PastNumberOfClaims: string;
  NumberOfSuppliments: string;
  NumberOfCars: string;
  "Days:Policy-Accident": string;
  "Days:Policy-Claim": string;
  "AddressChange-Claim": string;
}

// API error shape
export interface ApiError {
  message: string;
  status: number;
}