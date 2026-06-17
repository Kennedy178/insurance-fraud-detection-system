import type {
  PredictionResponse,
  PredictionRecord,
  PredictionsResponse,
  StatsResponse,
  ModelInfoResponse,
  HealthResponse,
  ClaimInput,
  ApiError,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      message = response.statusText || message;
    }
    const error: ApiError = { message, status: response.status };
    throw error;
  }

  return response.json() as Promise<T>;
}

export async function predictClaim(
  claim: ClaimInput
): Promise<PredictionResponse> {
  return apiFetch<PredictionResponse>("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify(claim),
  });
}

export async function getPredictions(
  limit = 20,
  offset = 0
): Promise<PredictionRecord[]> {
  const data = await apiFetch<PredictionsResponse>(
    `/api/v1/predictions?limit=${limit}&offset=${offset}`
  );
  return data.predictions;
}

export async function getStats(): Promise<StatsResponse> {
  return apiFetch<StatsResponse>("/api/v1/stats");
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return apiFetch<ModelInfoResponse>("/api/v1/model/info");
}

// Calls /api/v1/ping instead of /health to avoid ad-blocker blocklists
export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/v1/ping");
}