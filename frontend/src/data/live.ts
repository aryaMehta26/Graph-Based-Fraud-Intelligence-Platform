export type LivePrediction = {
  case_id: string;
  Timestamp: string;
  src_acct: string;
  dst_acct: string;
  'Amount Paid': number;
  'Payment Format': string;
  xgboost_score: number;
  fraud_probability: number;
  risk_level: string;
  prediction: string;
  source: string;
  investigatable?: boolean;
};

const api = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') || '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${api}${path}`, options);
  if (!response.ok) throw new Error((await response.text()) || `Live API returned ${response.status}`);
  return response.json() as Promise<T>;
}

export function getLiveStatus() {
  return request<{ status: string; data_available: boolean; model_available: boolean; gemma_enabled: boolean; gemma_loaded: boolean }>('/api/live/status');
}

export function getLivePredictions(limit = 8) {
  return request<{ items: LivePrediction[] }>(`/api/live/predictions?limit=${limit}`);
}

export function getLiveGraph(accountId: string) {
  return request<{ account: Record<string, unknown>; edges: Record<string, unknown>[]; source: string }>(`/api/graph/${encodeURIComponent(accountId)}?limit=100`);
}

export function runLiveInvestigation(payload: { case_id: string; context: Record<string, unknown> }) {
  return request<{ report: Record<string, unknown>; latency_seconds: number; case_id: string }>('/api/live/investigate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
