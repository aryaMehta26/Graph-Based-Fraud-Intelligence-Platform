import snapshot from './dashboard.json';

export type DashboardSnapshot = typeof snapshot;

/**
 * Read-only data boundary for the dashboard. Set VITE_API_BASE_URL when the
 * additive API service is available; otherwise the generated artifact snapshot
 * keeps the UI usable and makes the fallback explicit.
 */
export async function getDashboardSnapshot(): Promise<DashboardSnapshot> {
  const baseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (!baseUrl) return snapshot;

  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, '')}/api/dashboard/snapshot`);
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    return (await response.json()) as DashboardSnapshot;
  } catch {
    return snapshot;
  }
}

