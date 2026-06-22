import { fetchWithTimeout } from './httpClient'

/** One source's result from the control plane: a uniform {ok, ...} envelope (#803). */
export interface OpsSourceEnvelope {
  ok: boolean
  source: string
  configured?: boolean
  error?: string
  data?: unknown
}

/** GET /api/ops/summary — the podcast_obs control-plane glance (same data the MCP exposes). */
export interface OpsSummary {
  target: string
  live: string[]
  unconfigured: string[]
  failed: string[]
  sources: Record<string, OpsSourceEnvelope>
}

export async function fetchOpsSummary(): Promise<OpsSummary> {
  // Slightly longer timeout than the default: the endpoint fans out to several backends.
  const res = await fetchWithTimeout('/api/ops/summary', undefined, {
    timeoutMs: 20_000,
    timeoutDetail: 'ops',
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as OpsSummary
}
