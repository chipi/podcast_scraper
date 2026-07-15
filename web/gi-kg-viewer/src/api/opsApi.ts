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

/** One LLM provider's circuit-breaker state (ADR-113). */
export interface LlmBreakerState {
  open: boolean
  recent_failures: number
  cooldown_remaining_seconds: number
  trips_total: number
}

/** GET /api/resilience — open breakers, cooldowns, and the configured LLM call-fuse budgets. */
export interface ResilienceSnapshot {
  llm_breakers: Record<string, LlmBreakerState>
  llm_breakers_open: string[]
  rss: Record<string, unknown>
  fuses: {
    llm_max_calls_per_episode: number | null
    llm_max_calls_per_run: number | null
    note: string
  }
  any_open: boolean
}

export async function fetchResilience(): Promise<ResilienceSnapshot> {
  const res = await fetchWithTimeout('/api/resilience', undefined, {
    timeoutMs: 10_000,
    timeoutDetail: 'resilience',
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as ResilienceSnapshot
}

/** POST /api/ops/resilience/reset — force-close breakers early (operator-gated). */
export async function resetResilience(scope: 'all' | 'llm' | 'rss' = 'all'): Promise<void> {
  const res = await fetchWithTimeout(`/api/ops/resilience/reset?scope=${scope}`, {
    method: 'POST',
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
}

/** One slice of the token/cost rollup (a group + its totals). */
export interface UsageGroup {
  [dimension: string]: string | number
  calls: number
  input_tokens: number
  output_tokens: number
  cached_input_tokens: number
  cache_write_tokens: number
  estimated_cost_usd: number
  guardrail_calls: number
}

/** GET /api/usage — token/cost rollup sliced by dimension, de-duplicated by request_id. */
export interface UsageSnapshot {
  group_by: string[]
  total: {
    calls: number
    input_tokens: number
    output_tokens: number
    cached_input_tokens: number
    cache_write_tokens: number
    estimated_cost_usd: number
    guardrail_calls: number
  }
  groups: UsageGroup[]
  dimensions: string[]
  source_files: string[]
  run_id: string | null
  uninstrumented: boolean
}

export async function fetchUsage(groupBy = 'provider,model'): Promise<UsageSnapshot> {
  const res = await fetchWithTimeout(`/api/usage?group_by=${encodeURIComponent(groupBy)}`, undefined, {
    timeoutMs: 15_000,
    timeoutDetail: 'usage',
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as UsageSnapshot
}
