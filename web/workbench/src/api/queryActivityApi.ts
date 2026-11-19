import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

/**
 * Client for `GET /api/corpus/query-activity` (PRD-033 FR6.2) — daily search volume
 * from the append-only activity log. An honest search-volume-over-time signal.
 */

export interface QueryActivityBucket {
  date: string
  count: number
}

export interface QueryActivityResponse {
  total: number
  buckets: QueryActivityBucket[]
  error?: string | null
}

export async function fetchQueryActivity(
  corpusPath: string,
  days = 30,
): Promise<QueryActivityResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  q.set('days', String(Math.max(1, Math.min(365, Math.floor(days)))))
  const url = `/api/corpus/query-activity?${q.toString()}`
  return dedupeInFlight(`GET|${url}`, async () => {
    const res = await fetchWithTimeout(url)
    if (!res.ok) {
      const t = await res.text()
      throw new Error(t || `HTTP ${res.status}`)
    }
    return (await res.json()) as QueryActivityResponse
  })
}
