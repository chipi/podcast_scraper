import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

/** One trending entity (RFC-103 momentum) — velocity (rising) + volume (recent level). */
export type TrendingEntity = {
  entity_id: string
  kind: string
  label: string
  velocity: number
  volume: number
  heating_up: boolean
  total: number
  series: number[]
}

/** GET /api/corpus/trending — operator global view: top momentum per kind, corpus-wide. */
export type CorpusTrendingDocument = {
  as_of_week: string
  kinds: Record<string, TrendingEntity[]>
}

export type CorpusTrendingFetchResult =
  | { status: 'ok'; document: CorpusTrendingDocument }
  | { status: 'missing' }
  | { status: 'error'; message: string }

function corpusQuery(path: string, limitPerKind: number): string {
  const q = new URLSearchParams()
  const t = path.trim()
  if (t) {
    q.set('path', t)
  }
  q.set('limit_per_kind', String(limitPerKind))
  const s = q.toString()
  return s ? `?${s}` : ''
}

/**
 * Fetch the operator global trending view via the viewer API. ``missing`` on 404 (no corpus) so the
 * caller degrades to "no trending panel".
 */
export async function fetchCorpusTrending(
  corpusPath: string,
  limitPerKind = 6,
): Promise<CorpusTrendingFetchResult> {
  const url = `/api/corpus/trending${corpusQuery(corpusPath, limitPerKind)}`
  try {
    const res = await dedupeInFlight(url, () =>
      fetchWithTimeout(url, undefined, { timeoutDetail: 'corpus/trending' }),
    )
    if (res.status === 404) {
      return { status: 'missing' }
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      return { status: 'error', message: text.trim() || `HTTP ${res.status} trending` }
    }
    const document = (await res.json()) as CorpusTrendingDocument
    return { status: 'ok', document }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e)
    return { status: 'error', message }
  }
}
