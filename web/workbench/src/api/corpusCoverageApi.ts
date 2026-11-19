import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

export interface CoverageByMonthItem {
  month: string
  total: number
  with_gi: number
  with_kg: number
  with_both: number
}

export interface CoverageFeedItem {
  feed_id: string
  display_title: string
  total: number
  with_gi: number
  with_kg: number
}

export interface CorpusCoverageResponse {
  path: string
  total_episodes: number
  with_gi: number
  with_kg: number
  with_both: number
  with_neither: number
  by_month: CoverageByMonthItem[]
  by_feed: CoverageFeedItem[]
}

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export async function fetchCorpusCoverage(corpusPath: string): Promise<CorpusCoverageResponse> {
  const qs = new URLSearchParams({ path: corpusPath.trim() }).toString()
  return dedupeInFlight(`GET|/api/corpus/coverage?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/coverage?${qs}`)
    const body = await readJson<CorpusCoverageResponse>(res)
    // Defensive: guarantee the array fields even if the payload omits them, so
    // CoverageByMonthChart/FeedCoverageTable never crash on rows.filter (aborting
    // the Dashboard render).
    return { ...body, by_month: body.by_month ?? [], by_feed: body.by_feed ?? [] }
  })
}
