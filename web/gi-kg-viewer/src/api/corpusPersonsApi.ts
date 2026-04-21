import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

export interface TopPersonItem {
  person_id: string
  display_name: string
  episode_count: number
  insight_count: number
  top_topics: string[]
}

export interface CorpusTopPersonsResponse {
  path: string
  persons: TopPersonItem[]
  total_persons: number
}

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export async function fetchCorpusTopPersons(
  corpusPath: string,
  limit = 5,
): Promise<CorpusTopPersonsResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim(), limit: String(limit) })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/persons/top?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/persons/top?${qs}`)
    return readJson<CorpusTopPersonsResponse>(res)
  })
}
