/**
 * Storylines and themes over time (RFC-128, #2349) — the dashboard's clustering card.
 *
 *  - GET  /api/corpus/clustering/health      the latest run, its changes, the series, the seed
 *  - POST /api/corpus/storylines/seed?mode=  preview | apply | undo the storyline history seed
 */
import { fetchWithTimeout } from './httpClient'

export type ClusteringChange = { id: string; name?: string; from?: string; to?: string; into?: string; into_name?: string }

export type ClusteringChanges = Partial<
  Record<'created' | 'renamed' | 'quiet' | 'woke' | 'absorbed' | 'gone', ClusteringChange[]>
>

export type StorylineRunRow = {
  ts: number
  source?: string
  ok?: boolean
  storylines?: number
  active?: number
  quiet?: number
  candidates?: number
  episodes_covered?: number
  episodes_total?: number
  changes?: ClusteringChanges
}

export type ThemeRunRow = {
  ts: number
  themes?: number
  topics?: number
  kept?: number
  changes?: ClusteringChanges
}

export type ClusteringHealth = {
  storylines: {
    latest: StorylineRunRow | null
    health: { ok?: boolean; source?: string; shared_topics?: number; vanished?: string[] } | null
    seed: { state: 'none' | 'pending' | 'taken'; id: string | null }
    series: StorylineRunRow[]
  }
  themes: { latest: ThemeRunRow | null; series: ThemeRunRow[] }
}

export type SeedMode = 'preview' | 'apply' | 'undo'

export type SeedResult = {
  mode: SeedMode
  current?: number
  replayed?: number
  with_history?: number
  folded?: number
  current_only?: number
  added_from_history?: string[]
  candidates?: number
  seed_id?: string
  already_taken?: boolean
  applied?: boolean
  history_months?: number
  undone?: string
  reason?: string
}

function corpusQuery(path: string, extra: Record<string, string> = {}): string {
  const q = new URLSearchParams({ ...extra })
  if (path.trim()) q.set('path', path.trim())
  const s = q.toString()
  return s ? `?${s}` : ''
}

export async function getClusteringHealth(corpusPath: string): Promise<ClusteringHealth> {
  const res = await fetchWithTimeout(
    `/api/corpus/clustering/health${corpusQuery(corpusPath)}`,
    {},
    { timeoutDetail: 'corpus/clustering/health' },
  )
  if (!res.ok) throw new Error(`clustering health failed: HTTP ${res.status}`)
  return (await res.json()) as ClusteringHealth
}

export async function postStorylineSeed(corpusPath: string, mode: SeedMode): Promise<SeedResult> {
  const res = await fetchWithTimeout(
    `/api/corpus/storylines/seed${corpusQuery(corpusPath, { mode })}`,
    { method: 'POST' },
    { timeoutDetail: 'corpus/storylines/seed' },
  )
  if (!res.ok) throw new Error(`storyline seed (${mode}) failed: HTTP ${res.status}`)
  return (await res.json()) as SeedResult
}
