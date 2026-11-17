import { fetchWithTimeout } from './httpClient'

/** One configured cron schedule + its next-run preview (#708 / #709). */
export interface ScheduledJobItem {
  name: string
  cron: string
  enabled: boolean
  /** Job fired on the schedule: 'pipeline' (ingestion) or 'enrichment' (#1069). */
  kind: string
  /** ISO time; null when disabled, the cron is invalid, or the scheduler is off. */
  next_run_at: string | null
}

export interface ScheduledJobsList {
  path: string
  scheduler_running: boolean
  timezone: string
  jobs: ScheduledJobItem[]
}

export async function getScheduledJobs(corpusPath: string): Promise<ScheduledJobsList> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(`/api/scheduled-jobs?${q}`, undefined, {
    timeoutDetail: 'jobs',
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as ScheduledJobsList
}
