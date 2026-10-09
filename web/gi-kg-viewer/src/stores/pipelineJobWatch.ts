import { defineStore } from 'pinia'
import { listPipelineJobs, type PipelineJobRow } from '../api/jobsApi'
import { useShellStore } from './shell'

/** How often to look while a job is in flight and nothing else is polling. */
export const JOB_WATCH_INTERVAL_MS = 15_000

/**
 * Notices a pipeline job FINISHING, whichever tab is open, and tells the kept-alive Library and
 * Digest tabs through `shell.corpusRevision` (2026-10-09).
 *
 * The Dashboard's jobs card is the only thing that lists jobs, and the Dashboard is not kept alive:
 * leave it and the polling stops. A job started there and finished while the operator was on Library
 * then changed the corpus with nothing on screen hearing about it. This store outlives every tab.
 *
 * The card feeds it every list it fetches (`observe`); between those, while any job is queued or
 * running, the store looks for itself on a slow timer and stops as soon as none is.
 */
export const usePipelineJobWatchStore = defineStore('pipelineJobWatch', () => {
  let inFlight = new Set<string>()
  let timer: ReturnType<typeof setTimeout> | null = null

  /** One fresh list of jobs. A job seen queued/running before and `succeeded` now = corpus changed. */
  function observe(jobs: PipelineJobRow[]): void {
    if (jobs.some((j) => j.status === 'succeeded' && inFlight.has(j.job_id))) {
      useShellStore().noteCorpusChanged()
    }
    inFlight = new Set(jobs.filter((j) => j.status === 'queued' || j.status === 'running').map((j) => j.job_id))
    schedule()
  }

  function schedule(): void {
    if (timer !== null) {
      clearTimeout(timer)
      timer = null
    }
    if (!inFlight.size) return
    timer = setTimeout(() => void tick(), JOB_WATCH_INTERVAL_MS)
  }

  async function tick(): Promise<void> {
    timer = null
    const shell = useShellStore()
    const root = shell.corpusPath.trim()
    if (!root || !shell.jobsApiAvailable) return
    try {
      const res = await listPipelineJobs(root)
      observe(Array.isArray(res.jobs) ? res.jobs : [])
    } catch {
      // A blip is not a verdict: keep what we know and look again later.
      schedule()
    }
  }

  /** Test seam: forget everything and stop the timer. */
  function reset(): void {
    inFlight = new Set()
    if (timer !== null) clearTimeout(timer)
    timer = null
  }

  return { observe, reset }
})
