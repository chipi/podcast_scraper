import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { PipelineJobRow } from '../api/jobsApi'

/**
 * Which job's log the in-app viewer (#695) is showing. Held in a store so any
 * pipeline surface (Jobs card, history strip, explore panel) can open the single
 * dialog instance mounted at the dashboard root, instead of each surface hosting
 * its own modal. Mirrors the dashboardNav store's "open from anywhere" pattern.
 */
export type PipelineJobLogTarget = {
  corpusPath: string
  jobId: string
  /** Snapshot of the job status at open time (drives auto-refresh on/off). */
  status: string
  /** Relative log path (for the header + download link); null when unknown. */
  logRelpath: string | null
}

export const usePipelineJobLogStore = defineStore('pipelineJobLog', () => {
  const open = ref(false)
  const target = ref<PipelineJobLogTarget | null>(null)

  function viewLog(t: PipelineJobLogTarget): void {
    target.value = t
    open.value = true
  }

  function viewLogForRow(corpusPath: string, row: PipelineJobRow): void {
    viewLog({
      corpusPath,
      jobId: row.job_id,
      status: row.status,
      logRelpath: row.log_relpath || null,
    })
  }

  function close(): void {
    open.value = false
  }

  return { open, target, viewLog, viewLogForRow, close }
})
