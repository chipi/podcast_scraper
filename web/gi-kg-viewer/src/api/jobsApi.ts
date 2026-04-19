import { fetchWithTimeout } from './httpClient'

export interface PipelineJobRow {
  job_id: string
  command_type: string
  status: string
  created_at: string
  started_at: string | null
  ended_at: string | null
  pid: number | null
  argv_summary: string
  exit_code: number | null
  log_relpath: string
  error_reason: string | null
  cancel_requested?: boolean
  queue_position?: number | null
}

export interface PipelineJobsList {
  path: string
  jobs: PipelineJobRow[]
}

export interface PipelineJobAccepted {
  job_id: string
  status: string
  corpus_path: string
  queue_position?: number | null
}

export interface PipelineJobReconcileResult {
  path: string
  updated: number
  details: string[]
}

export async function listPipelineJobs(corpusPath: string): Promise<PipelineJobsList> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(`/api/jobs?${q}`, undefined, { timeoutDetail: 'jobs' })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as PipelineJobsList
}

export async function submitPipelineJob(corpusPath: string): Promise<PipelineJobAccepted> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(
    `/api/jobs?${q}`,
    { method: 'POST' },
    { timeoutDetail: 'jobs' },
  )
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as PipelineJobAccepted
}

export async function reconcilePipelineJobs(corpusPath: string): Promise<PipelineJobReconcileResult> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(
    `/api/jobs/reconcile?${q}`,
    { method: 'POST' },
    { timeoutDetail: 'jobs' },
  )
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as PipelineJobReconcileResult
}

export async function cancelPipelineJob(corpusPath: string, jobId: string): Promise<PipelineJobRow> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const res = await fetchWithTimeout(
    `/api/jobs/${encodeURIComponent(jobId)}/cancel?${q}`,
    { method: 'POST' },
    { timeoutDetail: 'jobs' },
  )
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as PipelineJobRow
}
