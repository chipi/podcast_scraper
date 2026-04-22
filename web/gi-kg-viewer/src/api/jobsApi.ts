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

/** Open raw job log in a new tab (same origin as the viewer). */
export function pipelineJobLogUrl(corpusPath: string, jobId: string): string {
  const q = new URLSearchParams({ path: corpusPath.trim(), job_id: jobId.trim() })
  return `/api/jobs/subprocess-log?${q.toString()}`
}

export interface PipelineJobLogTailResponse {
  text: string
  truncated: boolean
}

/** Turn FastAPI JSON error bodies into a short string for UI. */
export function formatJobHttpErrorMessage(message: string): string {
  const m = message.trim()
  if (!m.startsWith('{')) {
    return m
  }
  try {
    const o = JSON.parse(m) as { detail?: unknown }
    const d = o.detail
    if (typeof d === 'string') {
      if (d === 'Not Found') {
        return 'HTTP 404: log-tail route missing or wrong URL — upgrade serve or open full log from Command & paths.'
      }
      if (d === 'Job not found.' || d === 'Job not found') {
        return 'No registry row for this job id under the current corpus path (check Corpus root, refresh Job history, restart serve after upgrading).'
      }
      return d
    }
    if (Array.isArray(d)) {
      return d.map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: unknown }).msg) : String(x))).join('; ')
    }
  } catch {
    /* not JSON */
  }
  return m
}

/** Last portion of the job subprocess log (UTF-8); for dashboard summary + metrics. */
export async function fetchPipelineJobLogTail(
  corpusPath: string,
  jobId: string,
  maxBytes = 96_000,
): Promise<PipelineJobLogTailResponse> {
  const q = new URLSearchParams({
    path: corpusPath.trim(),
    job_id: jobId.trim(),
    max_bytes: String(maxBytes),
  })
  const res = await fetchWithTimeout(`/api/jobs/subprocess-log-tail?${q.toString()}`, undefined, {
    timeoutDetail: 'jobs',
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as PipelineJobLogTailResponse
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
