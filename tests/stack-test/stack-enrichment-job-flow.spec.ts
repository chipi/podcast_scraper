import { expect, test } from '@playwright/test'

/**
 * RFC-088 enrichment jobs API — flow against the live stack.
 *
 * POSTs an enrichment job (corpus_only — no episode bundles required),
 * confirms it appears in the shared ``/api/jobs`` registry with
 * ``command_type == "corpus_enrichment"``, and verifies the cancel
 * endpoint accepts the job id (command_type-agnostic cancel).
 *
 * Smoke-spec scope: the enricher set is empty in airgapped_thin
 * (chunk 2 wires real enrichers); the job's terminal status here is
 * "queued" → "running" → ("ok" with no work performed, or "cancelled"
 * if we cancel it before it picks up the slot). Either is acceptable.
 */

const CORPUS = '/app/output'
const JOB_POLL_TIMEOUT_MS = 60_000

type JobRecord = {
  job_id?: string
  status?: string
  command_type?: string
  error_reason?: string | null
}

async function pollForJobInList(
  request: import('@playwright/test').APIRequestContext,
  jobId: string,
  timeoutMs: number,
): Promise<JobRecord> {
  const deadline = Date.now() + timeoutMs
  let last: JobRecord | undefined
  while (Date.now() < deadline) {
    const r = await request.get('/api/jobs', { params: { path: CORPUS } })
    if (r.ok()) {
      const body = (await r.json()) as { jobs?: JobRecord[] }
      const jobs = Array.isArray(body.jobs) ? body.jobs : []
      const match = jobs.find((j) => j.job_id === jobId)
      if (match) {
        last = match
        return match
      }
    }
    await new Promise((r2) => setTimeout(r2, 500))
  }
  throw new Error(
    `enrichment job ${jobId} did not appear in /api/jobs within ${timeoutMs}ms (last=${JSON.stringify(last)})`,
  )
}

test.describe('stack test — RFC-088 enrichment job flow', () => {
  test('POST /api/jobs/enrichment 202 + appears in /api/jobs with command_type', async ({
    request,
  }) => {
    const submit = await request.post('/api/jobs/enrichment', {
      params: { path: CORPUS },
      data: { corpus_only: true },
    })
    expect(submit.status()).toBe(202)
    const accepted = (await submit.json()) as {
      job_id?: string
      status?: string
      corpus_path?: string
    }
    expect(accepted.job_id).toBeTruthy()
    expect(accepted.status).toMatch(/^(running|queued)$/)
    expect(accepted.corpus_path).toBeTruthy()

    // /api/jobs lists pipeline AND enrichment jobs. The new one must show.
    const jobId = accepted.job_id as string
    const listed = await pollForJobInList(request, jobId, JOB_POLL_TIMEOUT_MS)
    expect(listed.command_type).toBe('corpus_enrichment')
  })

  test('POST /api/jobs/{job_id}/cancel is command_type-agnostic', async ({ request }) => {
    // Submit a second enrichment job, then cancel via the shared route.
    const submit = await request.post('/api/jobs/enrichment', {
      params: { path: CORPUS },
      data: { corpus_only: true },
    })
    expect(submit.status()).toBe(202)
    const { job_id } = (await submit.json()) as { job_id: string }

    const cancel = await request.post(`/api/jobs/${job_id}/cancel`, {
      params: { path: CORPUS },
    })
    // Either the cancel transitioned the record (200 + record) or the job
    // already finished (still a 200 with a terminal status). The route's
    // contract here is "no exception"; both terminal states are acceptable.
    expect(cancel.status()).toBe(200)
    const body = (await cancel.json()) as JobRecord
    expect(body.job_id).toBe(job_id)
    expect(body.command_type).toBe('corpus_enrichment')
  })
})
