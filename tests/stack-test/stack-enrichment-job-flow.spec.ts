import { expect, test } from '@playwright/test'

/**
 * RFC-088 enrichment jobs API — flow against the live stack.
 *
 * POSTs an enrichment job (corpus_only — no episode bundles required),
 * confirms it appears in the shared ``/api/jobs`` registry with
 * ``command_type == "corpus_enrichment"``, and verifies the cancel
 * endpoint accepts the job id (command_type-agnostic cancel).
 *
 * The seeded operator YAML has no ``profile:`` (the jobs-flow spec saves one later), and a submit
 * with nothing to resolve an enricher from is a 400. So this spec seeds ``profile: airgapped_thin``
 * (5 corpus-scope enrichers) through ``PUT /api/operator-config`` and puts the original back
 * afterwards. Until 2026-10-28 it posted against the profile-less YAML, got a 202, and only
 * checked the job was listed: the job then failed on every run with "resolved ZERO enrichers"
 * (#1648) and nothing noticed. Now the submitted job must reach ``succeeded``; anything else
 * fails the spec with the job's own error.
 */

const CORPUS = '/app/output'
const JOB_POLL_TIMEOUT_MS = 60_000
const JOB_FINISH_TIMEOUT_MS = 180_000
const TERMINAL = new Set(['succeeded', 'failed', 'cancelled'])

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

async function waitForJobTerminal(
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
      last = (Array.isArray(body.jobs) ? body.jobs : []).find((j) => j.job_id === jobId) ?? last
      if (last?.status && TERMINAL.has(last.status)) return last
    }
    await new Promise((r2) => setTimeout(r2, 1000))
  }
  throw new Error(`enrichment job ${jobId} did not finish within ${timeoutMs}ms (last=${JSON.stringify(last)})`)
}

test.describe('stack test — RFC-088 enrichment job flow', () => {
  let originalOperatorYaml: string | null = null

  test.beforeAll(async ({ request }) => {
    const got = await request.get('/api/operator-config', { params: { path: CORPUS } })
    expect(got.ok(), `GET /api/operator-config ${got.status()}`).toBeTruthy()
    originalOperatorYaml = ((await got.json()) as { content?: string }).content ?? ''
    const seeded = /^profile:/m.test(originalOperatorYaml)
      ? originalOperatorYaml
      : `profile: airgapped_thin\n${originalOperatorYaml}`
    const put = await request.put('/api/operator-config', {
      params: { path: CORPUS },
      data: { content: seeded },
    })
    expect(put.ok(), `PUT /api/operator-config ${put.status()}: ${await put.text()}`).toBeTruthy()
  })

  test.afterAll(async ({ request }) => { // codespell:ignore afterall (the Playwright hook)
    if (originalOperatorYaml === null) return
    const put = await request.put('/api/operator-config', {
      params: { path: CORPUS },
      data: { content: originalOperatorYaml },
    })
    // A failed restore would leak `profile: airgapped_thin` into the specs that run after this one.
    expect(put.ok(), `restore PUT /api/operator-config ${put.status()}: ${await put.text()}`).toBeTruthy()
  })

  test('POST /api/jobs/enrichment 202 + appears in /api/jobs with command_type', async ({
    request,
  }) => {
    // Both waits must fit: the config's 120 s per-test timeout would otherwise kill a slow job
    // as a bare "Test timeout" before waitForJobTerminal can report its record.
    test.setTimeout(JOB_POLL_TIMEOUT_MS + JOB_FINISH_TIMEOUT_MS + 30_000)
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

    // The job must actually run: a 202 is not a result.
    const done = await waitForJobTerminal(request, jobId, JOB_FINISH_TIMEOUT_MS)
    expect(
      done.status,
      `enrichment job ${jobId} ended ${done.status} (error_reason=${done.error_reason ?? '-'})`,
    ).toBe('succeeded')
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
