/**
 * Tier-2 production-shaped matrix helpers (RFC-086 / ADR-095).
 *
 * Serves a checked-in fixture extracted from a real corpus snapshot
 * (operator-supplied at fixture-build time — see
 * ``scripts/build_production_shaped_fixture.py``). The matrix rows
 * exercise scale + timing conditions the 1-episode fast-matrix fixture
 * cannot reach:
 *
 * - 9 episodes across 5 feeds (~270 cy nodes after GI+KG merge)
 * - 150 topic clusters defined; compound parents formed for any cluster
 *   whose members are present in cy
 * - GI + KG both populated per episode → triggers KG second-wave merge
 * - 1–3 s layout times → exposes supersession races
 *
 * Re-exports ``assertHandoffApplied`` / ``assertFsmEventEnvelope`` /
 * ``captureConsoleErrors`` from the Tier-1 helpers so the same 6-point
 * assertion contract applies. The only difference is the fixture.
 *
 * Refresh fixture from a fresher real corpus:
 *   python scripts/build_production_shaped_fixture.py \
 *     --corpus /path/to/real-corpus \
 *     --output web/gi-kg-viewer/e2e/fixtures/production-shaped
 */

import { readFileSync } from 'node:fs'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Page } from '@playwright/test'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FIXTURE_ROOT = path.resolve(__dirname, '..', 'fixtures', 'production-shaped')

function readFixtureJson(rel: string): unknown {
  return JSON.parse(readFileSync(path.join(FIXTURE_ROOT, rel), 'utf-8'))
}

function readFixtureText(rel: string): string {
  return readFileSync(path.join(FIXTURE_ROOT, rel), 'utf-8')
}

interface ManifestEpisode {
  episode_id: string
  feed_id: string
  metadata_relative_path: string
  gi_relative_path: string | null
  kg_relative_path: string | null
  publish_date: string | null
}

interface Manifest {
  schema_version: string
  source_corpus: string
  picked: {
    feeds: string[]
    episodes: ManifestEpisode[]
    queries: string[]
  }
}

let cachedManifest: Manifest | null = null
function manifest(): Manifest {
  if (!cachedManifest) {
    cachedManifest = readFixtureJson('manifest.json') as Manifest
  }
  return cachedManifest
}

/**
 * Set up API mocks against the production-shaped fixture. Mirrors the
 * shape of ``setupHandoffMatrixMocks`` from the Tier-1 helpers but serves
 * realistic data volumes. Optional toggles control which optional
 * endpoints (search, dashboard) get rich responses vs empty stubs.
 */
export async function setupProductionShapedMocks(
  page: Page,
  opts?: {
    search?: boolean
    dashboard?: boolean
    /**
     * Artificial latency (ms) on artifact (GI/KG) fetches. Simulates real
     * backend I/O so timing-sensitive bugs (V5-class supersession races)
     * have a window to fire. Default 0 — pure mock speed.
     */
    artifactLatencyMs?: number
  },
): Promise<void> {
  const artifactDelayMs = opts?.artifactLatencyMs ?? 0
  const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))
  const m = manifest()
  const epIds = new Set(m.picked.episodes.map((e) => e.episode_id))

  // Map metadata_relative_path → episode for the detail endpoint.
  const detailsByPath = readFixtureJson('corpus/episode-details.json') as Record<
    string,
    unknown
  >
  // Map episode_id → artifact text for GI/KG endpoints.
  const giByEpId = new Map<string, string>()
  const kgByEpId = new Map<string, string>()
  for (const ep of m.picked.episodes) {
    if (ep.gi_relative_path) {
      try {
        giByEpId.set(ep.episode_id, readFixtureText(`artifacts/${ep.episode_id}.gi.json`))
      } catch {
        /* skip — fixture may have skipped this episode */
      }
    }
    if (ep.kg_relative_path) {
      try {
        kgByEpId.set(ep.episode_id, readFixtureText(`artifacts/${ep.episode_id}.kg.json`))
      } catch {
        /* skip */
      }
    }
  }
  void epIds // currently unused; surfaced for future filter logic

  // Pre-load static endpoint payloads.
  const feedsBody = readFixtureText('corpus/feeds.json')
  const episodesBody = readFixtureText('corpus/episodes.json')
  const digestBody = readFixtureText('corpus/digest.json')
  const topicClustersBody = readFixtureText('corpus/topic-clusters.json')
  const statsBody = readFixtureText('corpus/stats.json')
  const coverageBody = readFixtureText('corpus/coverage.json')
  const personsTopBody = readFixtureText('corpus/persons-top.json')
  const runsSummaryBody = readFixtureText('corpus/runs-summary.json')
  const indexStatsBody = readFixtureText('corpus/index-stats.json')
  const searchByQuery = readFixtureJson('search/results-by-query.json') as Record<
    string,
    unknown
  >

  // --- API mocks ---

  // Match ``/api/health`` AND ``/api/health?path=…`` (the debounced
  // re-probe on corpus-path change fires the second form — see
  // shell.ts §S4-shell followup).
  await page.route('**/api/health**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        corpus_library_api: true,
        corpus_digest_api: true,
        cil_queries_api: true,
        search_api: true,
      }),
    }),
  )

  // /api/artifacts?path=... — list all artifact files for the fixture.
  // The viewer's auto-load reads this to know which episodes to fetch.
  const artifactEntries = m.picked.episodes.flatMap((ep) => {
    const out: Array<Record<string, unknown>> = []
    if (giByEpId.has(ep.episode_id) && ep.gi_relative_path) {
      out.push({
        name: path.basename(ep.gi_relative_path),
        relative_path: ep.gi_relative_path,
        kind: 'gi',
        size_bytes: giByEpId.get(ep.episode_id)?.length ?? 0,
        mtime_utc: '2026-05-01T00:00:00Z',
        publish_date: ep.publish_date ?? '2026-05-01',
      })
    }
    if (kgByEpId.has(ep.episode_id) && ep.kg_relative_path) {
      out.push({
        name: path.basename(ep.kg_relative_path),
        relative_path: ep.kg_relative_path,
        kind: 'kg',
        size_bytes: kgByEpId.get(ep.episode_id)?.length ?? 0,
        mtime_utc: '2026-05-01T00:00:00Z',
        publish_date: ep.publish_date ?? '2026-05-01',
      })
    }
    return out
  })
  await page.route('**/api/artifacts?**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/production-shaped',
        artifacts: artifactEntries,
      }),
    }),
  )

  await page.route('**/api/corpus/feeds**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: feedsBody }),
  )
  await page.route('**/api/corpus/episodes?**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: episodesBody }),
  )
  await page.route('**/api/corpus/episodes/detail**', (r) => {
    const url = new URL(r.request().url())
    const rel = url.searchParams.get('metadata_relpath') ||
      url.searchParams.get('metadata_relative_path') || ''
    const detail = detailsByPath[rel]
    if (detail) {
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(detail),
      })
    } else {
      r.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'episode not found in fixture' }),
      })
    }
  })

  // Artifact file endpoints serve the GI/KG JSON via path lookup.
  for (const ep of m.picked.episodes) {
    if (ep.gi_relative_path && giByEpId.has(ep.episode_id)) {
      const giBody = giByEpId.get(ep.episode_id)!
      const giUrlGlob = `**/api/artifacts/${encodeURIComponent(ep.gi_relative_path).replace(/%2F/gi, '/')}**`
      await page.route(giUrlGlob, async (r) => {
        if (artifactDelayMs > 0) await sleep(artifactDelayMs)
        await r.fulfill({ status: 200, contentType: 'application/json', body: giBody })
      })
      // Some viewer paths fetch without URL-encoding — also register the
      // raw path glob.
      await page.route(`**/api/artifacts/${ep.gi_relative_path}**`, async (r) => {
        if (artifactDelayMs > 0) await sleep(artifactDelayMs)
        await r.fulfill({ status: 200, contentType: 'application/json', body: giBody })
      })
    }
    if (ep.kg_relative_path && kgByEpId.has(ep.episode_id)) {
      const kgBody = kgByEpId.get(ep.episode_id)!
      const kgUrlGlob = `**/api/artifacts/${encodeURIComponent(ep.kg_relative_path).replace(/%2F/gi, '/')}**`
      await page.route(kgUrlGlob, async (r) => {
        if (artifactDelayMs > 0) await sleep(artifactDelayMs)
        await r.fulfill({ status: 200, contentType: 'application/json', body: kgBody })
      })
      await page.route(`**/api/artifacts/${ep.kg_relative_path}**`, async (r) => {
        if (artifactDelayMs > 0) await sleep(artifactDelayMs)
        await r.fulfill({ status: 200, contentType: 'application/json', body: kgBody })
      })
    }
  }

  await page.route('**/api/corpus/topic-clusters**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: topicClustersBody,
    }),
  )
  await page.route('**/api/corpus/digest**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: digestBody }),
  )
  await page.route('**/api/corpus/stats**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: statsBody }),
  )
  await page.route('**/api/corpus/coverage**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: coverageBody }),
  )
  await page.route('**/api/corpus/persons/top**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: personsTopBody }),
  )
  await page.route('**/api/corpus/runs/summary**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: runsSummaryBody }),
  )
  await page.route('**/api/corpus/episodes/similar**', (r) =>
    r.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        path: '/mock/production-shaped',
        source_metadata_relative_path: '',
        query_used: '',
        items: [],
        error: null,
        detail: null,
      }),
    }),
  )
  await page.route('**/api/index/stats**', (r) =>
    r.fulfill({ status: 200, contentType: 'application/json', body: indexStatsBody }),
  )

  if (opts?.search) {
    // Search: pick the first recorded query's results for any incoming query.
    const firstResults = Object.values(searchByQuery)[0] ?? {
      query: '',
      results: [],
    }
    await page.route('**/api/search?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(firstResults),
      }),
    )
  }
}

/** First episode in the fixture — handy target for matrix rows. */
export function firstFixtureEpisodeId(): string {
  return manifest().picked.episodes[0]!.episode_id
}

export function firstFixtureEpisodeMetadataPath(): string {
  return manifest().picked.episodes[0]!.metadata_relative_path
}

/** All episodes — useful for hot-state tests that need a SECOND target. */
export function fixtureEpisodes(): ManifestEpisode[] {
  return manifest().picked.episodes
}
