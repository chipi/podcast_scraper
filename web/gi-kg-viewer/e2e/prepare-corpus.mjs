#!/usr/bin/env node
/**
 * Prepare the corpus the viewer e2e API serves (#1619). Runs as the FIRST half of that server's
 * `webServer.command`, not as Playwright's `globalSetup`.
 *
 * That distinction is the whole point of this file. Playwright starts `webServer` BEFORE it runs
 * `globalSetup`, so seeding from globalSetup loses the race every time — the server came up first,
 * found no corpus and exited 2:
 *
 *     [WebServer] Output directory does not exist or is not a directory: …/.e2e-corpus/v3
 *     Error: Process from config.webServer was not able to start. Exit code: 2
 *
 * Chaining it into the server's own command is what makes the precondition unskippable: the server
 * cannot start before this has finished. (When the container from `e2e/run-local-stack.sh` is
 * already serving 8012, `reuseExistingServer` means the whole command — this included — never
 * runs, which is correct: that path seeds its own copy.)
 *
 * Two things happen, both idempotent.
 *
 * 1. A DISPOSABLE corpus copy. The operator plane writes into whatever corpus it is given:
 *    `GET /api/operator-config` creates `viewer_operator.yaml` when absent, and the jobs API
 *    creates `.viewer/jobs.jsonl.lock`. `.gitignore` force-includes
 *    `tests/fixtures/app-validation-corpus/**`, so serving the fixture directly leaves a dirty
 *    tracked tree — and the next run starts from a corpus the last one mutated, so "fresh corpus"
 *    assertions quietly stop being fresh.
 *
 * 2. The two-tier search index. The corpus is committed; `search/lance_index/` is not (binary,
 *    coupled to a lance format version). Ten specs reach `/api/search`, and routes branch on
 *    `has_index` — so an absent index does not fail loudly, it changes which claim the page
 *    surfaces. Needs the `[search]` extras and a cached MiniLM: `index-two-tier` runs with
 *    `allow_download=False`. CI provides both before invoking Playwright (see
 *    `.github/workflows/python-app.yml`, job `viewer-e2e`).
 */
import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const viewerRoot = resolve(here, '..')
const repoRoot = resolve(viewerRoot, '..', '..')
const source = join(repoRoot, 'tests', 'fixtures', 'app-validation-corpus', 'v3')
const workdir = process.env.E2E_CORPUS_WORKDIR || join(viewerRoot, '.e2e-corpus')
const corpus = join(workdir, 'v3')

if (!existsSync(source)) {
  console.error(`[prepare-corpus] missing fixture corpus: ${source}`)
  process.exit(1)
}

rmSync(workdir, { recursive: true, force: true })
mkdirSync(workdir, { recursive: true })
cpSync(source, corpus, { recursive: true })
console.log(`[prepare-corpus] seeded a disposable corpus copy at ${corpus}`)

// 3. RE-DATE the query log. The fixture's `search/query_log.jsonl` carries ABSOLUTE timestamps
//    (46 entries, all 2026-08-03) while `GET /api/corpus/query-activity` only counts the last
//    `days` — 30 by default. On 2026-09-03 the whole log fell one day outside that window,
//    `total` went to 0, and `dashboard.spec.ts` FR6.2 started failing on every branch at once,
//    for a reason no commit caused. A fixed date behind a rolling window is a time bomb: it
//    passes until the calendar moves, then reports a code failure that is really a calendar one.
//
//    Shift every entry by ONE offset so the newest lands now. Relative spacing is preserved, so
//    this re-dates the data without inventing any, and it happens in the DISPOSABLE COPY only —
//    the committed fixture stays byte-identical for `tests/unit/search/test_query_log.py`.
const queryLog = join(corpus, 'search', 'query_log.jsonl')
if (existsSync(queryLog)) {
  const lines = readFileSync(queryLog, 'utf8').split('\n').filter((l) => l.trim())
  const stamps = lines.map((l) => {
    try {
      return Date.parse(JSON.parse(l).ts)
    } catch {
      return NaN
    }
  })
  const newest = Math.max(...stamps.filter((n) => Number.isFinite(n)))
  if (Number.isFinite(newest)) {
    const offset = Date.now() - newest
    const shifted = lines.map((line, i) => {
      if (!Number.isFinite(stamps[i])) return line
      const rec = JSON.parse(line)
      rec.ts = new Date(stamps[i] + offset).toISOString()
      return JSON.stringify(rec)
    })
    writeFileSync(queryLog, `${shifted.join('\n')}\n`)
    const dayShift = Math.round(offset / 86_400_000)
    console.log(
      `[prepare-corpus] re-dated ${shifted.length} query-log entries (+${dayShift}d) into the activity window`,
    )
  } else {
    console.warn('[prepare-corpus] query_log.jsonl has no parsable `ts` — left as-is')
  }
}

const lanceDir = join(corpus, 'search', 'lance_index')
if (existsSync(lanceDir) && readdirSync(lanceDir).length > 0) {
  console.log('[prepare-corpus] search index already present in the copy; not rebuilding')
  process.exit(0)
}

const venvPython = join(repoRoot, '.venv', 'bin', 'python')
const interpreter = existsSync(venvPython) ? venvPython : 'python3'
console.log('[prepare-corpus] building the two-tier search index…')
execFileSync(interpreter, ['-m', 'podcast_scraper.cli', 'index-two-tier', '--output-dir', corpus], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHONPATH: join(repoRoot, 'src'),
    // index_corpus runs allow_download=False, so the model must already be in the HF cache.
    HF_HUB_OFFLINE: '1',
    TRANSFORMERS_OFFLINE: '1',
  },
})
