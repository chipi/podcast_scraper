import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs'
import { join, resolve } from 'node:path'

/**
 * Give the viewer suite the backend it now needs (#1619).
 *
 * Until #1619 this suite had no backend: its webServer was Vite alone and every API-dependent spec
 * route-fulfilled its own payloads. The migration moved the suite onto the real API — which worked
 * locally via ``e2e/run-local-stack.sh`` and was never wired into CI, because the ``viewer-e2e``
 * job is gated on viewer files changing and had been skipped on every run of the PR that did the
 * migration. The first run that actually executed failed 81 specs across 20 files, all of them
 * ``ECONNREFUSED`` to the API. This is the missing half.
 *
 * Two things happen here, both idempotent, so a local run and a CI run start from the same state.
 *
 * **1. A disposable corpus copy.** The operator plane WRITES into whatever corpus directory it is
 * given: ``GET /api/operator-config`` *creates* ``viewer_operator.yaml`` when absent, and the jobs
 * API creates ``.viewer/jobs.jsonl.lock``. ``.gitignore`` force-includes
 * ``tests/fixtures/app-validation-corpus/**``, so serving the fixture directly leaves a dirty
 * tracked tree after every run — and worse, the next run starts from a corpus the last one
 * mutated, so 'fresh corpus' assertions quietly stop being fresh. Copying is ~6 MB and makes each
 * run hermetic. Same rationale, and same destination, as ``run-local-stack.sh``.
 *
 * **2. The two-tier search index.** The corpus is committed but ``search/lance_index/`` is not
 * (binary, and coupled to a lance format version), and ten specs reach ``/api/search``. Routes
 * branch on ``has_index``, so an absent index does not fail loudly — it changes which claim the
 * page surfaces. Building it here makes local and CI identical instead of local passing only when
 * a stale index happens to be lying around.
 *
 * Requires the ``[search]`` extras and a cached MiniLM: ``index-two-tier`` runs with
 * ``allow_download=False``, so the model must already be in the HF cache. CI provides both before
 * invoking Playwright (see ``.github/workflows/python-app.yml``, job ``viewer-e2e``); locally the
 * dev venv has them. On a machine that cannot install ``[search]`` at all — macOS x86_64, where
 * torch publishes no wheel — use the container path (``make test-ui-e2e-live``) instead, which
 * ships its own model cache.
 */
export default function globalSetup(): void {
  // Playwright runs the config from web/gi-kg-viewer/.
  const viewerRoot = process.cwd()
  const repoRoot = resolve(viewerRoot, '..', '..')
  const source = join(repoRoot, 'tests', 'fixtures', 'app-validation-corpus', 'v3')
  const workdir = process.env.E2E_CORPUS_WORKDIR || join(viewerRoot, '.e2e-corpus')
  const corpus = join(workdir, 'v3')

  if (!existsSync(source)) {
    throw new Error(`missing fixture corpus: ${source}`)
  }

  // Re-seed every run: hermetic, and cheap enough not to bother caching.
  rmSync(workdir, { recursive: true, force: true })
  mkdirSync(workdir, { recursive: true })
  cpSync(source, corpus, { recursive: true })

  const lanceDir = join(corpus, 'search', 'lance_index')
  const hasIndex = existsSync(lanceDir) && readdirSync(lanceDir).length > 0
  if (hasIndex) return

  const python = join(repoRoot, '.venv', 'bin', 'python')
  const interpreter = existsSync(python) ? python : 'python3'
  // eslint-disable-next-line no-console
  console.log('[globalSetup] building the two-tier search index for the viewer e2e corpus…')
  execFileSync(
    interpreter,
    ['-m', 'podcast_scraper.cli', 'index-two-tier', '--output-dir', corpus],
    {
      cwd: repoRoot,
      stdio: 'inherit',
      env: {
        ...process.env,
        PYTHONPATH: join(repoRoot, 'src'),
        // index_corpus runs allow_download=False, so the model must already be cached.
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
      },
    }
  )
}
