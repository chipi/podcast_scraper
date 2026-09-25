/**
 * Seed a DISPOSABLE copy of the app validation corpus for the e2e run.
 *
 * playwright.config.ts used to point `serve` straight at
 * `tests/fixtures/app-validation-corpus/v3`, and its comment claimed "the committed corpus
 * tree is never mutated" because APP_DATA_DIR redirects the per-user state the API writes
 * (queue, profile, interests). That covered everything except the one thing written INSIDE
 * the corpus: `search/query_log.jsonl`. Every search a spec runs appends to it, so a clean
 * `make test-app-e2e` left a dirty tracked file — 9 stray entries turned up this way, and
 * the next run then started from a corpus the last one had grown.
 *
 * The viewer suite already solved exactly this, and its own note names the second half of
 * the cost: "the next run starts from a corpus the last one mutated, so 'fresh corpus'
 * assertions quietly stop being fresh." Same fix here, so both surfaces treat the committed
 * fixture as read-only.
 *
 * Idempotent: the workdir is wiped and re-seeded each time.
 */
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const appRoot = resolve(here, "..")
const repoRoot = resolve(appRoot, "..", "..")
const source = join(repoRoot, "tests", "fixtures", "app-validation-corpus", "v3")
const workdir = process.env.APP_E2E_CORPUS_WORKDIR || join(appRoot, ".e2e-corpus")
const corpus = join(workdir, "v3")

if (!existsSync(source)) {
  console.error(`[prepare-corpus] missing fixture corpus: ${source}`)
  process.exit(1)
}

rmSync(workdir, { recursive: true, force: true })
mkdirSync(workdir, { recursive: true })
cpSync(source, corpus, { recursive: true })
console.log(`[prepare-corpus] seeded a disposable app corpus copy at ${corpus}`)
