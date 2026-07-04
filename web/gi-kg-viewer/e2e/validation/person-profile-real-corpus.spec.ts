/**
 * #1076 chunk 3 — Tier-3 real-corpus validation for the Person Profile +
 * Position Tracker viewer surfaces shipped in #1048 + #1049 + #1050.
 *
 * Per ADR-095 the Tier-3 spec runs against a running `make serve` stack
 * and an operator-supplied corpus. Closes the PRD-029 acceptance loop:
 * "Person Profile renders for any Person id present in the prod-v2
 * corpus (99 eps)" — using a real Person id pulled from the corpus's
 * KG artifacts rather than a synthetic stub.
 *
 * Run:
 *
 *   make ci-ui-validation CORPUS=/abs/path/to/your/corpus
 *
 * The shipped reference corpus (post-2026-06-23) lives at
 * `.test_outputs/manual/prod-v2/corpus`. If your local copy is at
 * schema_version "2.0" (pre-RFC-097 chunk 9), migrate it first:
 *
 *   for f in $CORPUS/feeds/*\/run_*\/metadata/*.gi.json; do
 *     .venv/bin/python scripts/migrate_gi_to_v3.py --in "$f" --out "$f"
 *     .venv/bin/python scripts/compute_gi_position_hints.py --in "$f" --out "$f"
 *   done
 *   for f in $CORPUS/feeds/*\/run_*\/metadata/*.kg.json; do
 *     .venv/bin/python scripts/migrate_kg_entity_to_person_org.py --in "$f" --out "$f"
 *   done
 *
 * The spec asserts on the v3 shape; a v2 corpus will surface a clean
 * failure pointing back at the migration steps above.
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test, type Page } from '@playwright/test'

import { statusBarCorpusPathInput } from '../helpers'

const CORPUS_PATH = process.env.CORPUS_PATH ?? ''

if (!CORPUS_PATH) {
  test.describe('Tier-3 Person Profile (skipped — no CORPUS_PATH)', () => {
    test.skip(
      true,
      'Tier-3 Person Profile validation requires CORPUS_PATH. ' +
        'Run via `make ci-ui-validation CORPUS=/abs/path/to/your/corpus`.',
    )
  })
}

/** Pull a real Person id from one of the corpus's KG artifacts. Cached
 *  so we don't re-walk the filesystem per test. */
function pickRealPersonIdFromCorpus(): { personId: string; sourcePath: string } {
  // Walk shallow: most corpora have `feeds/<feed>/run_*/metadata/*.kg.json`.
  // We don't want to walk a 4 GB corpus inside the spec; pull the first
  // viable KG artifact via a Node-level scan of the first feed.
  const fs = require('node:fs') as typeof import('node:fs')

  function findFirstKgUnder(dir: string): string | null {
    if (!fs.existsSync(dir)) return null
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const next = join(dir, entry.name)
      if (entry.isFile() && entry.name.endsWith('.kg.json')) return next
      if (entry.isDirectory()) {
        const found = findFirstKgUnder(next)
        if (found) return found
      }
    }
    return null
  }

  const kgPath = findFirstKgUnder(CORPUS_PATH)
  if (!kgPath) {
    throw new Error(
      `No .kg.json found under ${CORPUS_PATH}. Confirm the corpus is the ` +
        'shape the pipeline writes (feeds/<feed>/run_*/metadata/*.kg.json).',
    )
  }

  const body = JSON.parse(readFileSync(kgPath, 'utf-8')) as {
    nodes?: Array<{ id?: string; type?: string }>
  }
  const personNode = (body.nodes ?? []).find((n) => n.type === 'Person')
  if (!personNode?.id) {
    throw new Error(
      `KG artifact ${kgPath} has no Person node. Either pick a corpus that ` +
        'includes speaker detection output, or pre-load the spec with a ' +
        'known Person id.',
    )
  }
  return { personId: String(personNode.id), sourcePath: kgPath }
}

async function loadCorpusAndOpenGraph(page: Page): Promise<void> {
  await page.goto('/')
  const input = statusBarCorpusPathInput(page)
  await input.waitFor({ state: 'visible', timeout: 15_000 })
  await input.fill(CORPUS_PATH)
  await input.press('Enter').catch(() => {})
  // Switch to Graph tab + wait for the canvas to mount.
  await page
    .getByRole('navigation', { name: 'Main views' })
    .getByRole('button', { name: 'Graph' })
    .click()
  await expect(page.locator('.graph-canvas')).toBeVisible({ timeout: 60_000 })
}

test.describe('Tier-3 — Person Profile against a real prod-v2 corpus (#1076 chunk 3)', () => {
  test('PersonLandingView renders + Position Tracker tab operates against a real Person', async ({
    page,
  }) => {
    const { personId, sourcePath } = pickRealPersonIdFromCorpus()
    test.info().annotations.push({
      type: 'corpus-person',
      description: `personId=${personId} from ${sourcePath}`,
    })

    // Capture console errors — a real-corpus session that fires fatal
    // errors during PersonLandingView mount means the v2/v3 shape
    // contract drifted in the wild.
    const consoleErrors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => consoleErrors.push(err.message))

    await loadCorpusAndOpenGraph(page)

    // Surface the Person via the DEV-gated subject store hook (same
    // pattern stack-person-profile.spec.ts uses; sidesteps Explore
    // data dependency).
    await page.evaluate((pid) => {
      const win = window as unknown as {
        __GIKG_SUBJECT__?: { focusPerson?: (id: string) => void }
      }
      const hook = win.__GIKG_SUBJECT__
      if (!hook || typeof hook.focusPerson !== 'function') {
        throw new Error('__GIKG_SUBJECT__.focusPerson hook unavailable')
      }
      hook.focusPerson(pid)
    }, personId)

    // === Strict shell — the embedded Person node view (post node-view fold) ===
    // The standalone Person rail is retired; PLV is folded embedded into the
    // NodeDetail rail, which owns the header + tabs (PLV's own name header +
    // internal tablist are hidden via `v-if="!embedded"`).
    const rail = page.getByTestId('graph-node-detail-rail')
    await expect(page.getByTestId('person-landing-view')).toBeVisible({
      timeout: 15_000,
    })
    await expect(rail.getByRole('heading').first()).toContainText('Person')
    await expect(page.getByTestId('node-detail-rail-tab-details')).toHaveText(/Details/)
    await expect(page.getByTestId('person-landing-panel-profile')).toBeVisible()

    // Positions rail tab surfaces the positions lens view.
    await page.getByTestId('node-detail-rail-tab-position-tracker').click()
    await expect(page.getByTestId('person-landing-positions-view')).toBeVisible()

    // === Rich-data path — what a v3 prod corpus should produce ===
    //
    // The acceptance loop in PRD-029 expects the "topic → Position Tracker"
    // entry point (Insights voiced, ranked over MENTIONS_PERSON ∩ ABOUT) to
    // populate for ANY Person id from prod-v2. It now lives in the Positions
    // tab's default "By topic" lens. We assert it strictly because prod-v2
    // (post-RFC-097 chunk 9) should consistently emit MENTIONS_PERSON ∩ ABOUT.
    //
    // If this fails on a corpus you migrated from v2.0: confirm
    // add_insight_entity_edges has been re-run (the migration scripts
    // alone don't synthesize MENTIONS_PERSON; the typed-mentions
    // post-pass is a separate workflow step).
    const insightsVoiced = page.getByTestId('person-landing-insights-voiced')
    await expect(
      insightsVoiced,
      'PersonLandingView insights-voiced empty for a real Person id in the ' +
        'corpus. Either the corpus is pre-RFC-097 v3 (run the migration ' +
        'steps in this spec\'s header) OR the typed-MENTIONS post-pass did ' +
        'not fire on this corpus (re-run add_insight_entity_edges).',
    ).toBeVisible({ timeout: 10_000 })

    // Click first topic → Position Tracker arc (NodeDetail keeps the Positions
    // tab active so the arc is visible for that pair).
    await page.getByTestId('person-landing-insights-voiced-topic-button').first().click()
    await expect(page.getByTestId('position-tracker-arc')).toBeVisible({
      timeout: 5_000,
    })
    await expect(page.getByTestId('position-tracker-row').first()).toBeVisible()

    // Console-error gate. Real-corpus sessions ARE allowed to log some
    // benign warnings (vite HMR noise, deprecation notes); a fatal
    // error means the v3 shape contract drifted.
    if (consoleErrors.length) {
      const fatal = consoleErrors.filter(
        (e) =>
          !/HMR|deprecated|Vite|dmn_chk.*invalid domain|rejected for invalid domain/i.test(
            e,
          ) && !/"notify",\s*\w+ is null/i.test(e),
      )
      expect(
        fatal,
        `Fatal console errors during real-corpus PersonLandingView walk:\n${fatal.join(
          '\n',
        )}`,
      ).toEqual([])
    }
  })
})
