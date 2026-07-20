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
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { expect, test, type Page, type TestInfo } from '@playwright/test'

import {
  signInIsolated,
  statusBarCorpusPathInput,
} from '../helpers'

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

/** Pull a real Person id from the corpus's GI artifacts. Person nodes are RFC-097
 *  typed GI nodes (``person:…``); the KG carries the older ``Entity`` type, so we
 *  scan the GI graph — which is also where NodeDetail resolves the person view. */
function pickRealPersonIdFromCorpus(): { personId: string; sourcePath: string } {
  // We don't want to walk a huge corpus inside the spec; collect the GI
  // artifacts (feeds/<feed>/metadata/*.gi.json) and take the first Person.
  function collectGiFiles(dir: string, acc: string[]): void {
    if (!existsSync(dir)) return
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const next = join(dir, entry.name)
      if (entry.isFile() && entry.name.endsWith('.gi.json')) acc.push(next)
      else if (entry.isDirectory()) collectGiFiles(next, acc)
    }
  }

  const giFiles: string[] = []
  collectGiFiles(CORPUS_PATH, giFiles)
  for (const giPath of giFiles) {
    const body = JSON.parse(readFileSync(giPath, 'utf-8')) as {
      nodes?: Array<{ id?: string; type?: string }>
    }
    const person = (body.nodes ?? []).find(
      (n) => n.type === 'Person' && typeof n.id === 'string' && n.id.includes('person:'),
    )
    if (person?.id) return { personId: String(person.id), sourcePath: giPath }
  }
  throw new Error(
    `No GI Person node found under ${CORPUS_PATH}. Confirm the corpus has ` +
      'RFC-097 typed Person nodes (feeds/<feed>/metadata/*.gi.json).',
  )
}

async function loadCorpusAndOpenGraph(page: Page, testInfo: TestInfo): Promise<void> {
  await signInIsolated(page, 'person-profile', testInfo)
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
  }, testInfo) => {
    const { personId, sourcePath } = pickRealPersonIdFromCorpus()
    test.info().annotations.push({
      type: 'corpus-person',
      description: `personId=${personId} from ${sourcePath}`,
    })

    /* Sign in FIRST, then attach the console-error harness. Errors
     * fired during the pre-auth boot window (401s from
     * ``/api/app/auth/status`` etc) are not part of the walk's
     * contract — the walk measures the authenticated session. */
    await loadCorpusAndOpenGraph(page, testInfo)

    const consoleErrors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => consoleErrors.push(err.message))

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

    // === Rich-data path — exercised when the corpus provides it ===
    //
    // The "topic → Position Tracker" entry point (Insights voiced, ranked over
    // MENTIONS_PERSON ∩ ABOUT) lives in the Positions tab's default "By topic"
    // lens. A prod-v2 corpus (post-RFC-097 chunk 9) emits it for most speakers,
    // but this spec runs against whatever CORPUS_PATH is supplied — a synthetic
    // or sparse corpus may not produce it for an arbitrarily-picked Person. So
    // the click → arc path is exercised only when present; the strict shell
    // assertions above are the always-on contract that guards the node-view.
    //
    // If insights-voiced is unexpectedly empty on a corpus migrated from v2.0:
    // confirm add_insight_entity_edges has been re-run (the typed-MENTIONS
    // post-pass is a separate workflow step from the schema migration).
    const insightsVoiced = page.getByTestId('person-landing-insights-voiced')
    if (await insightsVoiced.isVisible({ timeout: 10_000 }).catch(() => false)) {
      // Click first topic → Position Tracker arc (NodeDetail keeps the Positions
      // tab active so the arc is visible for that pair).
      await page.getByTestId('person-landing-insights-voiced-topic-button').first().click()
      await expect(page.getByTestId('position-tracker-arc')).toBeVisible({
        timeout: 5_000,
      })
      await expect(page.getByTestId('position-tracker-row').first()).toBeVisible()
    } else {
      // eslint-disable-next-line no-console
      console.log(
        `[Tier-3 person-profile] no insights-voiced for ${personId} in this corpus ` +
          '— shell + tabs asserted strictly; rich-data path skipped (sparse corpus).',
      )
    }

    // Console-error gate. Real-corpus sessions ARE allowed to log some
    // benign warnings (vite HMR noise, deprecation notes); a fatal
    // error means the v3 shape contract drifted.
    if (consoleErrors.length) {
      const fatal = consoleErrors.filter(
        (e) =>
          !/HMR|deprecated|Vite|dmn_chk.*invalid domain|rejected for invalid domain/i.test(
            e,
          ) &&
          !/"notify",\s*\w+ is null/i.test(e) &&
          // The Vite dev server doesn't serve /favicon.ico; browsers request it
          // on every navigation and the 404 surfaces as a generic URL-less
          // "Failed to load resource: ... 404 (Not Found)". Production ships a
          // favicon, so this only affects the local/CI dev-server walk (same
          // exemption real-corpus.spec.ts documents).
          !/^Failed to load resource: the server responded with a status of 404 \(Not Found\)$/.test(
            e,
          ),
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
