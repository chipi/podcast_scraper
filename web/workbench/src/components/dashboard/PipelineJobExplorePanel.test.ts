import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * #666 review #12 — XSS surface guard for PipelineJobExplorePanel.vue.
 *
 * The panel is the primary viewer surface for pipeline-job exploration: it
 * renders untrusted pipeline output (corpus manifest summaries, log tails,
 * structured run metadata) pulled from the API. The reviewer flagged that
 * none of this was unit-tested, and specifically called out the risk of
 * v-html creeping into the template over time.
 *
 * The "happy path" (mount + assert) would require wiring @vue/test-utils and
 * happy-dom across a component whose imports include Pinia, fetch helpers,
 * and several humanize utilities — infrastructure nothing else in this repo
 * has today. The immediate, high-value guarantee is: no raw HTML sinks are
 * introduced into the template. Enforce that here; when the broader test
 * harness lands, this check stays as an invariant.
 *
 * Utility transforms used by the component (humanizeJsonDocument,
 * pipelineJobLogSummary, feedRunLinking) already have dedicated .test.ts —
 * those are where the actual parsing / rendering logic lives.
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const PANEL_SFC = resolve(HERE, 'PipelineJobExplorePanel.vue')

describe('PipelineJobExplorePanel.vue — XSS surface', () => {
  const source = readFileSync(PANEL_SFC, 'utf-8')

  it('has no v-html directive', () => {
    // v-html would render arbitrary HTML from untrusted pipeline output /
    // log tails / corpus manifests — every string in this component comes
    // from the API and must be treated as untrusted. Match the binding
    // syntax (``v-html=`` / ``:v-html=``) specifically so a doc-comment
    // mention does not trip the guard.
    expect(source).not.toMatch(/\sv-html\s*=/)
  })

  it('has no innerHTML assignment', () => {
    expect(source).not.toMatch(/\.innerHTML\s*=/)
    expect(source).not.toMatch(/\.outerHTML\s*=/)
  })

  it('has no document.write', () => {
    expect(source).not.toMatch(/document\.write\b/)
  })

  it('has no eval or Function constructor', () => {
    // Belt-and-suspenders: if someone ever tries to build dynamic code from
    // log content, this fires.
    expect(source).not.toMatch(/\beval\s*\(/)
    expect(source).not.toMatch(/\bnew\s+Function\s*\(/)
  })
})
