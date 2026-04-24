import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

/**
 * Pre-#656 foundation: invariants on the shared DiagnosticRow primitive.
 *
 * Like the ``PipelineJobExplorePanel`` guard (review #12 of #666), this
 * is a source-level check — the viewer doesn't have a component mount
 * harness wired, and the primitive is small enough that the real risks
 * are "did someone add v-html" / "did someone break the variant map"
 * rather than subtle rendering bugs.
 *
 * When the broader Playwright/behavior suite lands (post-#656), these
 * invariants stay valid and a real mount test can be added alongside.
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const COMPONENT = resolve(HERE, 'DiagnosticRow.vue')
const source = readFileSync(COMPONENT, 'utf-8')

describe('DiagnosticRow.vue — shape + safety invariants', () => {
  it('renders label + value via safe interpolation (no v-html directive)', () => {
    // Check for the binding syntax specifically — ``v-html=`` or ``:v-html=`` —
    // rather than the word ``v-html`` in a doc comment (which is allowed).
    expect(source).not.toMatch(/\sv-html\s*=/)
    expect(source).not.toMatch(/\.innerHTML\s*=/)
  })

  it('uses {{ label }} and {{ value }} interpolation', () => {
    expect(source).toMatch(/\{\{\s*label\s*\}\}/)
    expect(source).toMatch(/\{\{\s*value\s*\}\}/)
  })

  it('declares all four DiagnosticKind variants', () => {
    // Regression guard: if someone drops a variant from the union or
    // the KIND_CHIP_CLASS map, the two go out of sync and a production
    // diagnostic would render unclassed.
    const variants = ['default', 'info', 'warning', 'success']
    for (const v of variants) {
      expect(source).toContain(`'${v}'`)
    }
  })

  it('chip is hidden for the default variant', () => {
    // The template condition must exclude ``default`` so non-status
    // diagnostics (plain label + value rows) don't render an empty chip.
    expect(source).toMatch(/kind\s*!==\s*['"]default['"]/)
  })

  it('binds tooltip via native title attribute, not a raw innerHTML sink', () => {
    expect(source).toMatch(/:title="tooltip \|\| undefined"/)
  })

  it('exposes dataTestid hook for e2e selection', () => {
    expect(source).toMatch(/:data-testid="dataTestid"/)
  })
})
