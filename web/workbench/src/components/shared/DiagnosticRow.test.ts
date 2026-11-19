// @vitest-environment happy-dom
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import DiagnosticRow from './DiagnosticRow.vue'

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

describe('DiagnosticRow.vue — mount behaviour', () => {
  const mountRow = (props: Record<string, unknown> = {}) =>
    mount(DiagnosticRow, {
      props: { label: 'Bridge', value: 'partition-a', ...props },
      attachTo: document.body,
    })

  it('renders the label in a <dt> and the value in a <dd>', () => {
    const w = mountRow()
    expect(w.get('dt').text()).toBe('Bridge')
    expect(w.get('dd').text()).toBe('partition-a')
  })

  it('renders a numeric value by stringifying it', () => {
    const w = mountRow({ value: 42 })
    expect(w.get('dd').text()).toBe('42')
  })

  it('renders a zero numeric value (no falsy drop)', () => {
    const w = mountRow({ value: 0 })
    expect(w.get('dd').text()).toBe('0')
  })

  it('renders no chip by default (no kind/badge)', () => {
    const w = mountRow()
    expect(w.find('span').exists()).toBe(false)
  })

  it('renders no chip for the explicit "default" kind even with a badge', () => {
    const w = mountRow({ kind: 'default', badge: 'ml' })
    expect(w.find('span').exists()).toBe(false)
  })

  it('renders no chip when a non-default kind is set but the badge is missing', () => {
    const w = mountRow({ kind: 'info' })
    expect(w.find('span').exists()).toBe(false)
  })

  it('renders the info chip with badge text and the info colour class', () => {
    const w = mountRow({ kind: 'info', badge: 'ml' })
    const chip = w.get('span')
    expect(chip.text()).toBe('ml')
    expect(chip.classes().join(' ')).toContain('text-primary')
  })

  it('renders the warning chip colour class', () => {
    const w = mountRow({ kind: 'warning', badge: 'filtered' })
    expect(w.get('span').classes().join(' ')).toContain('text-warning')
  })

  it('renders the success chip colour class', () => {
    const w = mountRow({ kind: 'success', badge: 'ok' })
    expect(w.get('span').classes().join(' ')).toContain('text-success')
  })

  it('does not set a title attribute when no tooltip is given', () => {
    const w = mountRow()
    expect(w.get('[data-testid], div').attributes('title')).toBeUndefined()
  })

  it('maps tooltip to the native title attribute', () => {
    const w = mountRow({ tooltip: 'hover help' })
    // The outermost row div carries the title.
    expect(w.element.getAttribute('title')).toBe('hover help')
  })

  it('applies the dataTestid hook to the row', () => {
    const w = mountRow({ dataTestid: 'diag-bridge' })
    expect(w.find('[data-testid="diag-bridge"]').exists()).toBe(true)
  })

  it('renders an empty-string value without crashing', () => {
    const w = mountRow({ value: '' })
    expect(w.get('dd').text()).toBe('')
  })
})
