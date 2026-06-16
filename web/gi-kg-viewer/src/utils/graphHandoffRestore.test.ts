// @vitest-environment node
import { describe, expect, it } from 'vitest'
import cytoscape from 'cytoscape'
import { resolveHandoffCandidateNode } from './graphHandoffRestore'

function cyWith(ids: string[]) {
  return cytoscape({ headless: true, elements: ids.map((id) => ({ data: { id } })) })
}

describe('resolveHandoffCandidateNode (#967)', () => {
  it('resolves an exact id', () => {
    const cy = cyWith(['g:alpha', 'k:beta'])
    expect(resolveHandoffCandidateNode(cy, 'g:alpha')?.id()).toBe('g:alpha')
    cy.destroy()
  })

  it('falls back to the alternative prefix when g: flipped to k: (KG joined after apply)', () => {
    // FSM applied ``g:alpha`` (GI-only graph); the redraw merged KG so the same logical
    // node is now ``k:alpha``. Restore must still find it.
    const cy = cyWith(['k:alpha'])
    expect(resolveHandoffCandidateNode(cy, 'g:alpha')?.id()).toBe('k:alpha')
    cy.destroy()
  })

  it('falls back from k: to g:', () => {
    const cy = cyWith(['g:alpha'])
    expect(resolveHandoffCandidateNode(cy, 'k:alpha')?.id()).toBe('g:alpha')
    cy.destroy()
  })

  it('prefixes a bare id and finds either variant', () => {
    expect(resolveHandoffCandidateNode(cyWith(['g:bare']), 'bare')?.id()).toBe('g:bare')
    expect(resolveHandoffCandidateNode(cyWith(['k:bare']), 'bare')?.id()).toBe('k:bare')
  })

  it('prefers the exact id over the prefix-flip variant', () => {
    // Both present → the exact match wins, not the flipped one.
    const cy = cyWith(['g:alpha', 'k:alpha'])
    expect(resolveHandoffCandidateNode(cy, 'g:alpha')?.id()).toBe('g:alpha')
    cy.destroy()
  })

  it('returns null for an empty/blank id or a missing node', () => {
    const cy = cyWith(['g:alpha'])
    expect(resolveHandoffCandidateNode(cy, '')).toBeNull()
    expect(resolveHandoffCandidateNode(cy, '   ')).toBeNull()
    expect(resolveHandoffCandidateNode(cy, 'g:ghost')).toBeNull()
    cy.destroy()
  })
})
