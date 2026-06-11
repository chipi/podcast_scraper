// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import SemanticSearchTip from './SemanticSearchTip.vue'

// SemanticSearchTip is a static info aside (no props, no emits, no store). The
// tests assert the rendered structure / content that operators rely on.

function mountTip() {
  return mount(SemanticSearchTip, { attachTo: document.body })
}

describe('SemanticSearchTip', () => {
  it('renders an aside labelled for semantic search tips', () => {
    const w = mountTip()
    const aside = w.get('aside')
    expect(aside.attributes('aria-label')).toBe('Semantic search tips')
  })

  it('renders the heading and the four explanatory bullets', () => {
    const w = mountTip()
    expect(w.text()).toContain('How semantic search works')
    expect(w.findAll('ul > li')).toHaveLength(4)
  })

  it('documents the FAISS index location and the build command', () => {
    const w = mountTip()
    const text = w.text()
    expect(text).toContain('FAISS index')
    // The build command code snippet.
    expect(text).toContain('podcast index')
  })

  it('explains the embedding-model parity requirement', () => {
    const w = mountTip()
    expect(w.text()).toContain('embedded with the same model as the index')
  })

  it('documents the G and L result-card affordances', () => {
    const w = mountTip()
    const strongs = w.findAll('strong').map((s) => s.text())
    expect(strongs).toContain('G')
    expect(strongs).toContain('L')
    expect(w.text()).toContain('subject panel')
  })

  it('renders the artifact-type and metadata path code snippets', () => {
    const w = mountTip()
    const codeText = w.findAll('code').map((c) => c.text().trim())
    expect(codeText).toContain('.gi.json')
    expect(codeText).toContain('.kg.json')
    expect(codeText).toContain('source_metadata_relative_path')
  })

  it('mentions the filterable metadata facets', () => {
    const w = mountTip()
    expect(w.text()).toContain('doc types, feed, date, speaker, or grounded insights')
  })
})
