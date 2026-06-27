import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import Sparkline from './Sparkline.vue'

describe('Sparkline', () => {
  it('draws a line + area path scaled to the viewBox', () => {
    const w = mount(Sparkline, { props: { values: [0, 1, 2, 4], width: 100, height: 40 } })
    const paths = w.findAll('path')
    expect(paths).toHaveLength(2) // area + line
    const line = paths[1].attributes('d')!
    expect(line.startsWith('M')).toBe(true)
    // The peak value (4) maps to the top (y≈1); the trough (0) to the bottom (y≈height-1).
    expect(line).toContain('M0.0,39.0') // first point at x=0, bottom
    expect(line.trimEnd().endsWith('100.0,1.0')).toBe(true) // last point at x=width, top
  })

  it('renders safely with empty values', () => {
    const w = mount(Sparkline, { props: { values: [] } })
    expect(w.findAll('path')).toHaveLength(2)
  })
})
