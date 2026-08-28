import { describe, it, expect } from 'vitest'
import {
  clampToContrast,
  contrastRatio,
  relativeLuminance,
  MIN_CONTRAST,
  SURFACE_BG,
} from './contrast'

describe('WCAG contrast primitives (#1598)', () => {
  it('luminance spans black→white', () => {
    expect(relativeLuminance('#000000')).toBeCloseTo(0, 5)
    expect(relativeLuminance('#ffffff')).toBeCloseTo(1, 5)
  })

  it('black-on-white is the maximum ratio and is order-independent', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 0)
    expect(contrastRatio('#ffffff', '#000000')).toBeCloseTo(21, 0)
  })

  it('a colour against itself is 1:1', () => {
    expect(contrastRatio(SURFACE_BG, SURFACE_BG)).toBeCloseTo(1, 5)
  })

  it('a bad hex is treated as luminance 0, never throws', () => {
    expect(relativeLuminance('not-a-colour')).toBe(0)
    expect(clampToContrast('not-a-colour')).toBe('not-a-colour')
  })
})

describe('clampToContrast lifts a low-contrast accent to legibility (#1598)', () => {
  it('leaves a colour that already clears the threshold untouched', () => {
    // Bright ember already clears 4.5:1 on the dark surface.
    const bright = '#ff6a3d'
    expect(contrastRatio(bright, SURFACE_BG)).toBeGreaterThanOrEqual(MIN_CONTRAST)
    expect(clampToContrast(bright)).toBe(bright)
  })

  it('lifts a too-dark artwork colour until it meets ≥4.5:1 on the surface', () => {
    const tooDark = '#3a1f14' // a deep, low-luminance brown-orange
    expect(contrastRatio(tooDark, SURFACE_BG)).toBeLessThan(MIN_CONTRAST)
    const fixed = clampToContrast(tooDark)
    expect(fixed).not.toBe(tooDark)
    expect(contrastRatio(fixed, SURFACE_BG)).toBeGreaterThanOrEqual(MIN_CONTRAST)
  })

  it('preserves the hue while lifting — a dark orange stays orange (r>g>b)', () => {
    const fixed = clampToContrast('#3a1f14')
    const n = fixed.replace('#', '')
    const r = parseInt(n.slice(0, 2), 16)
    const g = parseInt(n.slice(2, 4), 16)
    const b = parseInt(n.slice(4, 6), 16)
    expect(r).toBeGreaterThan(g)
    expect(g).toBeGreaterThan(b)
  })

  it('over a LIGHT surface it darkens instead of lightening', () => {
    // A pale yellow is illegible on white; the clamp must lower its lightness to gain contrast.
    const pale = '#fff7cc'
    const fixed = clampToContrast(pale, '#ffffff', MIN_CONTRAST)
    expect(contrastRatio(fixed, '#ffffff')).toBeGreaterThanOrEqual(MIN_CONTRAST)
    expect(relativeLuminance(fixed)).toBeLessThan(relativeLuminance(pale))
  })
})
