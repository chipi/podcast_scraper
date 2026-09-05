import { describe, it, expect } from 'vitest'
import {
  ACCENT_TEXT_BG,
  MIN_CONTRAST,
  SURFACE_BG,
  clampToContrast,
  compositeOver,
  contrastRatio,
  relativeLuminance,
} from './contrast'

// The overlay-composited surfaces axe measured accent text against on the player. #232125 =
// `--lp-surface` + 6% `--lp-overlay`; #1c1a1d = `--lp-canvas` + overlay. ACCENT_TEXT_BG (#2c2830 =
// elevated + overlay) is the lightest, so clearing it clears these too.
const SURFACE_PLUS_OVERLAY = '#232125'
const CANVAS_PLUS_OVERLAY = '#1c1a1d'

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

  it('lifts a too-dark artwork colour until it meets ≥4.5:1 on the accent-text bg', () => {
    const tooDark = '#3a1f14' // a deep, low-luminance brown-orange
    expect(contrastRatio(tooDark, ACCENT_TEXT_BG)).toBeLessThan(MIN_CONTRAST)
    const fixed = clampToContrast(tooDark)
    expect(fixed).not.toBe(tooDark)
    expect(contrastRatio(fixed, ACCENT_TEXT_BG)).toBeGreaterThanOrEqual(MIN_CONTRAST)
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

describe('clamp target must be the overlay-composited surface, not bare --lp-surface (#1598 axe)', () => {
  it('reproduces the shipped failure: an accent legible on --lp-surface fails on surface+overlay', () => {
    // #668b55 is the muted green a real episode artwork produced. It cleared 4.5:1 on the bare
    // surface (why the old clamp stopped there) but axe measured it at 4.09:1 on #232125.
    const shipped = '#668b55'
    expect(contrastRatio(shipped, SURFACE_BG)).toBeGreaterThanOrEqual(MIN_CONTRAST)
    expect(contrastRatio(shipped, SURFACE_PLUS_OVERLAY)).toBeLessThan(MIN_CONTRAST)
  })

  it('clamping against ACCENT_TEXT_BG clears 4.5:1 on EVERY surface accent text sits on', () => {
    // Any muted artwork colour, once clamped, must pass on the lightest bg and therefore all darker.
    for (const raw of ['#668b55', '#3a1f14', '#2e4a6b', '#555']) {
      const fixed = clampToContrast(raw)
      for (const bg of [ACCENT_TEXT_BG, SURFACE_PLUS_OVERLAY, CANVAS_PLUS_OVERLAY, SURFACE_BG]) {
        expect(contrastRatio(fixed, bg)).toBeGreaterThanOrEqual(MIN_CONTRAST)
      }
    }
  })

  it('the OLD target (bare surface) was insufficient — it leaves <4.5:1 on surface+overlay', () => {
    // Locks in WHY the target moved: clamping only against #161419 reproduces the bug.
    const underClamped = clampToContrast('#3a5c2e', SURFACE_BG, MIN_CONTRAST)
    expect(contrastRatio(underClamped, SURFACE_BG)).toBeGreaterThanOrEqual(MIN_CONTRAST)
    expect(contrastRatio(underClamped, SURFACE_PLUS_OVERLAY)).toBeLessThan(MIN_CONTRAST)
  })
})

describe('compositeOver derives the clamp background instead of assuming it (#1949)', () => {
  it('reproduces the hard-coded ACCENT_TEXT_BG from the shipping tokens', () => {
    // The load-bearing assertion. `ACCENT_TEXT_BG` used to be a number someone worked out by hand
    // and wrote down; `accentTextBg()` now computes the same thing from whatever `--lp-elevated`
    // and `--lp-overlay` currently hold, so a visual direction that changes the ground changes the
    // clamp target with it. If those two ever stop agreeing on the SHIPPING tokens, the derivation
    // has drifted from the value the #1598 fix was verified against, and this fails.
    expect(compositeOver('rgba(244, 241, 234, 0.06)', '#1f1b24')).toBe(ACCENT_TEXT_BG)
  })

  it('composites toward the overlay colour as alpha rises', () => {
    expect(compositeOver('rgba(255, 255, 255, 0)', '#1f1b24')).toBe('#1f1b24')
    expect(compositeOver('rgba(255, 255, 255, 1)', '#1f1b24')).toBe('#ffffff')
  })

  it('handles a light ground, which is the case the constant could not express', () => {
    // Paper's tokens: white elevated under a 6% near-black overlay. The result must stay light —
    // clamping an artwork accent against this walks its lightness DOWN, which is the inversion the
    // hard-coded dark constant made impossible.
    const bg = compositeOver('rgba(23, 20, 15, 0.06)', '#ffffff')!
    expect(relativeLuminance(bg)).toBeGreaterThan(0.8)
    expect(contrastRatio(clampToContrast('#4198c8', bg), bg)).toBeGreaterThanOrEqual(MIN_CONTRAST)
  })

  it('returns null for anything it cannot parse, so callers keep the safe constant', () => {
    expect(compositeOver('rgba(1,2,3,0.5)', 'not-a-colour')).toBeNull()
    expect(compositeOver('var(--nope)', '#1f1b24')).toBeNull()
  })
})

describe('compositeOver reads the form the BUNDLE actually ships (#1949)', () => {
  it('accepts 8-digit hex, which is what the minifier turns our rgba tokens into', () => {
    // `--lp-overlay: rgba(244, 241, 234, 0.06)` as authored; `#f4f1ea0f` as Lightning CSS emits it.
    // Both must land on the same background, or the derivation works in dev and silently falls
    // back to the hard-coded constant in production — which is the bug this test was added for,
    // after the fix shipped green and changed nothing in a real browser.
    expect(compositeOver('#f4f1ea0f', '#1f1b24')).toBe(ACCENT_TEXT_BG)
    expect(compositeOver('rgba(244, 241, 234, 0.06)', '#1f1b24')).toBe(
      compositeOver('#f4f1ea0f', '#1f1b24'),
    )
  })

  it('accepts 4-digit shorthand hex and 3-digit backgrounds', () => {
    // Paper's tokens minify to `--lp-elevated: #fff`.
    expect(compositeOver('#0000', '#fff')).toBe('#ffffff')
    expect(compositeOver('#000f', '#fff')).toBe('#000000')
  })
})
