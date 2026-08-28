import { describe, it, expect, beforeEach } from 'vitest'
import { vibrantColorFromPixels, deriveShowAccent } from './accent'
import { contrastRatio, MIN_CONTRAST, ACCENT_TEXT_BG } from './contrast'

/** N RGBA pixels of one colour, as the flat buffer `getImageData` returns. */
function pixels(rgba: [number, number, number, number], count: number): number[] {
  const out: number[] = []
  for (let i = 0; i < count; i++) out.push(...rgba)
  return out
}

describe('vibrantColorFromPixels (#1598)', () => {
  it('returns the dominant vivid colour of a solid artwork', () => {
    const hex = vibrantColorFromPixels(pixels([255, 106, 61, 255], 100))
    expect(hex).not.toBeNull()
    const n = (hex as string).replace('#', '')
    const r = parseInt(n.slice(0, 2), 16)
    const g = parseInt(n.slice(2, 4), 16)
    const b = parseInt(n.slice(4, 6), 16)
    expect(r).toBeGreaterThan(g)
    expect(g).toBeGreaterThan(b) // orange
  })

  it('lets a small vivid subject outvote a large muted background', () => {
    const grey = pixels([130, 130, 132, 255], 200) // near-grey, below MIN_SAT → ignored
    const teal = pixels([20, 180, 170, 255], 20) // small but vivid
    expect(vibrantColorFromPixels([...grey, ...teal])).not.toBeNull()
  })

  it('returns null when there is no colour worth accenting (all grey)', () => {
    expect(vibrantColorFromPixels(pixels([128, 128, 128, 255], 100))).toBeNull()
  })

  it('ignores near-transparent pixels', () => {
    expect(vibrantColorFromPixels(pixels([255, 106, 61, 10], 100))).toBeNull()
  })

  it('ignores near-black and near-white pixels (no usable hue)', () => {
    expect(vibrantColorFromPixels(pixels([8, 8, 10, 255], 100))).toBeNull()
    expect(vibrantColorFromPixels(pixels([250, 250, 252, 255], 100))).toBeNull()
  })
})

describe('deriveShowAccent applies + clamps + falls back (#1598)', () => {
  let el: HTMLElement
  const accent = (): string => el.style.getPropertyValue('--lp-accent')

  beforeEach(() => {
    el = document.createElement('div')
  })

  it('clears to the brand default when there is no artwork', async () => {
    el.style.setProperty('--lp-accent', '#123456')
    await deriveShowAccent(null, el)
    expect(accent()).toBe('') // removed → CSS falls through to --lp-brand-default
  })

  it('clears to the brand default when extraction fails', async () => {
    el.style.setProperty('--lp-accent', '#123456')
    await deriveShowAccent('art.png', el, async () => null)
    expect(accent()).toBe('')
  })

  it('sets a contrast-clamped accent from a too-dark artwork colour', async () => {
    await deriveShowAccent('art.png', el, async () => '#3a1f14')
    const applied = accent()
    expect(applied).not.toBe('')
    expect(applied).not.toBe('#3a1f14') // it was lifted
    expect(contrastRatio(applied, ACCENT_TEXT_BG)).toBeGreaterThanOrEqual(MIN_CONTRAST)
  })

  it('changes the token per show', async () => {
    await deriveShowAccent('a.png', el, async () => '#3a1f14') // orange-ish
    const first = accent()
    await deriveShowAccent('b.png', el, async () => '#14203a') // blue-ish
    const second = accent()
    expect(first).not.toBe('')
    expect(second).not.toBe('')
    expect(first).not.toBe(second)
  })
})
