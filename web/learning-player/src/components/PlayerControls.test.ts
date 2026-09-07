import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import type { InsightMarker } from '../player/insightMarkers'
import PlayerControls from './PlayerControls.vue'
import playerControlsSource from './PlayerControls.vue?raw'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountPC(props: Record<string, unknown> = {}) {
  return mount(PlayerControls, {
    props: { playing: false, currentTime: 0, duration: 100, rate: 1, ...props },
    global: { plugins: [i18n] },
  })
}

describe('PlayerControls insight-density strip (#1140)', () => {
  it('renders no density strip without markers', () => {
    expect(mountPC().find('[data-testid="player-insight-density"]').exists()).toBe(false)
  })

  it('renders one tick per marker, positioned by pct, opacity by weight, coloured by grounded', () => {
    const markers: InsightMarker[] = [
      { id: 'a', timeSec: 25, pct: 25, grounded: true, weight: 0.9 },
      { id: 'b', timeSec: 50, pct: 50, grounded: false, weight: 0.5 },
    ]
    const strip = mountPC({ markers }).find('[data-testid="player-insight-density"]')
    expect(strip.exists()).toBe(true)
    const ticks = strip.findAll('[data-testid="player-density-tick"]')
    expect(ticks).toHaveLength(2)
    expect(ticks[0].attributes('style')).toContain('left: 25%')
    expect(ticks[0].attributes('style')).toContain('opacity: 0.9')
    expect(ticks[0].classes()).toContain('bg-canvas-foreground') // grounded — data viz, not a control (#2013)
    expect(ticks[1].classes()).toContain('bg-muted') // ungrounded
  })

  it('shades a density heat-band: the busiest bin is darker than an empty one', () => {
    // Three markers clustered near 25% → that bin peaks; a far bin stays faint.
    const markers: InsightMarker[] = [
      { id: 'a', timeSec: 24, pct: 24, grounded: true, weight: 1 },
      { id: 'b', timeSec: 25, pct: 25, grounded: true, weight: 1 },
      { id: 'c', timeSec: 26, pct: 26, grounded: true, weight: 1 },
    ]
    const bands = mountPC({ markers }).findAll('[data-testid="player-density-band"]')
    expect(bands).toHaveLength(40)
    const opacityOf = (i: number) =>
      Number(/opacity:\s*([\d.]+)/.exec(bands[i].attributes('style') ?? '')?.[1] ?? '0')
    // Bin 10 covers 25–27.5% (the cluster) → peak intensity; bin 30 (75–77.5%) is empty.
    expect(opacityOf(10)).toBeGreaterThan(opacityOf(30))
  })
})

describe('the transport row distributes instead of reserving (#2004 item 9)', () => {
  // Comments stripped: the doc-comment explaining this fix quotes `px-14` and the old absolute
  // classes as prose, and would otherwise fail the rule it documents. Third time this has bitten in
  // this issue — the guards read source, and source includes the explanation of the guard.
  const code = playerControlsSource.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\*[\s\S]*?\*\//g, '')

  it('uses flex groups, not absolute clusters with a fixed width reservation', () => {
    // `px-14` reserved 56px per side for two ABSOLUTELY positioned clusters. The reservation was
    // symmetric; the content was not — the right cluster holds two 44px controls plus a gap (~96px),
    // so it overhung by ~40px onto the forward-30 button. That is the "squeezed" report: arithmetic,
    // not styling.
    expect(code).not.toContain('px-14')
    expect(code).not.toMatch(/absolute[^"]*left-0/)
    expect(code).not.toMatch(/absolute[^"]*right-0[^"]*translate-y/)
    expect(code).toContain('justify-between')
  })

  it('keeps every secondary control at the 44px touch target', () => {
    // The ask was "slightly smaller". h-11 is exactly 44px — the iOS minimum — so the crowding is
    // fixed by layout instead, and the speed control loses its extra pill width rather than the row
    // losing tappability.
    expect(code).not.toContain('min-w-11')
    const circles = code.match(/h-11 w-11/g) ?? []
    expect(circles.length).toBeGreaterThanOrEqual(3)
  })
})
