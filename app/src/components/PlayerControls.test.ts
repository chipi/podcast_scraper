import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import type { InsightMarker } from '../player/insightMarkers'
import PlayerControls from './PlayerControls.vue'

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
    expect(ticks[0].classes()).toContain('bg-accent') // grounded
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
