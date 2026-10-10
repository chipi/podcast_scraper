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
  it('orders the timeline scrubber → density strip → timestamps (operator 2026-09-30)', () => {
    // The two timeline strips sit together and the time readout labels their ends at the bottom.
    // It was scrubber / times / density, which read as "line, two numbers, line again".
    const markers: InsightMarker[] = [{ id: 'a', timeSec: 25, pct: 25, grounded: true, weight: 1 }]
    const root = mountPC({ markers }).element as HTMLElement
    const order = [
      root.querySelector('input[type="range"]'),
      root.querySelector('[data-testid="player-insight-density"]'),
      root.querySelector('[data-testid="player-times"]'),
    ]
    expect(order.every(Boolean)).toBe(true)
    for (let i = 1; i < order.length; i++) {
      // DOCUMENT_POSITION_FOLLOWING = 4: each comes after the previous one.
      expect(order[i - 1]!.compareDocumentPosition(order[i]!) & 4).toBe(4)
    }
  })

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

describe('the transport row is a mirror (#2004 item 9; operator 2026-09-30)', () => {
  // Comments stripped: doc-comments here quote the old classes as prose, and the guards read source.
  const code = playerControlsSource.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\*[\s\S]*?\*\//g, '')

  it('uses a grid, not absolute clusters with a fixed width reservation', () => {
    // `px-14` once reserved 56px per side for two ABSOLUTELY positioned clusters that did not fit it.
    expect(code).not.toContain('px-14')
    expect(code).not.toMatch(/absolute[^"]*left-0/)
    expect(code).not.toMatch(/absolute[^"]*right-0[^"]*translate-y/)
  })

  it('lays out seven cells around a centred play button', () => {
    const row = mountPC().get('[data-testid="player-transport"]')
    expect(row.classes().join(' ')).toContain('grid-cols-[repeat(3,minmax(0,1fr))_auto_repeat(3,minmax(0,1fr))]')
    expect(row.element.children).toHaveLength(7)
  })

  it('gives every secondary control the one shared size, with a 44px hit area', () => {
    // 40px of ink on phones is only acceptable because `lp-tap` keeps the HIT area at 44px.
    const w = mountPC()
    for (const label of ['Skip back 15 seconds', 'Skip forward 30 seconds', 'Playback speed']) {
      const b = w.get(`button[aria-label="${label}"]`)
      expect(b.classes(), label).toEqual(expect.arrayContaining(['lp-tap', 'h-10', 'w-10', 'sm:h-11', 'sm:w-11']))
    }
    // Slot content gets the same size from the row, so a caller cannot drift from it.
    expect(code).toMatch(/<slot name="left-outer" :size="TRANSPORT_BUTTON_SIZE"/)
    expect(code).toMatch(/<slot name="left-inner" :size="TRANSPORT_BUTTON_SIZE"/)
    expect(code).toMatch(/<slot name="right-inner" :size="TRANSPORT_BUTTON_SIZE"/)
  })

  it('the geometry is verified where it can actually be measured', () => {
    // Source text cannot know whether the row FITS or MIRRORS. `e2e/design-invariants.spec.ts`
    // measures both against a real engine: fit, 44px hit areas, pitch, and equal twin distances.
    expect(code).toContain('TRANSPORT_BUTTON_SIZE')
  })
})

describe('the density strip seeks, like the scrubber (operator 2026-09-30)', () => {
  const markers: InsightMarker[] = [{ id: 'a', timeSec: 300, pct: 50, weight: 1, grounded: true }]

  function stripOf(w: ReturnType<typeof mountPC>) {
    const el = w.get('[data-testid="player-density-seek"]')
    // jsdom has no layout: give the strip a 200px box starting at x=100.
    ;(el.element as HTMLElement).getBoundingClientRect = () =>
      ({ left: 100, width: 200, top: 0, height: 26, right: 300, bottom: 26, x: 100, y: 0, toJSON: () => ({}) }) as DOMRect
    return el
  }

  it('a tap jumps to that point of the episode', async () => {
    const w = mountPC({ markers, duration: 600 })
    await stripOf(w).trigger('pointerdown', { clientX: 150, pointerId: 1 })
    // 50px into a 200px strip = 25% of 600s.
    expect(w.emitted('seek')?.at(-1)).toEqual([150])
  })

  it('press and drag scrubs; moving without a press does nothing', async () => {
    const w = mountPC({ markers, duration: 600 })
    const strip = stripOf(w)
    await strip.trigger('pointermove', { clientX: 250, pointerId: 1 })
    expect(w.emitted('seek')).toBeUndefined()
    await strip.trigger('pointerdown', { clientX: 100, pointerId: 1 })
    await strip.trigger('pointermove', { clientX: 300, pointerId: 1 })
    expect(w.emitted('seek')?.at(-1)).toEqual([600])
    await strip.trigger('pointerup', { pointerId: 1 })
    await strip.trigger('pointermove', { clientX: 200, pointerId: 1 })
    expect(w.emitted('seek')?.at(-1)).toEqual([600])
  })

  it('clamps a press outside the strip to the start', async () => {
    const w = mountPC({ markers, duration: 600 })
    await stripOf(w).trigger('pointerdown', { clientX: 20, pointerId: 1 })
    expect(w.emitted('seek')?.at(-1)).toEqual([0])
  })

  it('stays an image to assistive tech — the range input is the accessible control', () => {
    const w = mountPC({ markers, duration: 600 })
    expect(w.get('[data-testid="player-insight-density"]').attributes('role')).toBe('img')
    expect(w.get('[data-testid="player-density-seek"]').attributes('tabindex')).toBeUndefined()
  })
})

describe('PlayerControls — Step and Moments on the density strip (2026-10-10)', () => {
  const markers: InsightMarker[] = [
    { id: 'a', timeSec: 10, pct: 10, grounded: true, weight: 0.5 },
    { id: 'b', timeSec: 50, pct: 50, grounded: true, weight: 0.5 },
  ]
  it('the insight the listener is in stands out', () => {
    const w = mountPC({ markers, currentMarkerId: 'b' })
    const ticks = w.findAll('[data-testid="player-density-tick"]')
    expect(ticks.map((t) => t.attributes('data-current'))).toEqual([undefined, 'true'])
    expect(ticks[1].attributes('style')).toContain('opacity: 1')
  })
  it('moments are marked, never in the accent', () => {
    const w = mountPC({ markers, momentMarks: [10, 50] })
    const marks = w.findAll('[data-testid="player-moment-mark"]')
    expect(marks).toHaveLength(2)
    for (const m of marks) expect(m.classes().join(' ')).not.toContain('accent')
  })
})
