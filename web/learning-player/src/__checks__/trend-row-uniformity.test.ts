import { describe, expect, it } from 'vitest'

/**
 * A trend row is the SAME height whatever kind it renders (operator 2026-09-19).
 *
 * ## The bug
 *
 * `DiscoveryList` renders one row shape for three kinds. People rows carry a `ProfileAvatar` at
 * 28px; topics and storylines carry nothing taller than the 20px sparkline. With only `py-1.5`
 * around them that made People rows 40px and the other two 32px.
 *
 * Switching the kind tab therefore moved everything below the section by 8px per row — three rows
 * is 24px — and the page jumped under the reader mid-scroll. The list was correct, the data was
 * correct, and it still felt broken.
 *
 * ## Why a source check rather than a rendered one
 *
 * The failure is a computed height difference between two states of one component. Reproducing it
 * needs a real browser with real fonts: happy-dom reports 0px for everything, so a mounted test
 * would assert nothing while looking like it asserted something — the vacuous-guard shape this
 * repo keeps hitting.
 *
 * What IS legible in the markup: the row pins an explicit minimum height, so the tallest optional
 * child cannot set it. That is the invariant, and it is the thing a future edit would drop.
 *
 * Deliberately narrow — it pins ONE component's row, not a general theory of layout. The same
 * discipline as `rail-slot-shape`: assert the known-bad shape, not a guess about all shapes.
 */

const files = import.meta.glob('../**/*.vue', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>

function source(name: string): string {
  const hit = Object.entries(files).find(([p]) => p.endsWith(`/${name}`))
  if (!hit) throw new Error(`${name} not found — was it moved or renamed?`)
  return hit[1]
}

describe('discovery trend rows keep one height across kinds', () => {
  it('the row button pins a minimum height, so an avatar cannot make People taller', () => {
    const src = source('DiscoveryList.vue')

    // The conditional avatar is what creates the asymmetry. If it ever stops being conditional this
    // check is moot — assert the premise so the test cannot quietly outlive its reason.
    expect(
      src,
      'DiscoveryList no longer renders a kind-conditional avatar — re-check whether this guard still applies',
    ).toMatch(/<ProfileAvatar[\s\S]{0,120}v-if="kind === 'person'"/)

    // The row button is the one carrying `flex-1` — the growing child of the row `<li>`. Matching on
    // proximity to ProfileAvatar was brittle: inserting a comment between them broke it, which is a
    // test failing for a reason unrelated to what it guards.
    const rowButton = src.match(/<button[\s\S]{0,200}?class="([^"]*flex-1[^"]*)"/)
    expect(rowButton, 'could not find the trend row button by its `flex-1` class').not.toBeNull()
    expect(
      rowButton![1],
      'the trend row button has no `min-h-*`, so its height is set by whichever optional child is ' +
        'tallest. People rows carry a 28px avatar and topics/storylines do not, so the kind tabs ' +
        'render different heights and the page jumps when you switch between them.',
    ).toMatch(/\bmin-h-\d/)
  })

  it('the avatar is not taller than the row it sits in', () => {
    const src = source('DiscoveryList.vue')
    const size = src.match(/<ProfileAvatar[\s\S]{0,160}?:size="(\d+)"/)
    expect(size, 'ProfileAvatar in DiscoveryList has no explicit :size').not.toBeNull()

    const px = Number(size![1])
    const minH = Number(src.match(/min-h-(\d+)/)?.[1] ?? 0) * 4 // tailwind: min-h-10 = 2.5rem = 40px
    expect(
      px + 12, // py-1.5 top + bottom
      `avatar ${px}px plus padding exceeds the pinned row height ${minH}px, so People rows will ` +
        'still be taller than the others',
    ).toBeLessThanOrEqual(minH)
  })
})
