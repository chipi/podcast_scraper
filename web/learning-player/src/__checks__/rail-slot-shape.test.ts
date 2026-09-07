import { describe, expect, it } from 'vitest'

/**
 * A horizontal rail gets TILES, never row cards (#2004 follow-up).
 *
 * ## The bug
 *
 * "More like this" put `EpisodeCard` — a horizontal row, artwork column beside a text column — into
 * a 224px rail slot. The text column got about 100px, so a real title wrapped to eight lines, the
 * slot grew to roughly 800px tall, and the action row (positioned against the card's top-right)
 * floated over the artwork.
 *
 * Nothing errored and no test failed. It was only ever visible by looking at it, which is why this
 * check exists rather than a note in a doc.
 *
 * ## Why a source check
 *
 * The failure is structural — which component sits inside which container — and that is legible in
 * the markup. Catching it at runtime would need a rendered rail with real data at a real width; the
 * one rail this bit is fed by semantic similarity and is empty without the search index, so it
 * cannot be relied on to reproduce anywhere cheap.
 *
 * Deliberately narrow: it asserts a known-bad PAIRING, not a general layout theory. A rule broad
 * enough to police "is this component the right shape for this box" would be guesswork.
 */

const files = import.meta.glob('../**/*.vue', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/** Row cards: artwork column beside a text column. Correct in a vertical list, wrong in a slot. */
const ROW_CARDS = ['EpisodeCard']

/** Strip comments — this file's own explanation names the offending pairing. */
function code(src: string): string {
  return src.replace(/<!--[\s\S]*?-->/g, '').replace(/\/\*[\s\S]*?\*\//g, '')
}

describe('rail slots hold tiles, not row cards', () => {
  const entries = Object.entries(files).map(([p, src]) => [p, code(src)] as const)

  it('no CardRail contains a row card', () => {
    const offenders: string[] = []
    for (const [path, src] of entries) {
      for (const m of src.matchAll(/<CardRail[\s\S]*?<\/CardRail>/g)) {
        for (const card of ROW_CARDS) {
          if (m[0].includes(`<${card}`)) offenders.push(`${path}: <${card}> inside <CardRail>`)
        }
      }
    }
    expect(
      offenders,
      'A rail slot is narrow; a row card puts its text in a column beside the artwork and the ' +
        'title collapses to a few words per line. Use EpisodeTile (or another stacked tile).',
    ).toEqual([])
  })

  it('the sweep actually reads the rails — it is not passing on an empty match', () => {
    // `offenders === []` is satisfied perfectly by a glob that found nothing, or by a regex that
    // stopped matching the markup after a refactor.
    const rails = entries.filter(([, src]) => /<CardRail[\s\S]*?<\/CardRail>/.test(src))
    expect(rails.length, 'found no CardRail blocks to check').toBeGreaterThan(0)
    expect(Object.keys(files).length, 'the component glob found nothing').toBeGreaterThan(40)
  })

  it('the rail that broke is now a tile', () => {
    // Pins the specific regression rather than only the general rule.
    const player = entries.find(([p]) => p.endsWith('views/PlayerView.vue'))
    expect(player, 'PlayerView not found').toBeTruthy()
    const rail = player![1].match(/<CardRail[\s\S]*?<\/CardRail>/)?.[0] ?? ''
    expect(rail, 'the related-episodes rail lost its tile').toContain('<EpisodeTile')
  })
})
