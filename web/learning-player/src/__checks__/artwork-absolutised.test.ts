import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'

/**
 * Episode and show ARTWORK must be absolutised before it reaches an `<img src>`.
 *
 * This is the sibling of `person-photo-absolutised.test.ts`, which guards the other half of the same
 * defect and says of this half: *"a different field family, covered elsewhere."* That was not true.
 * On 2026-10-03 the landing page painted broken-image placeholders on a real device, and the sweep
 * found FOUR surfaces that had each re-implemented the artwork fallback chain by hand and dropped
 * the absolutise: `LandingView`, `EpisodeTile`, `RevisitRail` and `CollectionsView` (twice).
 *
 * THE MECHANISM, which is why it keeps happening
 * ──────────────────────────────────────────────
 * The API returns artwork RELATIVE (`/api/app/...`). On the web that is correct — the app and the
 * API share an origin. Inside the Capacitor WebView the document origin is `capacitor://localhost`,
 * so the same string resolves against the APP BUNDLE, 404s, and the card paints a broken image.
 *
 * `fetch` is never affected, because `apiFetch` prefixes an absolute base itself. `<img src>` has
 * nothing doing that for it. So the bug is invisible to every browser test — on the web it does not
 * exist — and it only appears on a device, which is the one tier nothing runs automatically.
 *
 * WHY A TEXT RULE RATHER THAN A LIST OF SURFACES
 * ─────────────────────────────────────────────
 * The person-photo guard's own history is the argument: its first version checked a hand-kept list
 * of five fetchers, `/trending` started returning photos, nobody added it, and the guard that existed
 * for exactly that bug passed while it shipped a sixth time. A list of what to check goes stale.
 *
 * What cannot go stale is the SHAPE of the mistake: reading the artwork fallback chain by hand.
 * `utils/episode.ts` owns that chain — in one place, so it cannot drift — and it absolutises. Any
 * other file spelling the chain out is re-implementing it, and re-implementing it is how all four
 * instances lost the absolutise.
 */

// Resolved from the vitest cwd (web/learning-player), not from `import.meta.url` — that yields a
// bare '/src' here and the reads fail with ENOENT.
const SRC = resolve(process.cwd(), 'src')

/** The helpers that own the chain, and are therefore allowed to spell it out. */
const OWNERS = ['utils/episode.ts']

/** Any of these in an expression means the author went through a resolver. */
const RESOLVERS = ['resolveMediaUrl', 'episodeArtwork', 'showArtwork', 'localArtworkFor']

function sourceFiles(dir: string, acc: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) {
      if (name === 'node_modules' || name === '__checks__') continue
      sourceFiles(full, acc)
      continue
    }
    if (!/\.(ts|vue)$/.test(name) || /\.test\.ts$/.test(name)) continue
    acc.push(full)
  }
  return acc
}

describe('artwork is absolutised before it reaches an <img> (#2267, device-only defect)', () => {
  it('no surface re-implements the episode artwork fallback chain', () => {
    const offenders: string[] = []
    for (const file of sourceFiles(SRC)) {
      const rel = relative(SRC, file)
      if (OWNERS.some((o) => rel.endsWith(o))) continue
      const text = readFileSync(file, 'utf8')
      text.split('\n').forEach((line, i) => {
        // The chain is recognisable by two of its fields appearing in ONE expression — that is the
        // hand-rolled fallback, and it is exactly what the four broken surfaces looked like.
        const fields = ['artwork_url', 'episode_image_url', 'feed_image_url', 'image_url'].filter((f) =>
          line.includes(f),
        )
        if (fields.length < 2) return
        // Declarations and object construction are not display; only expressions that CHOOSE
        // between fields are.
        if (!/\|\||\?\?/.test(line)) return
        if (RESOLVERS.some((r) => line.includes(r))) return
        offenders.push(`${rel}:${i + 1}: ${line.trim().slice(0, 100)}`)
      })
    }
    expect(
      offenders,
      'These spell out the artwork fallback chain by hand. Use episodeArtwork() / showArtwork() from ' +
        'utils/episode.ts — they apply the same order AND absolutise, which a hand-rolled chain ' +
        'forgets. Relative artwork 404s against capacitor://localhost on device and is invisible on web.',
    ).toEqual([])
  })

  it('no template binds an <img src> straight to a raw artwork field', () => {
    const offenders: string[] = []
    for (const file of sourceFiles(SRC)) {
      const rel = relative(SRC, file)
      if (!rel.endsWith('.vue')) continue
      const text = readFileSync(file, 'utf8')
      // `:src="…artwork_url…"` with no resolver in the same binding.
      for (const m of text.matchAll(/:src="([^"]*)"/g)) {
        const expr = m[1]
        if (!/artwork_url|episode_image_url|feed_image_url/.test(expr)) continue
        if (RESOLVERS.some((r) => expr.includes(r))) continue
        const line = text.slice(0, m.index).split('\n').length
        offenders.push(`${rel}:${line}: :src="${expr.slice(0, 80)}"`)
      }
    }
    expect(
      offenders,
      'An <img> bound straight to a raw artwork field renders broken on device. Route it through ' +
        'episodeArtwork() / showArtwork().',
    ).toEqual([])
  })

  it('the owning helpers really do absolutise — the rule is worthless if they stop', () => {
    const owner = readFileSync(join(SRC, 'utils/episode.ts'), 'utf8')
    expect(owner).toContain('resolveMediaUrl')
    // Both exported helpers, not just one: `showArtwork` was the one CollectionsView needed.
    for (const fn of ['export function episodeArtwork', 'export function showArtwork']) {
      const at = owner.indexOf(fn)
      expect(at, `${fn} must exist`).toBeGreaterThan(-1)
      const body = owner.slice(at, owner.indexOf('\n}', at))
      expect(body, `${fn} must absolutise`).toContain('resolveMediaUrl')
    }
  })
})
