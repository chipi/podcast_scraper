import { describe, expect, it } from 'vitest'
import appSrc from '../App.vue?raw'
import transcriptSrc from '../components/TranscriptList.vue?raw'

/**
 * Guardrail (#1588, #1592) — two properties of the shell that are easy to regress silently and
 * expensive when they do.
 *
 * Static source checks rather than mounted assertions, because both are about CSS that only takes
 * effect under a media query jsdom does not evaluate. A mounted test would pass either way, which
 * is precisely how the transcript capture button stayed invisible on phones for so long.
 */

/** Every component that hides an affordance behind hover must also show it where hover is absent. */
const HOVER_HIDDEN = /opacity-0[^"]*group-hover:opacity-100/

describe('affordances survive on touch', () => {
  it('the transcript capture control is visible without hover (#1592)', () => {
    // Phones are the primary platform (the e2e suite's default project is a Pixel 7) and have no
    // hover. `opacity-0 + group-hover` alone leaves the control transparent but tappable —
    // undiscoverable rather than obviously missing, which is worse than absent. This is the entry
    // point to capture → highlights → notes → resurfacing, i.e. the whole learning loop.
    const buttons = transcriptSrc.split('<button').filter((b) => HOVER_HIDDEN.test(b))
    expect(buttons.length, 'expected a hover-quiet capture button to exist').toBeGreaterThan(0)
    for (const b of buttons) {
      expect(
        b,
        'A hover-hidden control must also carry [@media(hover:none)]:opacity-100, or it is ' +
          'invisible on the primary platform.',
      ).toContain('[@media(hover:none)]:opacity-100')
    }
  })

  it('search is reachable from the primary nav (#1588)', () => {
    // Corpus-wide semantic search with jump-to-moment is the differentiator neither Spotify nor
    // Apple Podcasts offers. It previously had one entry point — the Home search box — so from the
    // catalogue, player, library or a show page there was no way to reach it at all.
    const nav = appSrc.slice(appSrc.indexOf('<nav'), appSrc.indexOf('</nav>'))
    expect(nav).toContain("name: 'search'")
  })

  it('search stays public, like browse — reads are open', () => {
    // If the search link were inside the `auth.isAuthenticated` block, signed-out visitors would
    // lose the one capability most likely to convert them.
    const nav = appSrc.slice(appSrc.indexOf('<nav'), appSrc.indexOf('</nav>'))
    const gated = nav.slice(nav.indexOf('auth.isAuthenticated'))
    expect(gated).not.toContain("name: 'search'")
  })
})
