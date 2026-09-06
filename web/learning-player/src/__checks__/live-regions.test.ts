import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The Player owns exactly ONE live region, and Zone D is not it (#1978 follow-up).
 *
 * ## Why this is a test
 *
 * Zone D replaces its contents every time the audio reaches a new insight — every few tens of
 * seconds, unprompted. Making it `aria-live` is the obvious "accessibility improvement" and it is
 * the wrong one: a screen-reader user is LISTENING to the episode, and the panel restates a claim
 * drawn from the words they are hearing right now. Announcing it means talking over the podcast to
 * paraphrase the podcast, on a loop, with no way to decline.
 *
 * That reasoning is invisible to whoever next runs an a11y audit and sees a region that updates
 * without announcing. Left as a comment it gets "fixed"; as a test, reversing it is a deliberate
 * act with a place to argue.
 *
 * The second assertion guards the policy the app already had, which was previously only written in
 * a `defineEmits` docstring in KnowledgePanel.vue: PlayerView owns one live region and other
 * components announce THROUGH it, because two live regions on one page compete.
 */
/**
 * HTML comments are stripped before any check.
 *
 * The comment recording this very decision names `aria-live` in prose, and the first version of
 * this file matched on it — a guard failing on the documentation of the rule it enforces. The same
 * strip also means a commented-out `aria-live` correctly counts as absent, which is what the
 * browser does.
 */
const PLAYER = readFileSync(resolve(__dirname, '..', 'views', 'PlayerView.vue'), 'utf8').replace(
  /<!--[\s\S]*?-->/g,
  '',
)

describe('live regions in the player', () => {
  it('finds the file it is meant to guard', () => {
    // Without this, a rename turns every assertion below into a vacuous pass.
    expect(PLAYER.length).toBeGreaterThan(1000)
    expect(PLAYER).toContain('player-zone-d-live')
  })

  it('declares exactly one live region', () => {
    const found = PLAYER.match(/aria-live=/g) ?? []
    expect(
      found.length,
      'PlayerView owns ONE live region (the capture announcer). Components that need to announce ' +
        'emit through it — two live regions on one page compete for the same speech queue.',
    ).toBe(1)
  })

  it('does not announce Zone D as it changes with the audio', () => {
    // Scoped to the live-insight panel: find its opening tag and assert no aria-live inside the
    // element that carries the changing insight text.
    const i = PLAYER.indexOf('data-testid="player-zone-d-live"')
    expect(i, 'the Zone D live panel must be findable for this check to mean anything').toBeGreaterThan(0)
    const zone = PLAYER.slice(i, PLAYER.indexOf('player-zone-d-rest'))
    expect(
      zone,
      'Zone D restates what the listener is already hearing. Announcing it talks over the episode ' +
        'to paraphrase the episode. See the comment above the panel before changing this.',
    ).not.toMatch(/aria-live/)
  })
})
