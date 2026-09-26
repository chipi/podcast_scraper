import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * A link or button must carry a NON-HIDDEN accessible name (2026-09-24).
 *
 * `aria-label` is not sufficient on its own. WebKit drops an interactive element from the
 * accessibility tree when its entire rendered subtree is `aria-hidden`, even though the element
 * itself is labelled — so the control becomes unreachable to VoiceOver AND to XCUITest while
 * remaining perfectly visible on screen.
 *
 * That is exactly what happened to the masthead profile link: its only child was `ProfileAvatar`,
 * whose root is `aria-hidden="true"` (correct — it is decorative). The link vanished from the
 * device's accessibility tree. It took an advisor pass and several wrong diagnoses to find, because
 * every screenshot showed the control present and every DOM query found it.
 *
 * The sibling `NavIconLink` was always fine: its tooltip span carries real text.
 *
 * This guard is deliberately NARROW — it checks the known-dangerous shape (an interactive element
 * whose sole child is a decorative avatar) rather than trying to compute accessible names from
 * source, which a regex cannot do honestly.
 */
const APP_VUE = readFileSync(join(__dirname, '../App.vue'), 'utf8')

/**
 * Components whose icon-only buttons must carry a real text node, not just `aria-label`.
 *
 * Android System WebView drops an `aria-label` when the button's subtree has no text, so the
 * control is announced as an unnamed "Button" and is unfindable by name. This has now been found
 * THREE times in this codebase — `SavedColorControl`'s trigger (2026-09-24), its swatches, and
 * `SavedFilterBar`'s filter swatches (both 2026-09-26) — each time only because a device test
 * tripped over it, and the third one arrived disguised as "the seed coloured too few items".
 *
 * A regex cannot compute accessible names honestly, so this does the narrow, checkable thing:
 * for the components listed here, every `<button>` that carries an `aria-label` must also contain
 * an `sr-only` span. Add a component when a device run finds the shape again.
 */
const SR_ONLY_REQUIRED = ['SavedColorControl.vue', 'SavedFilterBar.vue']

describe('icon-only buttons carry text, not only an aria-label', () => {
  for (const file of SR_ONLY_REQUIRED) {
    it(`${file}: every aria-labelled button has an sr-only name`, () => {
      const src = readFileSync(join(__dirname, '../components', file), 'utf8')
      const buttons = src.split('<button').slice(1)
      const labelled = buttons.filter((b) => b.includes('aria-label'))
      expect(labelled.length, `no aria-labelled buttons found in ${file} — did it move?`).toBeGreaterThan(0)
      for (const b of labelled) {
        const body = b.slice(0, b.indexOf('</button>'))
        expect(
          body.includes('sr-only'),
          `a button in ${file} has an aria-label but no sr-only text. On Android System WebView ` +
            `the label is dropped and the control is announced as an unnamed "Button" — and ` +
            `unfindable by name, which is how a colour filter looked like a broken seed.`,
        ).toBe(true)
      }
    })
  }
})

describe('interactive elements keep a reachable accessible name', () => {
  it('the masthead profile link carries text, not just an aria-label', () => {
    // The avatar is aria-hidden by design, so SOMETHING else inside the anchor has to be readable.
    const link = APP_VUE.slice(
      APP_VUE.indexOf('data-testid="header-profile"'),
      APP_VUE.indexOf('</RouterLink>', APP_VUE.indexOf('data-testid="header-profile"')),
    )
    expect(link, 'the profile link markup was not found — did the testid change?').toBeTruthy()
    expect(
      link.includes('sr-only'),
      'the profile link has no non-hidden accessible name. `aria-label` alone is NOT enough: ' +
        'ProfileAvatar is aria-hidden, so with no other content WebKit drops the whole link from ' +
        'the accessibility tree and it becomes unreachable to VoiceOver and to XCUITest.',
    ).toBe(true)
  })

  it('ProfileAvatar stays decorative — the fix is additive, not a swap', () => {
    // If someone "fixes" this by un-hiding the avatar, every list that renders one starts
    // announcing initials next to the name it already reads out.
    const avatar = readFileSync(join(__dirname, '../components/ProfileAvatar.vue'), 'utf8')
    expect(avatar.includes('aria-hidden="true"')).toBe(true)
  })

  it("the masthead's sr-only name is ANCHORED, so the link's a11y frame stays on screen", () => {
    // A name that is reachable but in the WRONG PLACE is its own bug, and it cost most of a day
    // on 2026-09-25.
    //
    // `sr-only` is `position:absolute` with no offsets, so the span takes its STATIC position —
    // after the 32px avatar, at the link's right edge. WebKit then derives the LINK's accessibility
    // frame from the text RUN, which lays out at its natural width even though the box is clipped
    // to 1x1. The masthead lives at the top-right, so the frame ran off the display: measured
    // x=381 w=48 on a 402pt screen, centre at 405, three points past the edge.
    //
    // XCUITest taps an element's centre. So the control was findable and untappable: every
    // `Journey.openProfile` reported success and navigated nowhere, and all four
    // `NativeOnlySurfacesTests` failed in `startClean` claiming "neither Sign in nor Sign out
    // present" about an app that was signed in on Home. VoiceOver inherits the same wrong
    // rectangle.
    //
    // `left-0 top-0` pins the run to the link's origin (hence `relative` on the anchor).
    const link = APP_VUE.slice(
      APP_VUE.indexOf('data-testid="header-profile"'),
      APP_VUE.indexOf('</RouterLink>', APP_VUE.indexOf('data-testid="header-profile"')),
    )
    expect(
      /class="sr-only[^"]*\bleft-0\b[^"]*"/.test(link) &&
        /class="sr-only[^"]*\btop-0\b[^"]*"/.test(link),
      'the masthead sr-only name is not anchored (`left-0 top-0`). Unanchored, it positions itself ' +
        'past the avatar at the screen edge and the link becomes untappable by centre-tap — ' +
        'findable, and dead. See the comment in App.vue.',
    ).toBe(true)

    const anchor = APP_VUE.slice(
      APP_VUE.indexOf('<RouterLink', APP_VUE.lastIndexOf('<RouterLink', APP_VUE.indexOf('data-testid="header-profile"'))),
      APP_VUE.indexOf('data-testid="header-profile"'),
    )
    expect(
      /\brelative\b/.test(anchor),
      'the profile anchor lost `relative`, so `left-0 top-0` on the sr-only span resolves against ' +
        'some ancestor instead of the link — which puts the a11y frame somewhere else entirely.',
    ).toBe(true)
  })

  it('NavIconLink names itself, and its tooltip stays out of the a11y tree', () => {
    // The SAME defect as the masthead profile link, in the shared masthead component — found the
    // hard way on 2026-09-25, after the profile one was already fixed.
    //
    // The icon slot and the badge are both `aria-hidden`, so the tooltip was the only text WebKit
    // could see, and it derives the LINK's accessibility frame from that run. The tooltip is
    // `absolute top-full` — BELOW the control — and `pointer-events-none`. So Queue and Search
    // reported a frame that was neither on the icon nor tappable: measured
    // `Link, {{265.0, 107.0}, {39.0, 16.0}}, label: 'Queue (3)'` against a 36x36 icon above it.
    // XCUITest reported present-but-not-hittable; VoiceOver draws the same wrong rectangle.
    const nav = readFileSync(join(__dirname, '../components/NavIconLink.vue'), 'utf8')
    expect(
      /class="sr-only[^"]*\bleft-0\b[^"]*\btop-0\b[^"]*"/.test(nav),
      'NavIconLink has no anchored sr-only name. Without it the link is named by its TOOLTIP, ' +
        'whose frame sits below the control and is pointer-events-none — findable, not tappable.',
    ).toBe(true)
    expect(
      /role="tooltip"[\s\S]{0,80}aria-hidden="true"|aria-hidden="true"[\s\S]{0,80}role="tooltip"/.test(nav),
      'the NavIconLink tooltip is exposed to assistive tech again. It repeats the aria-label ' +
        'verbatim and, left visible to the a11y tree, it becomes the element whose frame WebKit ' +
        'reports for the whole link.',
    ).toBe(true)
  })
})
