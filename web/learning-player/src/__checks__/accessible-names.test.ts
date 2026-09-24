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
})
