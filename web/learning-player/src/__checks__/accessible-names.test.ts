import { readFileSync, readdirSync } from 'node:fs'
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
 * A control with a ROLE-CHANGING aria attribute and nothing readable inside is unnamed on Android.
 *
 * Repo-wide, no list. This replaced a hand-maintained array of six filenames (2026-09-27), which
 * guarded whatever someone had remembered to add — which is why `views/` was invisible to it until
 * five controls had already shipped unnamed.
 *
 * ## The rule, and how it was settled on a device
 *
 * Two measurements used to contradict each other and the guard was drawn at their INTERSECTION,
 * with the contradiction written down as unresolved:
 *
 *   - `OverflowMenu.vue` (2026-09-24, Android System WebView 150.0.7871.181): the ⋯ trigger arrived
 *     as a zero-child `Button` with an empty contentDescription. The same page showed `Play`,
 *     `Skip back 15 seconds`, `Mark this moment` and `Playback speed` — all icon-only with an
 *     `aria-label` — all NAMED. It concluded the cause was `aria-haspopup` PLUS a hidden subtree,
 *     and that "neither alone does it".
 *   - `SavedColorControl.vue`: its five colour swatches measured `<UNLABELLED>[ToggleButton]` on
 *     2026-09-26 — with NO `aria-haspopup`.
 *
 * Settled 2026-09-28 by A/B on the device rather than by argument. With the swatches' `sr-only`
 * spans removed (full build → cap sync → assembleDebug → install), `AppJourneyTests
 * #test07SavedColourPicker` failed with exactly five `<UNLABELLED>[ToggleButton]` in the inventory;
 * restoring the spans made it pass again. Pass → fail → pass, one variable.
 *
 * So `OverflowMenu`'s "neither alone does it" is WRONG as a general rule, and the resolved shape
 * fits every measurement taken so far:
 *
 *      aria-haspopup + no text node  ->  UNNAMED   (the ⋯ trigger)
 *      aria-pressed  + no text node  ->  UNNAMED   (the colour swatches)
 *      plain button  + no text node  ->  named     (Play, Skip back 15 seconds)
 *
 * Both attributes change the node's ROLE — PopUpButton and ToggleButton respectively — and it is in
 * that remapping that the computed name is lost. A plain Button keeps it.
 *
 * ## Why still not "every icon-only button"
 *
 * Because the measurements say plain buttons are fine, and the wider rule is not free: an
 * `sr-only` span is `position:absolute`, and an unanchored one is what dragged the masthead link's
 * accessibility frame off the display (see below). Demanding one on already-named controls would
 * spread that hazard to fix a defect they do not have. The rule covers exactly what is measured —
 * which is now 32 controls rather than 6.
 *
 * (#2156)
 */
function vueFilesUnder(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...vueFilesUnder(full))
    else if (entry.name.endsWith('.vue')) out.push(full)
  }
  return out
}

/** `<button …>…</button>` bodies paired with their attribute string, template only. */
function buttonsIn(src: string): { attrs: string; body: string }[] {
  const withoutBlocks = src.replace(/<(script|style)\b[\s\S]*?<\/\1>/g, '')
  const out: { attrs: string; body: string }[] = []
  for (const m of withoutBlocks.matchAll(/<button\b/g)) {
    const start = m.index
    const end = withoutBlocks.indexOf('</button>', start)
    if (start === undefined || end === -1) continue
    const whole = withoutBlocks.slice(start, end)
    const gt = whole.indexOf('>')
    if (gt === -1) continue
    out.push({ attrs: whole.slice(0, gt), body: whole.slice(gt + 1) })
  }
  return out
}

describe('role-changed controls carry a name Android can read', () => {
  it('every aria-haspopup or aria-pressed button has readable text or an sr-only name', () => {
    const roots = [join(__dirname, '../components'), join(__dirname, '../views')]
    const offenders: string[] = []
    let checked = 0

    for (const root of roots) {
      for (const file of vueFilesUnder(root)) {
        for (const { attrs, body } of buttonsIn(readFileSync(file, 'utf8'))) {
          // BOTH attributes, since 2026-09-28. `aria-pressed` was added after a device A/B showed
          // the colour swatches going `<UNLABELLED>[ToggleButton]` with the span removed and named
          // with it restored — they carry no `aria-haspopup`, so the old rule did not cover them.
          if (!attrs.includes('aria-haspopup') && !attrs.includes('aria-pressed')) continue
          checked += 1
          // VISIBLE TEXT COUNTS. The requirement is something readable in the subtree, not the
          // `sr-only` class specifically — ToolbarMenu's pill has a label beside its icon and is
          // named because of it. Demanding a hidden span there would cargo-cult the fix rather
          // than the reason.
          const readable = body.includes('sr-only') || body.includes('{{')
          if (!readable) offenders.push(`${file.slice(file.indexOf('/src/') + 5)}`)
        }
      }
    }

    // If this collapses, the scan has broken rather than the app. 32 such controls existed when the
    // rule was widened; the floor is deliberately well below that so ordinary additions and
    // removals do not trip it, while a scan that suddenly matches almost nothing does.
    expect(
      checked,
      'almost no aria-haspopup/aria-pressed buttons found — the scan or the markup moved',
    ).toBeGreaterThanOrEqual(20)
    expect(
      offenders,
      `these controls change ROLE via aria-haspopup or aria-pressed and have nothing readable ` +
        `inside them. Measured on Android System WebView 150: such a node arrives as a ` +
        `PopUpButton/ToggleButton with an empty name — the label appears nowhere in the ` +
        `hierarchy, TalkBack announces only the role, and no device test can address it by ` +
        `intent. A PLAIN button in the same shape keeps its name; it is the role remapping that ` +
        `loses it. Verified by A/B on device: removing SavedColorControl's swatch spans produced ` +
        `five <UNLABELLED>[ToggleButton] and restoring them fixed it.`,
    ).toEqual([])
  })
})

/**
 * The controls a DEVICE has actually caught, pinned so their names cannot be removed.
 *
 * This list came back (2026-09-28) after being deleted with the old guard, and the deletion was a
 * mistake worth explaining, because the reasoning that produced it was half right.
 *
 * The old guard used a hand-kept list to do TWO jobs, and it was only bad at one of them. As a
 * DISCOVERY mechanism it was useless — it found what someone remembered to add, which is why
 * `views/` was invisible to it until five controls had already shipped unnamed. The repo-wide
 * `aria-haspopup` rule above replaces that job properly. But the list was also a REGRESSION PIN on
 * instances a device had measured, and that job it was doing correctly. Deleting both together
 * left the five controls below guarded by nothing at all.
 *
 * Why these specifically, and why the haspopup rule does not reach them: NONE of them carries
 * `aria-haspopup`, so under that rule alone their `sr-only` spans can be deleted and every check in
 * this repo stays green. `SavedColorControl`'s five swatches are the sharp case — they were
 * MEASURED on device as `<UNLABELLED>[ToggleButton]` (see the comment at SavedColorControl.vue) and
 * they have no `aria-haspopup`, which directly contradicts the OverflowMenu measurement the rule
 * above is drawn from. Two device measurements disagree and the distinguishing variable is
 * unidentified.
 *
 * When measurements contradict, a guard covers the UNION of the measured failure shapes, not the
 * intersection — the intersection is only safe if you know which variable separates them, and the
 * docstring above says plainly that nobody does. Drawing it at the intersection was me resolving a
 * contradiction in the direction that required less work.
 *
 * This costs nothing: every control here ALREADY has its name. The pin only forbids removing one.
 * It should be deleted when the contradiction is resolved on a device and the general rule can be
 * widened to cover these for a stated reason. (#2156)
 */
const MEASURED_ON_DEVICE = [
  'components/SavedColorControl.vue',
  'components/SavedFilterBar.vue',
  'components/NoteComposer.vue',
  'components/TranscriptList.vue',
  'views/CollectionsView.vue',
  'views/ResurfacingInbox.vue',
]

describe('controls a device has caught keep their names', () => {
  for (const file of MEASURED_ON_DEVICE) {
    it(`${file}: every aria-labelled button still has a readable name`, () => {
      const src = readFileSync(join(__dirname, '..', file), 'utf8')
      const labelled = buttonsIn(src).filter((b) => b.attrs.includes('aria-label'))
      expect(
        labelled.length,
        `no aria-labelled buttons found in ${file} — did it move? A pin that matches nothing ` +
          `passes silently, which is the failure mode this whole file exists to catch.`,
      ).toBeGreaterThan(0)
      for (const { body } of labelled) {
        expect(
          body.includes('sr-only') || body.includes('{{'),
          `a button in ${file} has an aria-label and NO readable text. This file is on the pinned ` +
            `list because a DEVICE reported one of its controls unnamed — SavedColorControl's ` +
            `swatches came back <UNLABELLED>[ToggleButton] with no aria-haspopup, so the ` +
            `repo-wide rule above does not cover them. Removing the name here is a regression a ` +
            `device already paid for once.`,
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
