import { describe, expect, it } from 'vitest'

/**
 * ONE share card, drawn by the SERVER (operator 2026-10-05: "I don't want entityShareCard.ts; I
 * want server/og/card.py as the base for all").
 *
 * The app used to draw its own card on a canvas, a plainer twin of the server's that the design
 * spec said must be kept in step "by eye". It was not: the server card got artwork, topics and per-
 * kind layouts, and the button kept sharing a black page with a title on it. These guards keep a
 * second card from growing back.
 */
const sources = import.meta.glob(['../**/*.ts', '../**/*.vue', '!../**/*.test.ts'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/**
 * Canvas drawing that is NOT an entity card, each with the reason it stays. Anything else that
 * draws on a canvas is a candidate second card renderer and fails here until it is named.
 */
const CANVAS_ALLOWED: Record<string, string> = {
  '../components/AvatarCropModal.vue': 'crops a profile photo — not a card',
  '../theme/accent.ts': 'samples a colour — not a card',
  '../composables/useShareCard.ts':
    'the HIGHLIGHT quote card — no server twin yet; moving it is an open operator decision',
}

describe('share card — one design, the server\'s', () => {
  it('reads the app (a guard over nothing passes vacuously)', () => {
    expect(Object.keys(sources).length).toBeGreaterThan(100)
    expect(sources['../components/ShareMenu.vue']).toBeTruthy()
  })

  it('nothing outside the allow-list draws on a canvas', () => {
    const drawing = Object.entries(sources)
      .filter(([, src]) => /getContext\(\s*['"]2d['"]\s*\)/.test(src))
      .map(([path]) => path)
      .filter((path) => !(path in CANVAS_ALLOWED))
    expect(drawing, 'a new canvas renderer — is this a second share card?').toEqual([])
  })

  it('the in-app card renderer stays deleted', () => {
    const importers = Object.entries(sources)
      .filter(([, src]) => /entityShareCard/.test(src))
      .map(([path]) => path)
    expect(importers).toEqual([])
  })

  it('the Share menu shares the server card', () => {
    const menu = sources['../components/ShareMenu.vue']
    expect(menu).toMatch(/from "\.\.\/composables\/shareCard"/)
    expect(sources['../composables/shareCard.ts']).toMatch(/fetchShareCard/)
    expect(sources['../services/api.ts']).toMatch(/`\/og\/\$\{kind\}\//)
  })
})
