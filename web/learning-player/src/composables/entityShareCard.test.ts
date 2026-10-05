import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  type EntityCardModel,
  accentForKind,
  entityCardText,
  entityCopyText,
  renderEntityCard,
} from './entityShareCard'

vi.mock('../services/native', () => ({
  isNative: () => false,
  saveAndShareText: vi.fn(),
}))

const TOPIC: EntityCardModel = {
  kicker: 'Topic',
  title: 'Risk management',
  quote: 'Risk is a systems property.',
  byline: '— Dr. Elena Fischer',
  stats: '28 episodes · 10 voices',
  hot: '↑ 2.3× rising',
  accent: '#8ad2e5',
  url: 'https://closelistening.app/topic/topic:risk-management',
}

afterEach(() => vi.restoreAllMocks())

describe('entityShareCard (#2036)', () => {
  it('builds a caption with kicker, title, quote, byline, stats + wordmark', () => {
    const t = entityCardText(TOPIC)
    expect(t).toContain('TOPIC')
    expect(t).toContain('Risk management')
    expect(t).toContain('“Risk is a systems property.”')
    expect(t).toContain('— Dr. Elena Fischer')
    expect(t).toContain('28 episodes · 10 voices · ↑ 2.3× rising')
    expect(t.trim().endsWith('closelistening.app')).toBe(true)
  })

  it('omits absent parts (a bare card is just kicker + title + wordmark)', () => {
    const t = entityCardText({ kicker: 'Show', title: 'My Show' })
    expect(t).toBe('SHOW\nMy Show\ncloselistening.app')
  })

  it('renderEntityCard degrades to null when canvas is unavailable (jsdom), never throws', async () => {
    await expect(renderEntityCard(TOPIC)).resolves.toBeNull()
  })

  it('Copy text: the name, what it is, then the link (operator 2026-10-05)', () => {
    expect(
      entityCopyText({
        kicker: 'Person',
        title: 'Grady Booch',
        context: 'American software engineer',
        url: 'https://closelistening.app/person/person%3Agrady-booch',
      }),
    ).toBe('Grady Booch — American software engineer\nhttps://closelistening.app/person/person%3Agrady-booch')
  })

  it('Copy text: no context, no dash; no link, no trailing line', () => {
    expect(entityCopyText({ kicker: 'Topic', title: 'Risk', url: 'https://x/topic/risk' })).toBe('Risk\nhttps://x/topic/risk')
    expect(entityCopyText({ kicker: 'Organization', title: 'The Fed' })).toBe('The Fed')
  })

  it('accentForKind: each kind has a distinct colour; show/episode/unknown fall back to cyan', () => {
    expect(accentForKind('topic')).toBe('#8ad2e5')
    expect(accentForKind('person')).toBe('#e0b354')
    expect(accentForKind('storyline')).toBe('#9d8cff')
    expect(accentForKind('organization')).toBe('#5fd0a8')
    // show/episode carry artwork, so they keep the brand cyan; empty case too.
    expect(accentForKind('show')).toBe('#8ad2e5')
    expect(accentForKind(null)).toBe('#8ad2e5')
  })
})
