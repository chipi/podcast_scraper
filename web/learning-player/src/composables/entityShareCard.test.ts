import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  type EntityCardModel,
  accentForKind,
  entityCardText,
  renderEntityCard,
  shareEntityLink,
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

  it('shareEntityLink: Web Share when available', async () => {
    const share = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { share } as unknown as Navigator)
    expect(await shareEntityLink(TOPIC)).toBe('shared')
    expect(share).toHaveBeenCalledWith(expect.objectContaining({ url: TOPIC.url }))
  })

  it('shareEntityLink: clipboard copy when Web Share absent', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } } as unknown as Navigator)
    expect(await shareEntityLink(TOPIC)).toBe('copied')
    expect(writeText).toHaveBeenCalledWith(TOPIC.url)
  })

  it('shareEntityLink: none when there is no url', async () => {
    expect(await shareEntityLink({ kicker: 'Topic', title: 'X' })).toBe('none')
  })

  it('accentForKind: topic + person own a token colour; other kinds fall back to brand cyan', () => {
    expect(accentForKind('topic')).toBe('#8ad2e5')
    expect(accentForKind('person')).toBe('#e0b354')
    // Kinds with no theme token (show / storyline / organization) + the empty case → brand cyan.
    expect(accentForKind('organization')).toBe('#8ad2e5')
    expect(accentForKind('show')).toBe('#8ad2e5')
    expect(accentForKind('storyline')).toBe('#8ad2e5')
    expect(accentForKind(null)).toBe('#8ad2e5')
  })
})
