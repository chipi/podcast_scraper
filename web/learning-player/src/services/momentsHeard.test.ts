import { nextTick, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { MomentEvent } from '../stores/player'

const post = vi.fn(async (_body: Record<string, unknown>) => undefined)
vi.mock('./api', () => ({ postMomentHeard: (body: Record<string, unknown>) => post(body) }))
const { watchMomentsHeard } = await import('./momentsHeard')

describe('watchMomentsHeard (operator 2026-10-12)', () => {
  beforeEach(() => post.mockClear())

  it('posts a heard clip with its short listen, and a keep, once each', async () => {
    const event = ref<MomentEvent | null>(null)
    watchMomentsHeard({ event: () => event.value, signedIn: () => true })
    event.value = {
      seq: 1,
      kind: 'clip',
      slug: 'ep1',
      insightId: 'i1',
      seconds: 31.26,
      topic: { id: 'topic:risk', label: 'risk management', minutes: 4, kind: 'topic' },
    }
    await nextTick()
    event.value = { seq: 2, kind: 'keep', slug: 'ep1' }
    await nextTick()
    expect(post).toHaveBeenCalledTimes(2)
    expect(post.mock.calls[0][0]).toMatchObject({
      kind: 'clip',
      slug: 'ep1',
      insight_id: 'i1',
      seconds: 31.3,
      subject_id: 'topic:risk',
      subject_kind: 'topic',
      subject_label: 'risk management',
    })
    expect(post.mock.calls[1][0]).toMatchObject({ kind: 'keep', slug: 'ep1' })
    expect(post.mock.calls[1][0]).not.toHaveProperty('subject_id')
  })

  it('counts a board’s clip but never names the board as a subject (#2340)', async () => {
    const event = ref<MomentEvent | null>(null)
    watchMomentsHeard({ event: () => event.value, signedIn: () => true })
    event.value = {
      seq: 1,
      kind: 'clip',
      slug: 'ep1',
      insightId: 'i1',
      seconds: 12,
      topic: { id: 'board:col_abc', label: 'Deep dive', minutes: 15, kind: 'board' },
    }
    await nextTick()
    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ kind: 'clip', slug: 'ep1', seconds: 12 })
    expect(post.mock.calls[0][0]).not.toHaveProperty('subject_id')
    expect(post.mock.calls[0][0]).not.toHaveProperty('subject_kind')
  })

  it('sends nothing signed out', async () => {
    const event = ref<MomentEvent | null>(null)
    watchMomentsHeard({ event: () => event.value, signedIn: () => false })
    event.value = { seq: 1, kind: 'clip', slug: 'ep1' }
    await nextTick()
    expect(post).not.toHaveBeenCalled()
  })
})
