import { nextTick, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { LearnEnd } from '../stores/player'
import { watchLearnDayHeard } from './learnDayHeard'

const api = vi.hoisted(() => ({ learnStepDone: vi.fn(), learnOpenNext: vi.fn() }))
vi.mock('./api', () => api)
vi.mock('./analytics', () => ({ track: vi.fn() }))

async function settled(): Promise<void> {
  for (let i = 0; i < 6; i++) await Promise.resolve()
  await nextTick()
}

describe('a learning day\'s clip heard to its end (operator 2026-10-14)', () => {
  const end = ref<LearnEnd | null>(null)
  const settle = vi.fn((seq: number, status: LearnEnd['status']) => {
    if (end.value?.seq === seq) end.value = { ...end.value, status }
  })
  const fire = (over: Partial<LearnEnd>): void => {
    end.value = {
      subjectId: 'thc:x',
      kind: 'storyline',
      day: 2,
      seq: (end.value?.seq ?? 0) + 1,
      heard: true,
      openNext: false,
      status: 'pending',
      ...over,
    }
  }
  beforeEach(() => {
    end.value = null
    settle.mockClear()
    api.learnStepDone.mockReset().mockResolvedValue({})
    api.learnOpenNext.mockReset().mockResolvedValue({})
  })

  it('marks the day done', async () => {
    const stop = watchLearnDayHeard({ event: () => end.value, settle })
    fire({})
    await settled()
    expect(api.learnStepDone).toHaveBeenCalledWith('thc:x', 1)
    expect(api.learnOpenNext).not.toHaveBeenCalled()
    expect(end.value?.status).toBe('done')
    stop()
  })

  it('"Play all" opens the next day once this one is done', async () => {
    const stop = watchLearnDayHeard({ event: () => end.value, settle })
    fire({ openNext: true, all: true })
    await settled()
    expect(api.learnStepDone).toHaveBeenCalledWith('thc:x', 1)
    expect(api.learnOpenNext).toHaveBeenCalledWith('thc:x')
    expect(end.value?.status).toBe('done')
    stop()
  })

  it('a clip skipped through leaves the day for the listener', async () => {
    const stop = watchLearnDayHeard({ event: () => end.value, settle })
    fire({ heard: false })
    await settled()
    expect(api.learnStepDone).not.toHaveBeenCalled()
    expect(end.value?.status).toBe('not_heard')
    stop()
  })

  it('a save that fails says so, and opens nothing', async () => {
    api.learnStepDone.mockRejectedValue(new Error('500'))
    const stop = watchLearnDayHeard({ event: () => end.value, settle })
    fire({ openNext: true })
    await settled()
    expect(end.value?.status).toBe('failed')
    expect(api.learnOpenNext).not.toHaveBeenCalled()
    stop()
  })
})
