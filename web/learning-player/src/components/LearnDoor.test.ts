/**
 * "Learn this in a week" on a topic, theme or storyline (operator 2026-10-11): shown only with a
 * path; "day N of 7" when already learning it; asks in place before replacing another path.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'

const api = vi.hoisted(() => ({ getLearn: vi.fn(), getLearnSubject: vi.fn(), startLearn: vi.fn() }))
vi.mock('../services/api', async (orig) => ({ ...(await orig<typeof import('../services/api')>()), ...api }))
const tracked = vi.hoisted(() => [] as Array<{ name: string; props: unknown }>)
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import LearnDoor from './LearnDoor.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', component: { template: '<div/>' } },
      { path: '/learn', name: 'learn', component: { template: '<div/>' } },
    ],
  })
}
const subject = { subject_id: 'thc:risk', subject_kind: 'storyline', subject_label: 'Managing risk', steps: 7 }
const pathOn = (id: string, label: string) => ({
  subject_id: id,
  subject_kind: 'theme',
  subject_label: label,
  started_at: 0,
  day: 3,
  steps: Array.from({ length: 7 }, () => ({})),
})
async function mountDoor() {
  const router = makeRouter()
  await router.push('/')
  const w = mount(LearnDoor, { props: { subjectId: 'thc:risk' }, global: { plugins: [i18n, router] } })
  await flushPromises()
  return { w, router }
}

afterEach(() => vi.clearAllMocks())

describe('LearnDoor', () => {
  it('starts a path and opens the week, when nothing else is being learned', async () => {
    api.getLearnSubject.mockResolvedValue(subject)
    api.getLearn.mockResolvedValue({ path: null, candidates: [] })
    api.startLearn.mockResolvedValue({})
    const { w, router } = await mountDoor()
    const door = w.get('[data-testid="learn-door"]')
    expect(door.text()).toBe('Learn this in 7 days')
    await door.trigger('click')
    await flushPromises()
    expect(api.startLearn).toHaveBeenCalledWith('thc:risk')
    expect(router.currentRoute.value.name).toBe('learn')
    expect(tracked.at(-1)).toEqual({
      name: 'learn_path',
      props: { action: 'start', kind: 'storyline', source: 'subject_page' },
    })
  })

  it('asks in place before replacing another path', async () => {
    api.getLearnSubject.mockResolvedValue(subject)
    api.getLearn.mockResolvedValue({ path: pathOn('tc:other', 'Reliability'), candidates: [] })
    api.startLearn.mockResolvedValue({})
    const { w } = await mountDoor()
    await w.get('[data-testid="learn-door"]').trigger('click')
    expect(api.startLearn).not.toHaveBeenCalled()
    expect(w.get('[data-testid="learn-door-confirm"]').text()).toContain('This replaces “Reliability”.')
    await w.get('[data-testid="learn-door-yes"]').trigger('click')
    await flushPromises()
    expect(api.startLearn).toHaveBeenCalledWith('thc:risk')
  })

  it('says the day when already learning this, and is absent without a path', async () => {
    api.getLearnSubject.mockResolvedValueOnce(subject)
    api.getLearn.mockResolvedValue({ path: pathOn('thc:risk', 'Managing risk'), candidates: [] })
    const { w } = await mountDoor()
    expect(w.get('[data-testid="learn-door"]').text()).toBe('Learning · day 3 of 7 ›')
    api.getLearnSubject.mockResolvedValueOnce(null)
    const none = await mountDoor()
    expect(none.w.find('[data-testid="learn-door"]').exists()).toBe(false)
  })
})
