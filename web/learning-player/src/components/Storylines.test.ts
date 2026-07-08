import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import type { Storyline } from '../services/types'
import Storylines from './Storylines.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = (setup?: () => void) => {
  setActivePinia(createPinia()) // fresh pinia; signed out unless setup signs in
  setup?.()
  return mount(Storylines, { global: { plugins: [i18n] } })
}

const STORYLINES: Storyline[] = [
  { id: 'thc:shadow-fleet', label: 'Shadow-fleet economics', size: 5, anchor_topic_id: 'topic:sanctions' },
  { id: 'thc:ai-safety', label: 'AI safety', size: 3, anchor_topic_id: 'topic:alignment' },
]
const withStorylines = (items = STORYLINES) => vi.spyOn(api, 'getStorylines').mockResolvedValue(items)
const signIn = () => {
  const auth = useAuthStore()
  auth.user = { user_id: 'u_1', email: 'd@l', name: 'Dev' } // → isAuthenticated
}

afterEach(() => vi.restoreAllMocks())

describe('Storylines rail', () => {
  it('renders a chip per storyline with its label and topic count', async () => {
    withStorylines()
    const w = mountIt()
    await flushPromises()
    const chips = w.findAll('[data-testid="storyline-chip"]')
    expect(chips).toHaveLength(2)
    expect(chips[0].text()).toContain('Shadow-fleet economics')
    expect(chips[0].text()).toContain('5 topics')
  })

  it('opens the anchor topic (not the thc: id) when the chip body is tapped', async () => {
    withStorylines()
    const w = mountIt()
    await flushPromises()
    await w.findAll('[data-testid="storyline-chip"]')[0].get('button').trigger('click')
    expect(w.emitted('open')![0]).toEqual(['topic:sanctions'])
  })

  it('hides entirely when the corpus has no storylines', async () => {
    withStorylines([])
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-storylines"]').exists()).toBe(false)
  })

  it('signed out: no follow buttons', async () => {
    withStorylines()
    const w = mountIt()
    await flushPromises()
    expect(w.findAll('[data-testid="storyline-follow"]')).toHaveLength(0)
  })

  it('signed in: the follow button adds the theme cluster (thc:) to interests', async () => {
    withStorylines()
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const add = vi.spyOn(api, 'addInterest').mockResolvedValue(['thc:shadow-fleet'])
    const w = mountIt(signIn)
    await flushPromises()

    const follow = w.findAll('[data-testid="storyline-follow"]')
    expect(follow).toHaveLength(2)
    expect(follow[0].attributes('aria-pressed')).toBe('false')

    await follow[0].trigger('click')
    await flushPromises()

    expect(add).toHaveBeenCalledWith('thc:shadow-fleet')
    expect(w.findAll('[data-testid="storyline-follow"]')[0].attributes('aria-pressed')).toBe('true')
  })

  it('signed in: a chip whose thc: is already followed starts pressed', async () => {
    withStorylines()
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['thc:ai-safety'])
    const w = mountIt(signIn)
    await flushPromises()
    const follow = w.findAll('[data-testid="storyline-follow"]')
    expect(follow[0].attributes('aria-pressed')).toBe('false') // shadow-fleet not followed
    expect(follow[1].attributes('aria-pressed')).toBe('true') // ai-safety followed
  })
})
