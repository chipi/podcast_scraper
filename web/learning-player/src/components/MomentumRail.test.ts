import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import type { TrendingEntity } from '../services/types'
import MomentumRail from './MomentumRail.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountRail = (setup?: () => void) => {
  setActivePinia(createPinia())
  setup?.()
  return mount(MomentumRail, {
    props: { kind: 'topic', title: 'Trending now' },
    global: { plugins: [i18n] },
  })
}

const ENT = (over: Partial<TrendingEntity>): TrendingEntity => ({
  entity_id: 'topic:ai',
  kind: 'topic',
  label: 'AI',
  velocity: 2.1,
  volume: 3,
  heating_up: true,
  total: 9,
  series: [0, 1, 2, 4],
  ...over,
})
const withTrending = (items: TrendingEntity[]) =>
  vi.spyOn(api, 'getTrending').mockResolvedValue(items)
const signIn = () => {
  const auth = useAuthStore()
  auth.user = { user_id: 'u1', email: 'd@l', name: 'Dev' }
}

afterEach(() => vi.restoreAllMocks())

describe('MomentumRail', () => {
  it('renders a chip per trending entity with its label + velocity + sparkline', async () => {
    withTrending([ENT({}), ENT({ entity_id: 'topic:health', label: 'Health', velocity: 1.6 })])
    const w = mountRail()
    await flushPromises()
    const chips = w.findAll('[data-testid="momentum-chip"]')
    expect(chips).toHaveLength(2)
    expect(chips[0].text()).toContain('AI')
    expect(chips[0].text()).toContain('2.1×')
    expect(chips[0].find('svg').exists()).toBe(true) // sparkline
  })

  it('emits open with the entity when a chip body is tapped', async () => {
    withTrending([ENT({})])
    const w = mountRail()
    await flushPromises()
    await w.findAll('[data-testid="momentum-chip"]')[0].get('button').trigger('click')
    expect(w.emitted('open')![0][0]).toMatchObject({ entity_id: 'topic:ai', kind: 'topic' })
  })

  it('hides entirely when nothing is trending', async () => {
    withTrending([])
    const w = mountRail()
    await flushPromises()
    expect(w.find('[data-testid="momentum-rail-topic"]').exists()).toBe(false)
  })

  it('signed out: no follow buttons', async () => {
    withTrending([ENT({})])
    const w = mountRail()
    await flushPromises()
    expect(w.findAll('[data-testid="momentum-follow"]')).toHaveLength(0)
  })

  it('signed in: follow adds the interest token for a followable kind', async () => {
    withTrending([ENT({})])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const add = vi.spyOn(api, 'addInterest').mockResolvedValue(['topic:ai'])
    const w = mountRail(signIn)
    await flushPromises()
    await w.find('[data-testid="momentum-follow"]').trigger('click')
    await flushPromises()
    expect(add).toHaveBeenCalledWith('topic:ai')
  })

  it('signed in: episodes (not interest tokens) have no follow button', async () => {
    withTrending([ENT({ entity_id: 'ep-hot', kind: 'episode', label: 'Hot' })])
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    const w = mountRail(signIn)
    await flushPromises()
    expect(w.findAll('[data-testid="momentum-follow"]')).toHaveLength(0)
  })
})
