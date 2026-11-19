// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import PersonLandingView from './PersonLandingView.vue'
import { personProfileSections } from '../../extensions'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { ParsedArtifact } from '../../types/artifact'

// The view reads the person from the graph slice; GI (positions, connections, the brief) is
// registered sections' business, so nothing here may reach the relational or CIL routes.
const relational = vi.fn()
vi.mock('../../api/relationalApi', () => ({
  fetchPositions: relational,
  fetchPersonTopics: relational,
  fetchCoSpeakers: relational,
}))
vi.mock('../../api/cilApi', () => ({ fetchPersonProfile: relational }))

function personArtifact(properties: Record<string, unknown> = {}): ParsedArtifact {
  return {
    id: 'a1',
    kind: 'kg',
    data: { nodes: [{ id: 'person:alice', type: 'Person', properties: { name: 'Alice', ...properties } }], edges: [] },
  } as unknown as ParsedArtifact
}

async function mountView(
  view: 'full' | 'profile' | 'positions',
  properties: Record<string, unknown> = {},
  embedded = false,
) {
  useArtifactsStore().parsedList = [personArtifact(properties)]
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  shell.cilQueriesApiAvailable = true
  useSubjectStore().focusPerson('person:alice')
  const w = mount(PersonLandingView, {
    attachTo: document.body,
    props: { subjectIdOverride: 'person:alice', view, embedded },
  })
  await flushPromises()
  return w
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.clearAllMocks()
})

describe('PersonLandingView — identity (#1048, PRD-029 FR1)', () => {
  it('shows the name and a role badge with the Person.role value', async () => {
    const w = await mountView('full', { role: 'host' })
    expect(w.get('[data-testid="person-landing-view-name"]').text()).toBe('Alice')
    const badge = w.get('[data-testid="person-landing-role"]')
    expect(badge.text()).toBe('Host')
    expect(badge.attributes('data-role')).toBe('host')
  })

  it('the profile carries the role, aliases and description', async () => {
    const w = await mountView('profile', { role: 'guest', aliases: ['Al', 'A.'], description: 'A guest.' }, true)
    expect(w.get('[data-testid="person-landing-role-embedded"]').attributes('data-role')).toBe('guest')
    expect(w.get('[data-testid="person-landing-aliases"]').text()).toContain('Al, A.')
    expect(w.get('[data-testid="person-landing-description"]').text()).toBe('A guest.')
  })

  it('the footer opens the graph and prefills search with the name', async () => {
    const w = await mountView('full')
    await w.get('[data-testid="person-landing-go-graph"]').trigger('click')
    await w.get('[data-testid="person-landing-prefill-search"]').trigger('click')
    expect(w.emitted('goGraph')).toHaveLength(1)
    expect(w.emitted('prefillSemanticSearch')?.[0]).toEqual([{ query: 'Alice' }])
  })
})

describe('PersonLandingView — no GI of its own', () => {
  it.each(['full', 'profile', 'positions'] as const)('the %s view renders no GI and calls no GI route', async (view) => {
    const w = await mountView(view)
    for (const id of [
      'person-landing-episode-count',
      'person-landing-organizations',
      'person-landing-connections',
      'person-landing-episodes-appeared',
      'person-landing-ranked-topics',
      'person-landing-insights-voiced',
      'person-landing-positions-lens',
      'person-landing-stated',
      'person-landing-positions',
      'position-tracker-panel',
    ]) {
      expect(w.find(`[data-testid="${id}"]`).exists(), id).toBe(false)
    }
    expect(relational).not.toHaveBeenCalled()
  })
})

describe('PersonLandingView — personProfileSections', () => {
  afterEach(() => personProfileSections.reset())

  function registerAll(): void {
    for (const region of ['header', 'profile-head', 'profile', 'positions', 'full'] as const) {
      personProfileSections.register({
        id: region,
        region,
        component: {
          props: ['personId', 'view'],
          template: `<span data-testid="ext-${region}">{{ personId }}/{{ view }}</span>`,
        },
      })
    }
  }
  const shown = (w: ReturnType<typeof mount>) =>
    ['header', 'profile-head', 'profile', 'positions', 'full'].filter((r) => w.find(`[data-testid="ext-${r}"]`).exists())

  it('the full view shows the header, profile-head, profile and full sections, with the person', async () => {
    registerAll()
    const w = await mountView('full')
    expect(shown(w)).toEqual(['header', 'profile-head', 'profile', 'full'])
    expect(w.get('[data-testid="ext-header"]').text()).toBe('person:alice/full')
    w.unmount()
  })

  it('a profile-head section renders after the role and before the aliases', async () => {
    registerAll()
    const w = await mountView('full', { aliases: ['Al'], role: 'host' })
    const html = w.html()
    expect(html.indexOf('person-landing-role-embedded')).toBeLessThan(html.indexOf('ext-profile-head'))
    expect(html.indexOf('ext-profile-head')).toBeLessThan(html.indexOf('person-landing-aliases'))
    w.unmount()
  })

  it('the positions view shows the header and positions sections', async () => {
    registerAll()
    const w = await mountView('positions')
    expect(shown(w)).toEqual(['header', 'positions'])
    expect(w.get('[data-testid="ext-positions"]').text()).toBe('person:alice/positions')
    w.unmount()
  })
})
