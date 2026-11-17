// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { artifactActivitySeries } from '../../extensions'
import ArtifactActivityChart from './ArtifactActivityChart.vue'

const SILENCE = 'No new artifacts in 14 days'
const today = new Date().toISOString()

describe('ArtifactActivityChart insight (UXS-006 §4.3)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })
  afterEach(() => artifactActivitySeries.reset())

  it('says nothing about silence before the artifact list has loaded', () => {
    // An empty list that was never fetched is "unknown", not "14 silent days" — the Dashboard
    // used to read it that way and raise a false alarm on an active corpus.
    const w = mount(ArtifactActivityChart, { props: { artifactItems: [], loaded: false } })
    expect(w.text()).not.toContain(SILENCE)
    expect(w.text()).not.toContain('Last KG:')
  })

  it('calls out 14 silent days once a loaded list really has no recent artifacts', () => {
    const w = mount(ArtifactActivityChart, { props: { artifactItems: [], loaded: true } })
    expect(w.text()).toContain(SILENCE)
  })

  it('names the newest KG day; GI files are not its series (a downstream build adds one)', () => {
    const w = mount(ArtifactActivityChart, {
      props: {
        artifactItems: [
          { kind: 'gi', mtime_utc: today },
          { kind: 'kg', mtime_utc: today },
        ],
        loaded: true,
      },
    })
    const day = today.slice(0, 10)
    expect(w.text()).toContain(`Last KG: ${day}`)
    expect(w.text()).not.toContain('GI')
    expect(w.text()).not.toContain(SILENCE)
  })

  it('puts a registered series first: in the legend, the "Last" line and the silence check', () => {
    artifactActivitySeries.register({ id: 'probe', kind: 'probe', label: 'PR', token: '--ps-gi', textClass: 'text-gi' })
    const old = new Date(Date.now() - 20 * 86_400_000).toISOString()
    const w = mount(ArtifactActivityChart, {
      props: {
        artifactItems: [
          { kind: 'probe', mtime_utc: today },
          { kind: 'kg', mtime_utc: old },
        ],
        loaded: true,
      },
    })
    expect(w.text()).toContain(`Last PR: ${today.slice(0, 10)} · Last KG: ${old.slice(0, 10)}`)
    // A recent file of the registered kind is activity, so no silence alarm.
    expect(w.text()).not.toContain(SILENCE)
    expect(w.text()).toMatch(/PR new files \/ day\s*KG new files \/ day/)
  })
})
