// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import ArtifactActivityChart from './ArtifactActivityChart.vue'

const SILENCE = 'No new artifacts in 14 days'
const today = new Date().toISOString()

describe('ArtifactActivityChart insight (UXS-006 §4.3)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('says nothing about silence before the artifact list has loaded', () => {
    // An empty list that was never fetched is "unknown", not "14 silent days" — the Dashboard
    // used to read it that way and raise a false alarm on an active corpus.
    const w = mount(ArtifactActivityChart, { props: { artifactItems: [], loaded: false } })
    expect(w.text()).not.toContain(SILENCE)
    expect(w.text()).not.toContain('Last GI:')
  })

  it('calls out 14 silent days once a loaded list really has no recent artifacts', () => {
    const w = mount(ArtifactActivityChart, { props: { artifactItems: [], loaded: true } })
    expect(w.text()).toContain(SILENCE)
  })

  it('names the newest GI and KG days from a loaded list', () => {
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
    expect(w.text()).toContain(`Last GI: ${day} · Last KG: ${day}`)
    expect(w.text()).not.toContain(SILENCE)
  })
})
