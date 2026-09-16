/**
 * Regression guard for "tapping an episode inside an entity sheet lands on HOME" (2026-09-16).
 *
 * The row must navigate and do NOTHING else. When it also asked its host sheet to close, the host
 * took the explicit-close path — which still owes a `router.back()` to pop the history entry the
 * sheet pushed on open — and that `back()` ran before the route settled, popping the push to the
 * player. The user arrived back where the sheet was opened from, i.e. Home.
 *
 * The same defect had already been found and fixed on `EntityCardBody`'s "Open in page" link; it
 * simply never reached these rows. Hence a test rather than another comment.
 */
import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import EpisodeRow from './EpisodeRow.vue'
import type { EpisodeSummary } from '../services/types'

const episode = {
  slug: 'ep-1',
  title: 'Risk Is a Systems Property',
  podcast_title: 'Cross-Show',
} as unknown as EpisodeSummary

function router() {
  return createRouter({
    history: createWebHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    ],
  })
}

describe('EpisodeRow', () => {
  it('links to the player for its episode', async () => {
    const r = router()
    r.push('/')
    await r.isReady()
    const w = mount(EpisodeRow, { props: { episode }, global: { plugins: [r] } })
    expect(w.find('a').attributes('href')).toBe('/episode/ep-1')
  })

  it('emits NOTHING on tap — a close hook here pops the navigation it just made', async () => {
    const r = router()
    r.push('/')
    await r.isReady()
    const w = mount(EpisodeRow, { props: { episode }, global: { plugins: [r] } })
    await w.find('a').trigger('click')
    // `click` itself shows up here because vue-test-utils records native DOM events that reach the
    // root — that one is the browser's, not ours. What must NOT appear is a COMPONENT event, whose
    // only ever consumer was a host sheet wiring it to `close`, which is what broke the link.
    expect(w.emitted('navigate')).toBeUndefined()
    expect(w.emitted('close')).toBeUndefined()
  })
})
