import { mount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it } from 'vitest'
import CaptureMoment from './CaptureMoment.vue'
import en from '../i18n/locales/en.json'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/library', name: 'library', component: { template: '<div/>' } },
  ],
})

async function mountAt(props: Record<string, unknown> = {}) {
  router.push('/')
  await router.isReady()
  return mount(CaptureMoment, { props, global: { plugins: [i18n, router] } })
}

describe('CaptureMoment (#1592)', () => {
  it('is a plain button when idle, and captures on click', async () => {
    const w = await mountAt()
    const btn = w.get('[data-testid="capture-moment"]')
    expect(btn.attributes('aria-label')).toBe(en.capture.markMoment)
    await btn.trigger('click')
    expect(w.emitted('capture')).toHaveLength(1)
  })

  it('becomes a FOLLOWABLE receipt when saved, pointing at where the capture went', async () => {
    // The whole point of the saved state. A capture that confirms itself and vanishes leaves a new
    // user with no idea the thing they saved lives under Library → Saved.
    const w = await mountAt({ state: 'saved' })
    const receipt = w.get('[data-testid="capture-receipt"]')
    expect(receipt.element.tagName).toBe('A')
    expect(receipt.attributes('href')).toBe('/library?tab=saved')
    expect(receipt.text()).toContain(en.capture.savedGoToHighlights)
    // It is a link, so it must NOT also be the capture button.
    expect(w.find('[data-testid="capture-moment"]').exists()).toBe(false)
  })

  it('shows failure VISIBLY, and stays tappable to retry', async () => {
    // The bug this state exists for: failure previously announced into an sr-only region and set no
    // visual state, so a sighted user could not tell a failed save from a missed tap.
    const w = await mountAt({ state: 'failed' })
    const btn = w.get('[data-testid="capture-moment"]')
    expect(btn.text()).toContain(en.capture.saveFailed)
    expect(btn.classes().join(' ')).toContain('text-danger')
    await btn.trigger('click')
    expect(w.emitted('capture')).toHaveLength(1)
  })

  it('does not rely on colour alone to say "failed"', async () => {
    // Colour-only state is unreadable to a colour-blind user and invisible in a greyscale
    // screenshot. The failed glyph carries two extra marks the other states do not have.
    const failed = await mountAt({ state: 'failed' })
    const idle = await mountAt()
    expect(failed.findAll('path').length).toBeGreaterThan(idle.findAll('path').length)
  })

  it('offers sign-in wording when gated, without pretending anything was saved', async () => {
    const w = await mountAt({ gated: true })
    const btn = w.get('[data-testid="capture-moment"]')
    expect(btn.attributes('aria-label')).toBe(en.auth.signInToCapture)
    expect(w.find('[data-testid="capture-receipt"]').exists()).toBe(false)
  })

  it('renders an icon when idle and a labelled control once there is an outcome', async () => {
    // Idle is icon-only so it sits quietly in a transport row; an outcome earns words.
    const idle = await mountAt()
    expect(idle.get('[data-testid="capture-moment"]').text()).toBe('')

    const saved = await mountAt({ state: 'saved' })
    expect(saved.get('[data-testid="capture-receipt"]').text()).not.toBe('')
  })
})
