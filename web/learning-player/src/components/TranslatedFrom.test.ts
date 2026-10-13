import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import en from '../i18n/locales/en.json'
import TranslatedFrom from './TranslatedFrom.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountLabel(props: { lang?: string | null; inline?: boolean }) {
  return mount(TranslatedFrom, { props, global: { plugins: [i18n] } })
}

describe('TranslatedFrom (D-36 slice b)', () => {
  it('names the source language in the UI locale', () => {
    const label = mountLabel({ lang: 'es' }).get('[data-testid="translated-from"]')
    expect(label.text()).toBe('Translated from Spanish')
    expect(label.element.tagName).toBe('P')
  })

  it('renders inline when asked', () => {
    expect(mountLabel({ lang: 'pt', inline: true }).get('[data-testid="translated-from"]').element.tagName).toBe('SPAN')
  })

  it.each([undefined, null, '', '  '])('renders nothing for a native quote (%s)', (lang) => {
    expect(mountLabel({ lang }).find('[data-testid="translated-from"]').exists()).toBe(false)
  })
})
