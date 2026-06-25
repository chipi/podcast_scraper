/**
 * i18n setup (vue-i18n). English ships first; the catalog structure is RTL-ready and adding
 * a locale requires no code change (UXS-011 / Epic-2 DoD: easily translatable from line 1).
 * No hard-coded user-facing strings in components — all copy flows through `t()`.
 */

import { createI18n } from 'vue-i18n'
import en from './locales/en.json'

export const SUPPORTED_LOCALES = ['en'] as const
export type Locale = (typeof SUPPORTED_LOCALES)[number]

export const i18n = createI18n({
  legacy: false,
  locale: 'en',
  fallbackLocale: 'en',
  messages: { en },
})
