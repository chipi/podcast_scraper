import { describe, expect, it } from 'vitest'

import { i18n } from './index'
import en from './locales/en.json'

/** Every leaf key of a nested catalog, as the dotted path `t()` takes. */
function leafKeys(node: unknown, prefix = ''): string[] {
  if (typeof node === 'string') return [prefix]
  if (Array.isArray(node) || node === null || typeof node !== 'object') return []
  return Object.entries(node).flatMap(([k, v]) => leafKeys(v, prefix ? `${prefix}.${k}` : k))
}

// vue-i18n compiles a message the first time it renders, so a syntax error surfaces as a THROW
// inside whatever component renders it — which blanks that component, with nothing on screen to
// say why. The magic-link placeholder `you@example.com` did exactly that (2026-10-03): `@` opens a
// linked-message reference, so opening the email form on /login blanked the page on web and
// native alike. A literal `@` must be written `{'@'}`; the same goes for `{`, `}`, `|` and `$`.
describe('every English message compiles', () => {
  it.each(leafKeys(en))('%s', (key) => {
    expect(() => i18n.global.t(key, {})).not.toThrow()
  })
})
