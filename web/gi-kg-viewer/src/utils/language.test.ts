import { describe, expect, it } from 'vitest'
import { primaryLanguage, spansLanguages } from './language'

describe('primaryLanguage', () => {
  it.each([
    ['en-US', 'en'],
    ['EN', 'en'],
    [' es ', 'es'],
    ['pt_BR', 'pt'],
    ['', null],
    [null, null],
    [undefined, null],
  ])('%s → %s', (tag, expected) => {
    expect(primaryLanguage(tag)).toBe(expected)
  })
})

describe('spansLanguages', () => {
  it('is true only with two distinct languages', () => {
    expect(spansLanguages(['en', 'es'])).toBe(true)
    expect(spansLanguages(['en', 'en'])).toBe(false)
    expect(spansLanguages(['en'])).toBe(false)
    expect(spansLanguages([])).toBe(false)
  })

  it('counts regional spellings of one language once and ignores unknowns', () => {
    expect(spansLanguages(['en-US', 'en-GB', 'EN', null, ''])).toBe(false)
  })
})
