import { describe, expect, it } from 'vitest'

import { personName, personNameFromId } from './personName'

describe('personName', () => {
  it('capitalises a lowercase name — the case this exists for', () => {
    expect(personName('simon wilson')).toBe('Simon Wilson')
    expect(personName('lenny rachitsky')).toBe('Lenny Rachitsky')
  })

  it('leaves an already-correct name untouched', () => {
    expect(personName('Lenny Rachitsky')).toBe('Lenny Rachitsky')
  })

  // The invariant. Every case below is one a naive `toLowerCase().replace(/\b\w/g, ...)` destroys.
  describe('never lowercases a letter that is already uppercase', () => {
    it.each([
      ['McCarthy', 'McCarthy'],
      ['MacLeod', 'MacLeod'],
      ['LeBron James', 'LeBron James'],
      ['DeAndre Jordan', 'DeAndre Jordan'],
      ["O'Brien", "O'Brien"],
      ['IBM', 'IBM'],
      ['JFK', 'JFK'],
    ])('%s stays %s', (input, expected) => {
      expect(personName(input)).toBe(expected)
    })
  })

  describe('internal capitalisation after a separator', () => {
    it('capitalises after an apostrophe', () => {
      expect(personName("o'brien")).toBe("O'Brien")
      expect(personName('d’angelo')).toBe('D’Angelo') // typographic apostrophe
    })

    it('capitalises each part of a hyphenated name', () => {
      expect(personName('jean-luc picard')).toBe('Jean-Luc Picard')
      expect(personName('mary-kate olsen')).toBe('Mary-Kate Olsen')
    })
  })

  describe('particles', () => {
    it('keeps a particle lowercase mid-name', () => {
      expect(personName('ludwig van beethoven')).toBe('Ludwig van Beethoven')
      expect(personName('vincent van gogh')).toBe('Vincent van Gogh')
      expect(personName('jan van der berg')).toBe('Jan van der Berg')
      expect(personName('charles de gaulle')).toBe('Charles de Gaulle')
    })

    it('capitalises a particle that LEADS the name, because then it is the name', () => {
      expect(personName('van halen')).toBe('Van Halen')
      expect(personName('de niro')).toBe('De Niro')
    })

    it('does not lowercase a particle the envelope already capitalised', () => {
      // Someone who spells it "Van Der Berg" keeps it: the invariant beats the particle rule.
      expect(personName('Jan Van Der Berg')).toBe('Jan Van Der Berg')
    })
  })

  describe('degenerate input', () => {
    it.each([
      [null, ''],
      [undefined, ''],
      ['', ''],
      ['   ', ''],
    ])('%s -> %s', (input, expected) => {
      expect(personName(input)).toBe(expected)
    })

    it('trims and collapses whitespace', () => {
      expect(personName('  simon   wilson  ')).toBe('Simon Wilson')
    })

    it('leaves a single name alone', () => {
      expect(personName('cher')).toBe('Cher')
    })
  })

  it('is idempotent — running it twice changes nothing', () => {
    for (const n of ['simon wilson', 'jan van der berg', "o'brien", 'McCarthy', 'jean-luc picard']) {
      expect(personName(personName(n))).toBe(personName(n))
    }
  })

  // Documented as WRONG rather than left to be discovered. See the module docstring.
  it('capitalises a deliberately-lowercase name — known, accepted', () => {
    expect(personName('danah boyd')).toBe('Danah Boyd')
  })
})

describe('personNameFromId', () => {
  it('de-slugs and capitalises', () => {
    expect(personNameFromId('simon-wilson')).toBe('Simon Wilson')
    expect(personNameFromId('ada_lovelace')).toBe('Ada Lovelace')
  })

  it('strips a prefix', () => {
    expect(personNameFromId('person:simon-wilson')).toBe('Simon Wilson')
  })

  it('is empty for an empty id', () => {
    expect(personNameFromId('')).toBe('')
    expect(personNameFromId(null)).toBe('')
  })

  it('does NOT split a real hyphenated name when used on a name', () => {
    // The reason this is a separate function: personName must not de-slug, or `Jean-Luc` loses its
    // hyphen. This asserts the two stay different.
    expect(personName('Jean-Luc Picard')).toBe('Jean-Luc Picard')
    expect(personNameFromId('jean-luc-picard')).toBe('Jean Luc Picard')
  })
})
