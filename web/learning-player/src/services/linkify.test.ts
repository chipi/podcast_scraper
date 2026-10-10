import { describe, expect, it } from 'vitest'
import { linkify } from './linkify'

describe('linkify — the publisher description, with its written-out links made tappable', () => {
  it('splits text around http(s) links', () => {
    expect(linkify('Show notes at https://example.com/ep1 and more')).toEqual([
      { text: 'Show notes at ' },
      { text: 'https://example.com/ep1', href: 'https://example.com/ep1' },
      { text: ' and more' },
    ])
  })

  it('treats a bare www. address as https', () => {
    expect(linkify('Visit www.example.org today')).toEqual([
      { text: 'Visit ' },
      { text: 'www.example.org', href: 'https://www.example.org' },
      { text: ' today' },
    ])
  })

  it('leaves sentence punctuation after a link out of it', () => {
    expect(linkify('See https://a.io/x. Then https://b.io/y, and (https://c.io/z)!')).toEqual([
      { text: 'See ' },
      { text: 'https://a.io/x', href: 'https://a.io/x' },
      { text: '. Then ' },
      { text: 'https://b.io/y', href: 'https://b.io/y' },
      { text: ', and (' },
      { text: 'https://c.io/z', href: 'https://c.io/z' },
      { text: ')!' },
    ])
  })

  it('keeps a closing bracket that belongs to the link', () => {
    const url = 'https://en.wikipedia.org/wiki/Mercury_(planet)'
    expect(linkify(`Read ${url}.`)).toEqual([{ text: 'Read ' }, { text: url, href: url }, { text: '.' }])
  })

  it('never links anything but http(s): javascript:, data:, mailto: stay text', () => {
    for (const s of ['javascript:alert(1)', 'data:text/html,<b>x</b>', 'mailto:a@b.co']) {
      expect(linkify(s)).toEqual([{ text: s }])
    }
  })

  it('text with no link is one plain segment; empty text is none', () => {
    expect(linkify('No links here.')).toEqual([{ text: 'No links here.' }])
    expect(linkify('')).toEqual([])
  })

  it('a link at the very start or end has no empty text beside it', () => {
    expect(linkify('https://a.io')).toEqual([{ text: 'https://a.io', href: 'https://a.io' }])
  })
})
