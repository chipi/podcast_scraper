import { describe, expect, it } from 'vitest'
import { escapeHtml, formatBytes, humanizeSlug, truncate } from './formatting'

describe('escapeHtml', () => {
  it('escapes &, <, >, "', () => {
    expect(escapeHtml('a & b < c > d "e"')).toBe(
      'a &amp; b &lt; c &gt; d &quot;e&quot;',
    )
  })

  it('returns empty string unchanged', () => {
    expect(escapeHtml('')).toBe('')
  })

  it('leaves safe text untouched', () => {
    expect(escapeHtml('hello world')).toBe('hello world')
  })
})

describe('truncate', () => {
  it('returns short strings unchanged', () => {
    expect(truncate('abc', 10)).toBe('abc')
  })

  it('truncates at max-1 and appends ellipsis', () => {
    expect(truncate('abcdefghij', 5)).toBe('abcd…')
  })

  it('handles exact boundary', () => {
    expect(truncate('abcde', 5)).toBe('abcde')
  })

  it('handles empty string', () => {
    expect(truncate('', 5)).toBe('')
  })
})

describe('humanizeSlug', () => {
  it('capitalises and joins dashes', () => {
    expect(humanizeSlug('hello-world')).toBe('Hello World')
  })

  it('handles single word', () => {
    expect(humanizeSlug('test')).toBe('Test')
  })

  it('strips empty segments from leading/trailing dashes', () => {
    expect(humanizeSlug('-foo-bar-')).toBe('Foo Bar')
  })

  it('handles empty string', () => {
    expect(humanizeSlug('')).toBe('')
  })
})

describe('formatBytes', () => {
  it('formats bytes', () => {
    expect(formatBytes(512)).toBe('512 B')
  })

  it('formats kilobytes', () => {
    expect(formatBytes(2048)).toBe('2.0 KB')
  })

  it('formats megabytes', () => {
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })

  it('returns dash for negative', () => {
    expect(formatBytes(-1)).toBe('—')
  })

  it('returns dash for NaN', () => {
    expect(formatBytes(NaN)).toBe('—')
  })

  it('returns dash for Infinity', () => {
    expect(formatBytes(Infinity)).toBe('—')
  })

  it('handles zero', () => {
    expect(formatBytes(0)).toBe('0 B')
  })
})
