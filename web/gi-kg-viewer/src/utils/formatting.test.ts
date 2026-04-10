import { describe, expect, it } from 'vitest'
import {
  escapeHtml,
  formatBytes,
  formatUtcDateTimeForDisplay,
  humanizeSlug,
  shortPhrase,
  truncate,
} from './formatting'

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

describe('shortPhrase', () => {
  it('returns short strings unchanged', () => {
    expect(shortPhrase('hello world', 40)).toBe('hello world')
  })

  it('cuts at comma within max', () => {
    expect(shortPhrase('Cuba faces a severe crisis, leading to blackouts', 40)).toBe(
      'Cuba faces a severe crisis',
    )
  })

  it('cuts at semicolon within max', () => {
    expect(shortPhrase('First clause; second clause is longer than max', 20)).toBe(
      'First clause',
    )
  })

  it('cuts at period within max', () => {
    expect(shortPhrase('Short sentence. Then more text that goes on and on', 20)).toBe(
      'Short sentence',
    )
  })

  it('cuts at em-dash within max', () => {
    expect(shortPhrase('Before dash—after dash continues here for a while', 20)).toBe(
      'Before dash',
    )
  })

  it('ignores break too early (< 8 chars) and word-breaks instead', () => {
    expect(shortPhrase('A, but this sentence is quite long and keeps going', 30)).toBe(
      'A, but this sentence is quite…',
    )
  })

  it('word-breaks with ellipsis when no natural break', () => {
    expect(shortPhrase('The quick brown fox jumps over the lazy dog and more', 25)).toBe(
      'The quick brown fox…',
    )
  })

  it('hard-truncates when no spaces after position 8', () => {
    expect(shortPhrase('abcdefghijklmnopqrstuvwxyzabcdefghijklmnop', 20)).toBe(
      'abcdefghijklmnopqrs…',
    )
  })

  it('trims input whitespace', () => {
    expect(shortPhrase('  hello  ', 40)).toBe('hello')
  })

  it('uses default max of 40', () => {
    const long = 'A'.repeat(50)
    expect(shortPhrase(long).length).toBeLessThanOrEqual(40)
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

describe('formatUtcDateTimeForDisplay', () => {
  it('formats UTC instant without fractional seconds in output', () => {
    const out = formatUtcDateTimeForDisplay('2026-04-10T18:05:06.789Z')
    expect(out).toBe('Apr 10, 2026, 6:05 PM UTC')
    expect(out).not.toMatch(/\.\d/)
  })

  it('returns empty for blank', () => {
    expect(formatUtcDateTimeForDisplay('')).toBe('')
    expect(formatUtcDateTimeForDisplay('   ')).toBe('')
  })

  it('returns original when not parseable', () => {
    expect(formatUtcDateTimeForDisplay('not-a-date')).toBe('not-a-date')
  })

  it('handles +00:00 style ISO', () => {
    expect(formatUtcDateTimeForDisplay('2024-06-01T00:00:00.000+00:00')).toBe(
      'Jun 1, 2024, 12:00 AM UTC',
    )
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
