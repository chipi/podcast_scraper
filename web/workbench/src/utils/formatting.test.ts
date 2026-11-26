import { afterEach, describe, expect, it } from 'vitest'
import {
  escapeHtml,
  formatBytes,
  formatCalendarDateForDisplay,
  formatDateTimeForDisplay,
  localDay,
  parseStoredInstant,
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

describe('formatCalendarDateForDisplay', () => {
  it('formats ISO instants like a calendar date in en-US (locale-aware)', () => {
    const raw = '2024-06-15T15:00:00.000Z'
    const d = new Date(raw)
    const expected = new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    }).format(d)
    expect(formatCalendarDateForDisplay(raw)).toBe(expected)
  })

  it('returns empty for blank', () => {
    expect(formatCalendarDateForDisplay('')).toBe('')
    expect(formatCalendarDateForDisplay('   ')).toBe('')
  })

  it('truncates very long unparseable strings', () => {
    const s = 'x'.repeat(50)
    expect(formatCalendarDateForDisplay(s)).toHaveLength(40)
    expect(formatCalendarDateForDisplay(s).endsWith('…')).toBe(true)
  })

  it('returns short garbage unchanged', () => {
    expect(formatCalendarDateForDisplay('not-a-date')).toBe('not-a-date')
  })
})

const ORIGINAL_TZ = process.env.TZ

function inZone(tz: string, fn: () => void): void {
  process.env.TZ = tz
  fn()
}

afterEach(() => {
  if (ORIGINAL_TZ === undefined) delete process.env.TZ
  else process.env.TZ = ORIGINAL_TZ
})

const ZONES = ['America/New_York', 'Europe/Belgrade', 'Pacific/Auckland'] as const

describe('parseStoredInstant', () => {
  it('reads a naive ISO datetime as UTC in every zone', () => {
    for (const tz of ZONES) {
      inZone(tz, () => {
        expect(parseStoredInstant('2026-09-17T02:00:00')?.toISOString()).toBe('2026-09-17T02:00:00.000Z')
        expect(parseStoredInstant('2026-09-17 02:00:00.5')?.toISOString()).toBe('2026-09-17T02:00:00.500Z')
        expect(parseStoredInstant('2026-09-17T02:00')?.toISOString()).toBe('2026-09-17T02:00:00.000Z')
      })
    }
  })

  it('keeps Z and offset values', () => {
    expect(parseStoredInstant('2026-09-17T02:00:00Z')?.toISOString()).toBe('2026-09-17T02:00:00.000Z')
    expect(parseStoredInstant('2026-09-17T02:00:00+02:00')?.toISOString()).toBe('2026-09-17T00:00:00.000Z')
    expect(parseStoredInstant('2026-09-17T02:00:00-0500')?.toISOString()).toBe('2026-09-17T07:00:00.000Z')
  })

  it('reads epoch seconds and milliseconds', () => {
    expect(parseStoredInstant(1789005600)?.toISOString()).toBe('2026-09-10T02:00:00.000Z')
    expect(parseStoredInstant(1789005600000)?.toISOString()).toBe('2026-09-10T02:00:00.000Z')
  })

  it('is null for a bare date, blank and garbage', () => {
    expect(parseStoredInstant('2026-09-17')).toBeNull()
    expect(parseStoredInstant('')).toBeNull()
    expect(parseStoredInstant('   ')).toBeNull()
    expect(parseStoredInstant(null)).toBeNull()
    expect(parseStoredInstant(undefined)).toBeNull()
    expect(parseStoredInstant('not-a-date')).toBeNull()
    expect(parseStoredInstant('2026-13-45T99:00:00')).toBeNull()
    expect(parseStoredInstant(Number.NaN)).toBeNull()
  })
})

describe('localDay', () => {
  it('shows the viewer-local day of a naive (UTC) datetime', () => {
    inZone('America/New_York', () => {
      expect(localDay('2026-09-17T02:00:00')).toBe('2026-09-16')
    })
    inZone('Europe/Belgrade', () => {
      expect(localDay('2026-09-17T02:00:00')).toBe('2026-09-17')
      expect(localDay('2026-09-16T23:00:00')).toBe('2026-09-17')
    })
  })

  it('honours Z and offsets', () => {
    inZone('America/New_York', () => {
      expect(localDay('2026-09-17T02:00:00Z')).toBe('2026-09-16')
      expect(localDay('2026-09-17T02:00:00+02:00')).toBe('2026-09-16')
      expect(localDay('2026-09-17T22:30:00-05:00')).toBe('2026-09-17')
    })
  })

  it('returns a bare date unchanged in every zone', () => {
    for (const tz of ZONES) {
      inZone(tz, () => {
        expect(localDay('2026-09-17')).toBe('2026-09-17')
        expect(localDay(' 2026-01-01 ')).toBe('2026-01-01')
      })
    }
  })

  it('is null for garbage', () => {
    expect(localDay('')).toBeNull()
    expect(localDay(null)).toBeNull()
    expect(localDay('garbage')).toBeNull()
    expect(localDay('2026-02-31')).toBeNull()
  })

  it('handles epoch numbers', () => {
    inZone('America/New_York', () => {
      expect(localDay(1789005600)).toBe('2026-09-09')
    })
  })
})

describe('formatCalendarDateForDisplay (local zone)', () => {
  it('shows the local day for a naive UTC datetime', () => {
    inZone('America/New_York', () => {
      expect(formatCalendarDateForDisplay('2026-09-17T02:00:00')).toBe('Sep 16, 2026')
    })
    inZone('Europe/Belgrade', () => {
      expect(formatCalendarDateForDisplay('2026-09-17T02:00:00')).toBe('Sep 17, 2026')
    })
  })

  it('handles Z and offset', () => {
    inZone('America/New_York', () => {
      expect(formatCalendarDateForDisplay('2026-09-17T02:00:00Z')).toBe('Sep 16, 2026')
      expect(formatCalendarDateForDisplay('2026-09-17T02:00:00+02:00')).toBe('Sep 16, 2026')
    })
  })

  it('never shifts a bare date', () => {
    for (const tz of ZONES) {
      inZone(tz, () => {
        expect(formatCalendarDateForDisplay('2026-09-17')).toBe('Sep 17, 2026')
        expect(formatCalendarDateForDisplay('2026-01-01')).toBe('Jan 1, 2026')
      })
    }
  })
})

describe('formatDateTimeForDisplay', () => {
  it('formats in the viewer zone with the short zone name and no fractional seconds', () => {
    inZone('America/New_York', () => {
      const out = formatDateTimeForDisplay('2026-04-10T18:05:06.789Z')
      expect(out).toBe('Apr 10, 2026, 2:05 PM EDT')
      expect(out).not.toMatch(/\.\d/)
    })
  })

  it('reads a naive datetime as UTC', () => {
    inZone('America/New_York', () => {
      expect(formatDateTimeForDisplay('2026-09-17T02:00:00')).toBe('Sep 16, 2026, 10:00 PM EDT')
    })
    inZone('Europe/Belgrade', () => {
      expect(formatDateTimeForDisplay('2026-09-17T02:00:00')).toMatch(/^Sep 17, 2026, 4:00 AM GMT\+2$/)
    })
  })

  it('applies an explicit offset', () => {
    inZone('America/New_York', () => {
      expect(formatDateTimeForDisplay('2026-09-17T02:00:00+02:00')).toBe('Sep 16, 2026, 8:00 PM EDT')
    })
  })

  it('shows a bare date as that calendar day', () => {
    for (const tz of ZONES) {
      inZone(tz, () => {
        expect(formatDateTimeForDisplay('2026-09-17')).toBe('Sep 17, 2026')
      })
    }
  })

  it('returns empty for blank', () => {
    expect(formatDateTimeForDisplay('')).toBe('')
    expect(formatDateTimeForDisplay('   ')).toBe('')
  })

  it('returns original when not parseable', () => {
    expect(formatDateTimeForDisplay('not-a-date')).toBe('not-a-date')
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
