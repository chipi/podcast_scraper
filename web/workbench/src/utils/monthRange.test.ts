import { describe, expect, it } from 'vitest'
import { filterByMonthRange } from './monthRange'

const rows = [
  { month: '2026-01', n: 1 },
  { month: '2026-02', n: 2 },
  { month: '2026-03', n: 3 },
  { month: '2026-04', n: 4 },
]

describe('filterByMonthRange', () => {
  it('returns all rows when range is empty', () => {
    expect(filterByMonthRange(rows, '', '').map((r) => r.month)).toEqual([
      '2026-01',
      '2026-02',
      '2026-03',
      '2026-04',
    ])
  })

  it('applies an inclusive lower + upper bound', () => {
    expect(filterByMonthRange(rows, '2026-02', '2026-03').map((r) => r.month)).toEqual([
      '2026-02',
      '2026-03',
    ])
  })

  it('supports open-ended from / to', () => {
    expect(filterByMonthRange(rows, '2026-03', '').map((r) => r.month)).toEqual(['2026-03', '2026-04'])
    expect(filterByMonthRange(rows, '', '2026-02').map((r) => r.month)).toEqual(['2026-01', '2026-02'])
  })
})
