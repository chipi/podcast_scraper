import { describe, expect, it } from 'vitest'
import {
  DEGREE_BUCKET_ORDER,
  degreeBucketFor,
  emptyDegreeCounts,
} from './graphDegreeBuckets'

describe('degreeBucketFor', () => {
  it('maps zero and negatives to the "0" bucket', () => {
    expect(degreeBucketFor(0)).toBe('0')
    expect(degreeBucketFor(-3)).toBe('0')
  })

  it('maps exactly 1 to the "1" bucket', () => {
    expect(degreeBucketFor(1)).toBe('1')
  })

  it('maps the [2, 5] range inclusive to "2-5"', () => {
    expect(degreeBucketFor(2)).toBe('2-5')
    expect(degreeBucketFor(3)).toBe('2-5')
    expect(degreeBucketFor(5)).toBe('2-5')
  })

  it('maps the [6, 10] range inclusive to "6-10"', () => {
    expect(degreeBucketFor(6)).toBe('6-10')
    expect(degreeBucketFor(9)).toBe('6-10')
    expect(degreeBucketFor(10)).toBe('6-10')
  })

  it('maps 11 and above to the open-ended "11+" bucket', () => {
    expect(degreeBucketFor(11)).toBe('11+')
    expect(degreeBucketFor(999)).toBe('11+')
  })
})

describe('emptyDegreeCounts', () => {
  it('returns a fresh record with every bucket zeroed and every key from the fixed order', () => {
    const counts = emptyDegreeCounts()
    for (const k of DEGREE_BUCKET_ORDER) {
      expect(counts[k]).toBe(0)
    }
    expect(Object.keys(counts).sort()).toEqual([...DEGREE_BUCKET_ORDER].sort())
  })

  it('returns a NEW object each call — no shared mutable state', () => {
    const a = emptyDegreeCounts()
    a['0'] = 5
    const b = emptyDegreeCounts()
    expect(b['0']).toBe(0)
  })
})
