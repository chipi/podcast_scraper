/** Fixed degree buckets for histogram + filter. */

export const DEGREE_BUCKET_ORDER = ['0', '1', '2-5', '6-10', '11+'] as const

export type DegreeBucketId = (typeof DEGREE_BUCKET_ORDER)[number]

export function degreeBucketFor(deg: number): DegreeBucketId {
  if (deg <= 0) {
    return '0'
  }
  if (deg === 1) {
    return '1'
  }
  if (deg >= 2 && deg <= 5) {
    return '2-5'
  }
  if (deg >= 6 && deg <= 10) {
    return '6-10'
  }
  return '11+'
}

export function emptyDegreeCounts(): Record<DegreeBucketId, number> {
  return { '0': 0, '1': 0, '2-5': 0, '6-10': 0, '11+': 0 }
}
