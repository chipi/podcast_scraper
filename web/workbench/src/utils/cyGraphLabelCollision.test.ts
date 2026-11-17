import { describe, expect, it } from 'vitest'
import { labelsToHide, type LabelBox } from './cyGraphLabelCollision'

const box = (id: string, x: number, y: number, priority = 0, w = 40, h = 10): LabelBox => ({
  id,
  x1: x,
  y1: y,
  x2: x + w,
  y2: y + h,
  priority,
})

describe('labelsToHide', () => {
  it('keeps both labels when they do not overlap', () => {
    expect(labelsToHide([box('a', 0, 0), box('b', 100, 0)])).toEqual(new Set())
  })

  it('hides the lower-priority label of an overlapping pair', () => {
    expect(labelsToHide([box('a', 0, 0, 1), box('b', 10, 2, 5)])).toEqual(new Set(['a']))
  })

  it('breaks priority ties by input order', () => {
    expect(labelsToHide([box('first', 0, 0), box('second', 5, 5)])).toEqual(new Set(['second']))
  })

  it('a hidden label does not block a third one it overlaps', () => {
    // b loses to a; c overlaps only b, so c stays.
    const hide = labelsToHide([box('a', 0, 0, 3), box('b', 30, 0, 2), box('c', 60, 0, 1)])
    expect(hide).toEqual(new Set(['b']))
  })

  it('treats labels closer than the gap as overlapping', () => {
    expect(labelsToHide([box('a', 0, 0, 1), box('b', 41, 0)], 2)).toEqual(new Set(['b']))
    expect(labelsToHide([box('a', 0, 0, 1), box('b', 43, 0)], 2)).toEqual(new Set())
  })

  it('finds overlaps across grid cells for labels of very different sizes', () => {
    const wide = box('wide', 0, 0, 1, 400, 10)
    const small = box('small', 350, 5, 0, 20, 10)
    expect(labelsToHide([wide, small, box('x', 1000, 0), box('y', 2000, 0)])).toEqual(new Set(['small']))
  })
})
