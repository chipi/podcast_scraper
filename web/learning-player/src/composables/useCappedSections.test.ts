import { describe, expect, it } from 'vitest'
import { SECTION_CAP, useCappedSections } from './useCappedSections'

const many = (n: number): number[] => Array.from({ length: n }, (_, i) => i)

describe('useCappedSections', () => {
  it('caps to the top N until the section is expanded', () => {
    const c = useCappedSections()
    const items = many(SECTION_CAP + 4)
    expect(c.visible('a', items)).toHaveLength(SECTION_CAP)
    expect(c.overflows(items.length)).toBe(true)
    c.toggle('a')
    expect(c.visible('a', items)).toHaveLength(items.length)
    c.toggle('a')
    expect(c.visible('a', items)).toHaveLength(SECTION_CAP)
  })

  it('a short section neither caps nor offers a toggle', () => {
    const c = useCappedSections()
    const items = many(3)
    expect(c.visible('a', items)).toHaveLength(3)
    expect(c.overflows(items.length)).toBe(false)
  })

  it('force (search active) lifts every cap and hides the toggle', () => {
    const c = useCappedSections()
    const items = many(SECTION_CAP + 4)
    expect(c.visible('a', items, true)).toHaveLength(items.length)
    expect(c.overflows(items.length, true)).toBe(false)
  })

  it('sections are capped independently', () => {
    const c = useCappedSections()
    c.toggle('a')
    expect(c.visible('a', many(10))).toHaveLength(10)
    expect(c.visible('b', many(10))).toHaveLength(SECTION_CAP)
  })
})
