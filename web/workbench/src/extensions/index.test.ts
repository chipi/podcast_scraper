import { describe, expect, it } from 'vitest'

import { extensionPoint } from './index'

describe('extensionPoint', () => {
  it('lists entries by order, then registration order', () => {
    const p = extensionPoint<{ id: string; order?: number }>('test')
    p.register({ id: 'b', order: 20 })
    p.register({ id: 'a' })
    p.register({ id: 'c', order: 20 })
    p.register({ id: 'd', order: 10 })
    expect(p.list().map((e) => e.id)).toEqual(['a', 'd', 'b', 'c'])
  })

  it('refuses a second entry with the same id', () => {
    const p = extensionPoint<{ id: string }>('test')
    p.register({ id: 'x' })
    expect(() => p.register({ id: 'x' })).toThrow(/test: "x" is already registered/)
  })

  it('reset empties the list', () => {
    const p = extensionPoint<{ id: string }>('test')
    p.register({ id: 'x' })
    p.reset()
    expect(p.list()).toEqual([])
  })
})
