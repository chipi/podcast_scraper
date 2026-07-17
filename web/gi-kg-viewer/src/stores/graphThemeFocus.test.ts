// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphThemeFocusStore } from './graphThemeFocus'

describe('useGraphThemeFocusStore (graph-v3 tier 7-3)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('starts empty and reports hasFocus/isFocused correctly', () => {
    const s = useGraphThemeFocusStore()
    expect(s.focusedThemeIds.size).toBe(0)
    expect(s.hasFocus()).toBe(false)
    expect(s.isFocused('thc:a')).toBe(false)
  })

  it('setFocus replaces the set (does not merge)', () => {
    const s = useGraphThemeFocusStore()
    s.setFocus(['thc:a', 'thc:b'])
    expect(s.focusedThemeIds.size).toBe(2)
    expect(s.isFocused('thc:a')).toBe(true)
    s.setFocus(['thc:c'])
    expect(s.focusedThemeIds.size).toBe(1)
    expect(s.isFocused('thc:a')).toBe(false)
    expect(s.isFocused('thc:c')).toBe(true)
  })

  it('clearFocus empties the set + is idempotent', () => {
    const s = useGraphThemeFocusStore()
    s.setFocus(['thc:a'])
    s.clearFocus()
    expect(s.focusedThemeIds.size).toBe(0)
    /* Calling clearFocus on an empty set is a no-op — must not
       trigger a redundant reactive update. */
    const before = s.focusedThemeIds
    s.clearFocus()
    expect(s.focusedThemeIds).toBe(before)
  })

  it('hasFocus flips to false only when the set is fully empty', () => {
    const s = useGraphThemeFocusStore()
    s.setFocus(['thc:a', 'thc:b'])
    expect(s.hasFocus()).toBe(true)
    s.setFocus(['thc:a'])
    expect(s.hasFocus()).toBe(true)
    s.setFocus([])
    expect(s.hasFocus()).toBe(false)
  })
})
