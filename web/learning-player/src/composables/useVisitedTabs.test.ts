import { describe, expect, it } from 'vitest'
import { nextTick, ref } from 'vue'
import { useVisitedTabs } from './useVisitedTabs'

describe('useVisitedTabs', () => {
  it('knows only the tab it opened on, then every tab visited, and forgets none', async () => {
    const tab = ref<'a' | 'b' | 'c'>('a')
    const visited = useVisitedTabs(tab)
    expect([visited.has('a'), visited.has('b'), visited.has('c')]).toEqual([true, false, false])
    tab.value = 'b'
    await nextTick()
    tab.value = 'a'
    await nextTick()
    expect([visited.has('a'), visited.has('b'), visited.has('c')]).toEqual([true, true, false])
  })
})
