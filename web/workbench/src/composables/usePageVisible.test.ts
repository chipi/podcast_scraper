/**
 * @vitest-environment happy-dom
 */
import { afterEach, describe, expect, it } from 'vitest'
import { effectScope, nextTick } from 'vue'
import { usePageVisible } from './usePageVisible'

describe('usePageVisible', () => {
  afterEach(() => {
    Object.defineProperty(document, 'hidden', {
      configurable: true,
      writable: true,
      value: false,
    })
  })

  it('tracks visibility after mount', async () => {
    const scope = effectScope()
    let pageVisible!: ReturnType<typeof usePageVisible>['pageVisible']
    scope.run(() => {
      ;({ pageVisible } = usePageVisible())
    })
    await nextTick()
    expect(pageVisible.value).toBe(true)

    Object.defineProperty(document, 'hidden', {
      configurable: true,
      writable: true,
      value: true,
    })
    document.dispatchEvent(new Event('visibilitychange'))
    await nextTick()
    expect(pageVisible.value).toBe(false)

    Object.defineProperty(document, 'hidden', {
      configurable: true,
      writable: true,
      value: false,
    })
    document.dispatchEvent(new Event('visibilitychange'))
    await nextTick()
    expect(pageVisible.value).toBe(true)

    scope.stop()
  })
})
