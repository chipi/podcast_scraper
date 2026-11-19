import { nextTick } from 'vue'

/**
 * Vertical list keyboard nav: ArrowUp/Down, Home/End.
 * Activates the target row (e.g. open Episode panel) and moves focus + scrolls into view.
 */
export function handleVerticalListArrowKeydown(
  e: KeyboardEvent,
  index: number,
  args: {
    itemCount: number
    scrollRoot: HTMLElement | null
    rowSelector: string
    activateIndex: (index: number) => void
  },
): void {
  const { itemCount, scrollRoot, rowSelector, activateIndex } = args
  if (itemCount === 0) {
    return
  }

  if (e.key === 'ArrowDown') {
    if (index >= itemCount - 1) {
      e.preventDefault()
      return
    }
    e.preventDefault()
    const target = index + 1
    activateIndex(target)
    void focusListRowAt(scrollRoot, rowSelector, target)
    return
  }

  if (e.key === 'ArrowUp') {
    if (index <= 0) {
      e.preventDefault()
      return
    }
    e.preventDefault()
    const target = index - 1
    activateIndex(target)
    void focusListRowAt(scrollRoot, rowSelector, target)
    return
  }

  if (e.key === 'Home') {
    e.preventDefault()
    activateIndex(0)
    void focusListRowAt(scrollRoot, rowSelector, 0)
    return
  }

  if (e.key === 'End') {
    e.preventDefault()
    const last = itemCount - 1
    activateIndex(last)
    void focusListRowAt(scrollRoot, rowSelector, last)
    return
  }
}

function focusListRowAt(
  scrollRoot: HTMLElement | null,
  rowSelector: string,
  targetIndex: number,
): void {
  void nextTick(() => {
    const rows = scrollRoot?.querySelectorAll<HTMLElement>(rowSelector)
    const el = rows?.item(targetIndex)
    el?.focus()
    el?.scrollIntoView({ block: 'nearest' })
  })
}
