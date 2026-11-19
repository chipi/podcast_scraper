import { describe, expect, it, vi } from 'vitest'
import { handleVerticalListArrowKeydown } from './listRowArrowNav'

describe('handleVerticalListArrowKeydown', () => {
  const base = {
    itemCount: 3,
    scrollRoot: null as HTMLElement | null,
    rowSelector: '[data-x]',
    activateIndex: vi.fn(),
  }

  it('calls activateIndex for ArrowDown when not at end', () => {
    base.activateIndex.mockClear()
    const e = { key: 'ArrowDown', preventDefault: vi.fn() } as unknown as KeyboardEvent
    handleVerticalListArrowKeydown(e, 0, base)
    expect(e.preventDefault).toHaveBeenCalled()
    expect(base.activateIndex).toHaveBeenCalledWith(1)
  })

  it('does not activate past last row on ArrowDown', () => {
    base.activateIndex.mockClear()
    const e = { key: 'ArrowDown', preventDefault: vi.fn() } as unknown as KeyboardEvent
    handleVerticalListArrowKeydown(e, 2, base)
    expect(e.preventDefault).toHaveBeenCalled()
    expect(base.activateIndex).not.toHaveBeenCalled()
  })

  it('calls activateIndex for ArrowUp when not at start', () => {
    base.activateIndex.mockClear()
    const e = { key: 'ArrowUp', preventDefault: vi.fn() } as unknown as KeyboardEvent
    handleVerticalListArrowKeydown(e, 1, base)
    expect(e.preventDefault).toHaveBeenCalled()
    expect(base.activateIndex).toHaveBeenCalledWith(0)
  })

  it('calls activateIndex 0 for Home', () => {
    base.activateIndex.mockClear()
    const e = { key: 'Home', preventDefault: vi.fn() } as unknown as KeyboardEvent
    handleVerticalListArrowKeydown(e, 2, base)
    expect(e.preventDefault).toHaveBeenCalled()
    expect(base.activateIndex).toHaveBeenCalledWith(0)
  })

  it('calls activateIndex last for End', () => {
    base.activateIndex.mockClear()
    const e = { key: 'End', preventDefault: vi.fn() } as unknown as KeyboardEvent
    handleVerticalListArrowKeydown(e, 0, base)
    expect(e.preventDefault).toHaveBeenCalled()
    expect(base.activateIndex).toHaveBeenCalledWith(2)
  })

  it('ignores other keys', () => {
    base.activateIndex.mockClear()
    const e = { key: 'a', preventDefault: vi.fn() } as unknown as KeyboardEvent
    handleVerticalListArrowKeydown(e, 1, base)
    expect(e.preventDefault).not.toHaveBeenCalled()
    expect(base.activateIndex).not.toHaveBeenCalled()
  })
})
