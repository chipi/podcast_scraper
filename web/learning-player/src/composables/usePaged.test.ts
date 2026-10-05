import { describe, expect, it } from "vitest"
import { ref } from "vue"
import { usePaged } from "./usePaged"

describe("usePaged", () => {
  it("shows a page, reveals the next page or the remainder, then folds back", () => {
    const items = ref(Array.from({ length: 12 }, (_, i) => i))
    const p = usePaged(items, 5)
    expect(p.visible.value).toEqual([0, 1, 2, 3, 4])
    expect([p.hidden.value, p.nextCount.value, p.canFold.value]).toEqual([7, 5, false])
    p.more()
    expect(p.visible.value).toHaveLength(10)
    expect([p.hidden.value, p.nextCount.value]).toEqual([2, 2])
    p.more()
    expect(p.visible.value).toHaveLength(12)
    expect([p.hidden.value, p.canFold.value]).toEqual([0, true])
    p.reset()
    expect(p.visible.value).toHaveLength(5)
  })

  it("has nothing to page or fold when the list fits one page", () => {
    const p = usePaged(ref([1, 2, 3]), 5)
    expect([p.visible.value.length, p.hidden.value, p.canFold.value]).toEqual([3, 0, false])
  })
})
