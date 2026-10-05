import { describe, expect, it } from "vitest"
import { newestFirst } from "./newestFirst"

describe("newestFirst", () => {
  it("orders by created_at, newest first", () => {
    expect(newestFirst([{ created_at: 1 }, { created_at: 3 }, { created_at: 2 }]).map((x) => x.created_at)).toEqual([3, 2, 1])
  })

  it("breaks a same-second tie by position: the later item is the newer", () => {
    const items = [
      { id: "a", created_at: 5 },
      { id: "b", created_at: 7 },
      { id: "c", created_at: 7 },
      { id: "d", created_at: 7 },
    ]
    expect(newestFirst(items).map((x) => x.id)).toEqual(["d", "c", "b", "a"])
  })

  it("does not mutate its input", () => {
    const items = [{ created_at: 1 }, { created_at: 2 }]
    newestFirst(items)
    expect(items.map((x) => x.created_at)).toEqual([1, 2])
  })
})
