import { describe, expect, it } from "vitest"
import { matchesAllWords } from "./textMatch"

describe("matchesAllWords — every word, any order", () => {
  it("matches a single word anywhere, ignoring case", () => {
    expect(matchesAllWords("The Pragmatic Engineer", "engineer")).toBe(true)
  })

  it("matches several words in any order, not only as one phrase", () => {
    expect(matchesAllWords("Notes on sleep and memory", "memory sleep")).toBe(true)
    expect(matchesAllWords("The Pragmatic Engineer", "engineer pragmatic")).toBe(true)
  })

  it("needs ALL the words — one missing is no match", () => {
    expect(matchesAllWords("The Pragmatic Engineer", "pragmatic designer")).toBe(false)
  })

  it("an empty query or empty text matches nothing", () => {
    expect(matchesAllWords("anything", "   ")).toBe(false)
    expect(matchesAllWords(null, "word")).toBe(false)
  })
})
