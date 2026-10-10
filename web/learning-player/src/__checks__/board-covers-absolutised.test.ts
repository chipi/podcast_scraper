import { describe, expect, it } from "vitest"
import apiSrc from "../services/api.ts?raw"

/**
 * Every API call that returns a BOARD absolutises its `cover_url` (2026-10-09).
 *
 * The cover arrives relative (`/api/app/artwork?…`), which on the device resolves into the app
 * bundle and paints a broken image. The list calls went through `withAbsoluteCovers`; create /
 * add-item / remove-item / the board read did not — harmless until their answers reached the shared
 * boards store, then a board just added to showed a broken cover on Home and none in Library.
 *
 * A text rule, the shape of artwork-absolutised.test.ts: a function in api.ts whose declared return
 * mentions `Collection` must go through `withAbsoluteCover` / `withAbsoluteCovers` (or delegate to
 * one that does — listed below).
 */
// None since 2026-10-10: the one delegate, pageCollectionLocally, left api.ts with the pre-1.0.3
// paging fallback (it is the tests' fake server now, test/localPagers.ts).
const DELEGATES: Record<string, string> = {}

function boardReturningFunctions(src: string): { name: string; body: string }[] {
  const re = /export (?:async )?function (\w+)\([^]*?\)\s*:\s*([^{]+)\{/g
  const out: { name: string; body: string }[] = []
  const matches = [...src.matchAll(re)]
  matches.forEach((m, i) => {
    if (!/\bCollection(Detail)?\b/.test(m[2])) return
    const start = m.index ?? 0
    const end = i + 1 < matches.length ? (matches[i + 1].index ?? src.length) : src.length
    out.push({ name: m[1], body: src.slice(start, end) })
  })
  return out
}

describe("board covers are absolutised at the API boundary", () => {
  const fns = boardReturningFunctions(apiSrc)

  it("finds the board-returning calls (the scan has not gone blind)", () => {
    const names = fns.map((f) => f.name)
    for (const n of ["getCollections", "createCollection", "addToCollection", "getCollectionPage"]) {
      expect(names).toContain(n)
    }
  })

  it("each one absolutises the cover, or is a listed delegate", () => {
    const missing = fns
      .filter((f) => !DELEGATES[f.name] && !/withAbsoluteCovers?\(/.test(f.body))
      .map((f) => f.name)
    expect(missing, "wrap the returned board(s) in withAbsoluteCover / withAbsoluteCovers").toEqual([])
  })
})
