import { existsSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import apiSrc from "../services/api.ts?raw"
import { CONSISTENCY_MATRIX } from "./consistencyMatrix"

/**
 * Every write in `services/api.ts` has a row in the consistency matrix (2026-10-09).
 *
 * A WRITE is an exported function whose body sends POST / PUT / PATCH / DELETE. The ten stale-surface
 * bugs of 2026-10-09 came from writes nobody had mapped to the surfaces that must show them; this
 * makes mapping one part of adding one. See consistencyMatrix.ts.
 */
function writeFunctions(src: string): string[] {
  const out: string[] = []
  const re = /export (?:async )?function (\w+)\(/g
  const starts = [...src.matchAll(re)].map((m) => ({ name: m[1], at: m.index ?? 0 }))
  starts.forEach((f, i) => {
    const body = src.slice(f.at, i + 1 < starts.length ? starts[i + 1].at : src.length)
    if (/method:\s*["'](POST|PUT|PATCH|DELETE)["']/.test(body)) out.push(f.name)
  })
  return out
}

describe("consistency matrix", () => {
  const writes = writeFunctions(apiSrc)
  const rows = new Map(CONSISTENCY_MATRIX.map((r) => [r.write, r]))

  it("finds the writes (the parser has not silently gone blind)", () => {
    expect(writes.length).toBeGreaterThan(30)
    expect(writes).toContain("addToCollection")
  })

  it("every write in api.ts has a row", () => {
    const missing = writes.filter((w) => !rows.has(w))
    expect(missing, "add a row to src/__checks__/consistencyMatrix.ts for each").toEqual([])
  })

  it("every row names a write that still exists", () => {
    const stale = CONSISTENCY_MATRIX.map((r) => r.write).filter((w) => !writes.includes(w))
    expect(stale, "remove (or rename) these rows").toEqual([])
  })

  it("every proof file exists, and a row with no other reader says why", () => {
    for (const r of CONSISTENCY_MATRIX) {
      for (const p of r.proof) {
        expect(existsSync(resolve(process.cwd(), p)), `${r.write}: no such proof file ${p}`).toBe(true)
      }
      if (!r.readers.length) expect(r.note, `${r.write}: no readers and no note`).toBeTruthy()
      if (r.readers.length) expect(r.proof.length, `${r.write}: readers but no proof`).toBeGreaterThan(0)
    }
  })
})
