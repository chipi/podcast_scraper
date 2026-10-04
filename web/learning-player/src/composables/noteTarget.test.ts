import { describe, expect, it } from "vitest"
import { NOTES_ANCHOR, noteRoute } from "./noteTarget"

describe("noteRoute — a note opens on its notes, not the page top (operator 2026-10-04)", () => {
  it.each([
    ["topic", "topic:ai", { name: "topic", params: { id: "topic:ai" } }],
    ["person", "person:jane", { name: "person", params: { id: "person:jane" } }],
    ["show", "f1", { name: "podcast", params: { feedId: "f1" } }],
    ["storyline", "topic:ai", { name: "storyline", params: { id: "topic:ai" } }],
    ["theme", "tc:x", { name: "theme", params: { id: "tc:x" } }],
  ])("a %s note lands on the page's notes section", (target, id, route) => {
    expect(noteRoute(target, id)).toEqual({ ...route, hash: NOTES_ANCHOR })
  })

  it("an episode note opens the episode-notes panel, where its notes live", () => {
    expect(noteRoute("episode", "ep-1")).toEqual({
      name: "player",
      params: { slug: "ep-1" },
      query: { notes: "1" },
    })
  })

  it("a highlight note still lands on its moment", () => {
    const highlights = [{ id: "h1", episode_slug: "ep-1", start_ms: 61500 }] as never
    expect(noteRoute("highlight", "h1", highlights)).toEqual({
      name: "player",
      params: { slug: "ep-1" },
      query: { t: "61" },
    })
  })
})
