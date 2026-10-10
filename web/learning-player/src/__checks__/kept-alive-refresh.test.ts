import { describe, expect, it } from "vitest"

/**
 * A component that reads PER-USER data straight from the API must re-read it on a return to its
 * tab, or say here why it does not need to (2026-10-09).
 *
 * Home, Search, Library, Profile, Catalog and Discover are KEPT ALIVE (App.vue KEEP_ALIVE_TABS), so
 * `onMounted` runs once per session. A per-user read done only there showed the data as of the first
 * visit, whatever happened on other tabs since — Your Week, Library › Revisit, Discover's "Your
 * trends" and Home's revisit line all shipped that way. The fix is `onActivated` (after the tab was
 * LEFT once — see src/test/keptAlive.ts) or reading a store that is told about writes.
 */
const sources = import.meta.glob("../**/*.vue", { query: "?raw", import: "default", eager: true }) as Record<
  string,
  string
>

/** API reads whose answer is the signed-in user's own state. */
const USER_GETTERS = [
  "getPlaybackList", "getPlayback", "getYourWeek", "getResurfacingPage", "getResurfacing", "getCollections",
  "getCollectionPage", "getFavoritesPage", "getFavorites", "getFavoriteRefs", "getHighlightsPage",
  "getNotesPage", "getMyStats", "getRecap", "getComms", "getUserInterests", "getCompleted", "getQueue",
  "getLibrary", "getNotifications", "getRecommended", "getMcpTokens", "getMcpConnections", "getMcpConfig",
  "getCollectionsContaining",
]

/** Files that read per-user data without `onActivated`, and why that is right. */
const EXEMPT: Record<string, string> = {
  '../views/DiscoverV2View.vue': 'A preview page outside KEEP_ALIVE_TABS: it remounts, and so re-reads, on every visit.',
  "../App.vue": "the shell, not a tab: it is never deactivated",
  "../components/InterestsPicker.vue": "a modal, mounted each time it opens",
  "../components/RecentlyPlayedList.vue": "inside QueueView, which is not kept alive (remounts per visit)",
  "../components/ConnectedAgents.vue": "inside Settings, which is not kept alive",
  "../components/AddToCollectionButton.vue": "reads when its sheet opens, and writes through the collections store",
  "../views/CollectionsView.vue": "renders the collections store, which every board write updates",
  "../views/PlayerView.vue": "a route that remounts per episode; not kept alive",
}

describe("kept-alive refresh", () => {
  const offenders = Object.entries(sources)
    .filter(([path]) => !path.includes("/__checks__/") && !path.endsWith(".test.vue"))
    .map(([path, src]) => {
      const script = src.split("<template")[0]
      const reads = USER_GETTERS.filter((g) => new RegExp(`\\b${g}\\(`).test(script))
      return { path, reads, activates: /\bonActivated\(/.test(script) }
    })
    .filter((f) => f.reads.length && !f.activates)

  it("finds per-user reads at all (the scan has not gone blind)", () => {
    const all = Object.values(sources).filter((s) => /\bgetYourWeek\(/.test(s))
    expect(all.length).toBeGreaterThan(0)
  })

  it("every per-user read without onActivated is exempt, with a reason", () => {
    const unexplained = offenders.filter((f) => !EXEMPT[f.path]).map((f) => `${f.path} (${f.reads.join(", ")})`)
    expect(unexplained, "re-read on onActivated (after a deactivation), read a store, or add to EXEMPT with why").toEqual([])
  })

  it("no stale exemptions", () => {
    const live = new Set(offenders.map((f) => f.path))
    expect(Object.keys(EXEMPT).filter((p) => !live.has(p))).toEqual([])
  })
})
