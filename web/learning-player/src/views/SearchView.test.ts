import { flushPromises, mount } from "@vue/test-utils"
import { createPinia, setActivePinia } from "pinia"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import { useAuthStore } from "../stores/auth"
import { clearCached } from "../services/contentCache"
import { useSavedQueriesStore } from "../stores/savedQueries"
import SearchView from "./SearchView.vue"

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/search", name: "search", component: SearchView },
      { path: "/episode/:slug", name: "player", component: { template: "<div/>" } },
      // Gated controls route here when signed out (#1590).
      { path: "/login", name: "login", component: { template: "<div/>" } },
      // #1261-9: EntityCardBody now renders an "Open in page" RouterLink to
      // these routes when open in overlay mode — the router must resolve
      // them or the mount silently drops the tree with a runtime error.
      { path: "/topic/:id", name: "topic", component: { template: "<div/>" }, props: true },
      { path: "/person/:id", name: "person", component: { template: "<div/>" }, props: true },
      // Storyline results link here. Without the route, `router.resolve` THROWS mid-render and
      // vue-router drops the subtree — which is why a missing route showed up as "the storylines
      // section did not render" AND corrupted later tests in this file, rather than as a routing
      // error (operator 2026-09-17). Same failure mode as an EpisodeCard with no feed_id.
      { path: "/storyline/:id", name: "storyline", component: { template: "<div/>" }, props: true },
    ],
  })
}

// Default: no entity match — search tests assert passage behaviour without an entity card.
beforeEach(async () => {
  // The capture store PERSISTS notes and highlights (`writeCached('captures', …)`) and recovers from
  // that cache when a fetch yields nothing — so notes seeded by one test were read back by the next.
  // It only surfaced once two tests here began seeding notes: the recall-empty case then rendered
  // another test's note list in place of its empty state.
  //
  // Clearing the CACHE is the whole fix. Do NOT add `setActivePinia` here: several tests below build
  // their own pinia, pass it to `mount`, and then read the store from the test body — which resolves
  // against whatever pinia is active. Installing one here made those two different instances, so a
  // store the component had written looked empty to the test.
  await clearCached(["captures"])
  vi.spyOn(api, "resolveEntity").mockResolvedValue({ query: "", entity: null })
  // Default: no storylines. Unmocked, the onMounted fetch rejects into an unhandled rejection that
  // Vitest reports across the whole run.
  vi.spyOn(api, "getStorylines").mockResolvedValue([])
  // Default: an EMPTY capture. Unmocked, `ensureLoaded()` fails against no server and the store then
  // RECOVERS FROM CACHE — so notes seeded by one test were served to every later test whose fetch
  // failed, which is most of them. A succeeding empty fetch overwrites instead, and tests that want
  // notes override these two. (Clearing the cache is not enough: it is namespaced per account, and
  // the namespace at write time is not the one a bare clear touches.)
  vi.spyOn(api, "getNotes").mockResolvedValue([])
  vi.spyOn(api, "getHighlights").mockResolvedValue([])
})
// Let a test's in-flight work SETTLE before its mocks are pulled. This view fires several unawaited
// fetches from `onMounted` (captures, storylines) plus the search itself; restoring mocks while one
// is pending let it resolve during the NEXT test and write into that test's stores. It showed up as
// three unrelated cases failing together the moment new tests here began seeding notes and
// storylines — each passed alone (operator 2026-09-17).
afterEach(async () => {
  // UNMOUNT first. This view fires unawaited fetches from `onMounted` (captures, storylines) and
  // keeps reacting to them; a wrapper left mounted resolved those during the NEXT test and wrote
  // into that test's stores. It surfaced as three unrelated cases failing together the moment new
  // tests here began seeding notes — each passed alone (operator 2026-09-17).
  mounted?.unmount()
  mounted = null
  await flushPromises()
  vi.restoreAllMocks()
})

// The wrapper from the most recent `mountAt`, so afterEach can tear it down.
let mounted: { unmount: () => void } | null = null

async function mountAt(q: string) {
  const router = makeRouter()
  router.push({ name: "search", query: { q } })
  await router.isReady()
  const w = mount(SearchView, {
    global: { plugins: [i18n, router, createPinia()], stubs: { teleport: true } },
  })
  mounted = w
  await flushPromises()
  return { w, router }
}

describe("SearchView", () => {
  it("renders grounded passages with source + jump, and jumps with ?t=", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "memory",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 0.9,
          text: "A grounded passage about memory.",
          source_tier: "segment",
          metadata: { episode_slug: "show-x", episode_title: "Ep X", podcast_title: "Show" },
          lifted: { quote: { timestamp_start_ms: 20000 } },
        },
      ],
    })
    const { w, router } = await mountAt("memory")
    expect(w.text()).toContain("A grounded passage about memory.")
    expect(w.text()).toContain("Ep X")
    const push = vi.spyOn(router, "push")
    await w
      .findAll("button")
      .find((b) => b.text().includes("0:20"))!
      .trigger("click")
    expect(push).toHaveBeenCalledWith({
      name: "player",
      params: { slug: "show-x" },
      query: { t: "20" },
    })
    // #2 — each episode result carries per-episode quick actions (favorite + queue), like a Library
    // row. The shared `episode-actions` testid, since Search renders the shared EpisodeCard now.
    const actions = w.get('[data-testid="episode-actions"]')
    expect(actions.findAll("button").length).toBeGreaterThanOrEqual(2)
  })

  it("persists results across a tab switch — navigate away and back does not clear them", async () => {
    // Regression (operator 2026-09-13): SearchView is kept-alive, but its `route.query.q` watcher
    // re-ran an empty search when navigating away (q → undefined) and when the bottom-nav Search
    // link returned with no `?q`, wiping the results. The fix ignores the watcher off-route and
    // restores the URL from the live query on return.
    const search = vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "memory",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 0.9,
          text: "A grounded passage about memory.",
          source_tier: "segment",
          metadata: { episode_slug: "show-x", episode_title: "Ep X", podcast_title: "Show" },
        },
      ],
    })
    const { w, router } = await mountAt("memory")
    expect(w.text()).toContain("A grounded passage about memory.")
    const callsAfterSearch = search.mock.calls.length

    // Away to another tab, then back via the nav link (which carries no ?q).
    await router.push({ name: "player", params: { slug: "x" } })
    await flushPromises()
    await router.push({ name: "search" })
    await flushPromises()

    // Results still on screen, and no redundant re-fetch of the (empty) query.
    expect(w.text()).toContain("A grounded passage about memory.")
    expect(search.mock.calls.length).toBe(callsAfterSearch)
  })

  it("shows a graceful error with a retry on failure (F1.4)", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "x", error: "no_index", results: [] })
    const { w } = await mountAt("x")
    // The error path now uses the shared SectionStatus (skeleton/error/retry), so a failed search
    // offers a retry instead of a dead-end line.
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
  })

  it("retry after an error actually re-fetches the SAME term (does not dead-return)", async () => {
    // Regression: the run-dedup latched `lastRunSig` even on error, so re-running the identical
    // term+scope early-returned and the retry button did nothing. On error the sig must NOT latch.
    const spy = vi
      .spyOn(api, "searchCorpus")
      .mockResolvedValueOnce({ query: "x", error: "no_index", results: [] })
    const { w } = await mountAt("x")
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    expect(spy).toHaveBeenCalledTimes(1)

    spy.mockResolvedValueOnce({
      query: "x",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 0.9,
          text: "Now it works.",
          source_tier: "segment",
          metadata: { episode_slug: "show-x", episode_title: "Ep X", podcast_title: "Show" },
        },
      ],
    })
    await w.get('[data-testid="section-retry"]').trigger("click")
    await flushPromises()
    expect(spy).toHaveBeenCalledTimes(2)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(false)
    expect(w.text()).toContain("Now it works.")
  })

  it("a slower OLDER response never overwrites the newer query's results (generation guard)", async () => {
    const resolvers: Array<(v: unknown) => void> = []
    vi.spyOn(api, "searchCorpus").mockImplementation(
      () => new Promise((resolve) => resolvers.push(resolve as (v: unknown) => void))
    )
    const { w } = await mountAt("aaa") // run #1 fires, left pending
    await w.get("#search-q").setValue("bbb")
    await w.get("form").trigger("submit")
    await flushPromises() // run #2 fires, left pending
    expect(resolvers.length).toBe(2)

    const hit = (text: string, slug: string) => ({
      query: "",
      error: null,
      results: [
        {
          doc_id: slug,
          score: 1,
          text,
          source_tier: "segment",
          metadata: { episode_slug: slug, episode_title: "E", podcast_title: "S" },
        },
      ],
    })
    // Newer query (bbb) resolves first, then the OLDER (aaa) lands late — the guard must drop it.
    resolvers[1](hit("BBB result", "b"))
    await flushPromises()
    resolvers[0](hit("AAA result", "a"))
    await flushPromises()

    expect(w.text()).toContain("BBB result")
    expect(w.text()).not.toContain("AAA result")
  })

  it("shows no-results when empty without error", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "x", error: null, results: [] })
    const { w } = await mountAt("x")
    expect(w.text()).toContain("No matches found.")
  })

  it("surfaces an entity card above passages and opens the full card on tap (3.4)", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "jane", error: null, results: [] })
    vi.spyOn(api, "resolveEntity").mockResolvedValue({
      query: "jane",
      entity: { id: "person:jane-doe", kind: "person", label: "Jane Doe" },
    })
    const getPerson = vi.spyOn(api, "getPersonCard").mockResolvedValue({
      id: "person:jane-doe",
      label: "Jane Doe",
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
    })
    const { w } = await mountAt("jane")
    // Entity hit card shows (Person kicker + name); the no-results line is suppressed.
    expect(w.text()).toContain("Person")
    expect(w.text()).toContain("Jane Doe")
    expect(w.text()).not.toContain("No matches found.")
    // Tapping it opens the full EntityCard overlay.
    await w
      .findAll("button")
      .find((b) => b.text().includes("View"))!
      .trigger("click")
    await flushPromises()
    expect(getPerson).toHaveBeenCalledWith("person:jane-doe")
    expect(w.find('[role="dialog"]').exists()).toBe(true)
  })

  it("surfaces a STORYLINE from the resolver, in its own section", async () => {
    // Storylines are INDEXED server-side, in the same resolver as people and topics (operator
    // 2026-09-17: "I don't want a client-side solution, I asked for them to be indexed"). The first
    // pass matched labels in the client; that could not rank, could not see past the list
    // endpoint's 50-item cap, and left every other consumer of the resolver blind to storylines.
    //
    // People / topics / storylines each get their OWN section, so a person and a storyline never
    // look like the same kind of result.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "risk", error: null, results: [] })
    vi.spyOn(api, "resolveEntity").mockResolvedValue({
      query: "risk",
      entity: { id: "thc:managing-risk", kind: "storyline", label: "Managing risk across domains" },
    })
    const { w } = await mountAt("risk")
    await flushPromises()
    expect(w.find('[data-testid="search-section-storylines"]').exists()).toBe(true)
    expect(w.find('[data-testid="search-section-people"]').exists()).toBe(false)
    expect(w.find('[data-testid="search-section-topics"]').exists()).toBe(false)
    expect(w.text()).toContain("Managing risk across domains")
    expect(w.text()).not.toContain("No matches found.")
  })

  it("links a storyline straight to its page, and never as an entity card", async () => {
    // A storyline is a theme cluster with a page of its own; there is no EntityCard for one. It
    // renders as a plain link rather than the entity card's "View" button, so there is nothing to
    // intercept and nothing to open as an overlay.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "risk", error: null, results: [] })
    vi.spyOn(api, "resolveEntity").mockResolvedValue({
      query: "risk",
      // The resolver answers with the ANCHOR topic id, not `thc:…` — there is no storyline
      // endpoint, so a `thc:` id is not routable (review, 2026-09-17).
      entity: {
        id: "topic:risk-management",
        kind: "storyline",
        label: "Managing risk across domains",
      },
    })
    const { w } = await mountAt("risk")
    await flushPromises()
    const link = w
      .findAllComponents({ name: "RouterLink" })
      .find((l) => JSON.stringify(l.props("to") ?? {}).includes("topic:risk-management"))
    expect(link, "no link to the storyline page").toBeTruthy()
    expect(link!.props("to")).toEqual({
      name: "storyline",
      params: { id: "topic:risk-management" },
    })
    expect(w.find('[role="dialog"]').exists(), "opened an entity card instead").toBe(false)
  })

  it("does NOT group a storyline hit as an episode", async () => {
    // A storyline row is corpus-level and carries no `episode_id`, so the episode grouping would key
    // it as `doc:<id>` and render a card titled "Not found" — an episode that does not exist
    // (operator 2026-09-17). It belongs in the Storylines section, from the INDEX, which is what
    // makes a partial or member-topic query reach it.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "risk",
      error: null,
      results: [
        {
          doc_id: "storyline:thc:managing-risk",
          score: 1,
          text: "Managing risk across domains risk management systems thinking",
          source_tier: "aux",
          // The shape the SERVER actually returns: `storyline_label` / `storyline_size` /
          // `anchor_topic_id` come from the query-time join, because the indexed row cannot carry
          // them (no such columns, and the read path rebuilds metadata from a fixed field list).
          metadata: {
            doc_type: "storyline",
            source_id: "thc:managing-risk",
            storyline_label: "Managing risk across domains",
            storyline_size: 4,
            anchor_topic_id: "topic:risk-management",
            episode_id: null,
          },
        },
      ],
    })
    const { w } = await mountAt("risk")
    await flushPromises()
    expect(w.find('[data-testid="search-storylines"]').exists()).toBe(true)
    expect(w.text()).toContain("Managing risk across domains")
    expect(w.find('[data-testid="episode-card"]').exists(), "storyline became an episode").toBe(
      false
    )
    expect(w.text(), "rendered a nonexistent episode").not.toContain("Not found")
    expect(w.text()).not.toContain("No matches found.")
  })

  it("skips a storyline hit that arrives without its joined fields", async () => {
    // Before the query-time join existed, a hit carried only `source_id`, and the section rendered
    // the raw `thc:` slug as a title and linked somewhere that 404s. Unusable is worse than absent,
    // so such a hit is dropped rather than displayed (review, 2026-09-17).
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "risk",
      error: null,
      results: [
        {
          doc_id: "storyline:thc:managing-risk",
          score: 1,
          text: "Managing risk across domains",
          source_tier: "aux",
          metadata: { doc_type: "storyline", source_id: "thc:managing-risk", episode_id: null },
        },
      ],
    })
    const { w } = await mountAt("risk")
    await flushPromises()
    expect(w.find('[data-testid="search-storylines"]').exists()).toBe(false)
    expect(w.text(), "rendered a raw thc: slug").not.toContain("thc:managing-risk")
  })

  it("filters which result KINDS show, and All clears it", async () => {
    // Same multi-select chips as Library's, none selected = all (operator 2026-09-17). Only kinds
    // actually present get a chip, so the filter cannot empty the page by itself.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "sleep",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 1,
          text: "a passage about sleep",
          source_tier: "transcript",
          metadata: {
            episode_slug: "ep-a",
            episode_title: "An Episode About Sleep",
            podcast_title: "Show",
            publish_date: "2026-01-01T00:00:00",
          },
        },
      ],
    })
    vi.spyOn(api, "getHighlights").mockResolvedValue([])
    vi.spyOn(api, "getNotes").mockResolvedValue([
      {
        id: "n1",
        target: "episode",
        target_id: "ep-a",
        text: "my own note about sleep",
        created_at: 1,
        updated_at: 1,
      },
    ])
    const { w } = await mountAt("sleep")
    await flushPromises()

    // Both kinds present → both chips, plus All.
    expect(w.find('[data-testid="search-type-filter"]').exists()).toBe(true)
    expect(w.text()).toContain("my own note about sleep")
    expect(w.text()).toContain("An Episode About Sleep")

    // Notes only: the episode block goes, the note stays.
    await w.find('[data-testid="search-type-notes"]').trigger("click")
    await flushPromises()
    expect(w.text()).toContain("my own note about sleep")
    expect(w.find('[data-testid="episode-card"]').exists(), "episodes survived a notes-only filter").toBe(
      false
    )

    // All restores everything.
    await w.find('[data-testid="search-type-all"]').trigger("click")
    await flushPromises()
    expect(w.find('[data-testid="episode-card"]').exists()).toBe(true)
  })

  it("names what each note is attached to, not just its kind", async () => {
    // A note row showed only "EPISODE" + Open, so a list read as "episode, episode, topic" with no
    // way to tell WHICH without following every link (operator 2026-09-17). Episode titles come from
    // the hits already on screen; entity ids carry their own label and need no lookup.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "sleep",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 1,
          text: "a passage",
          source_tier: "transcript",
          metadata: {
            episode_slug: "ep-a",
            episode_title: "An Episode About Sleep",
            podcast_title: "Show",
            publish_date: "2026-01-01T00:00:00",
          },
        },
      ],
    })
    vi.spyOn(api, "getHighlights").mockResolvedValue([])
    vi.spyOn(api, "getNotes").mockResolvedValue([
      { id: "n1", target: "episode", target_id: "ep-a", text: "note one about sleep", created_at: 2, updated_at: 2 },
      {
        id: "n2",
        target: "topic",
        target_id: "topic:deep-sleep",
        text: "note two about sleep",
        created_at: 1,
        updated_at: 1,
      },
    ])
    const { w } = await mountAt("sleep")
    await flushPromises()
    const targets = w.findAll('[data-testid="search-note-target"]').map((n) => n.text())
    expect(targets, "the episode note has no title").toContain("An Episode About Sleep")
    expect(targets, "the topic note was not de-slugged").toContain("deep sleep")
  })

  it("does NOT surface an organization as a result, even when the resolver returns one", async () => {
    // Organisations stay indexed and keep doing their upstream work — resolution, graph edges, the
    // entity pages that link them. Search just declines to offer one as a destination, where an org
    // card is a dead end beside the episodes and people a listener came for (operator 2026-09-17).
    // Filtered at render, so the resolver is still CALLED — this asserts the call happens and the
    // card does not.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "acme", error: null, results: [] })
    const resolve = vi.spyOn(api, "resolveEntity").mockResolvedValue({
      query: "acme",
      entity: { id: "org:acme", kind: "organization", label: "Acme Corp" },
    })
    const { w } = await mountAt("acme")
    expect(resolve, "the resolver should still be asked").toHaveBeenCalledWith("acme")
    expect(w.text(), "an organization was offered as a result").not.toContain("Acme Corp")
    // With no entity and no passages it is genuinely a no-results search, and must say so rather
    // than sitting silently empty.
    expect(w.text()).toContain("No matches found.")
  })

  it("shows the Recall scope toggle signed out, as a sign-in teaser (#1590)", async () => {
    // This test previously asserted the toggle was HIDDEN. Searching your own corpus is a
    // differentiator neither Spotify nor Apple Podcasts has, and hiding it hid it from exactly the
    // visitors deciding whether an account is worth making. "all" still works; "mine" defers.
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "x", error: null, results: [] })
    const { w } = await mountAt("x")
    // The scope switcher is now a single compact toggle in the search row (not a 2-radio segment).
    const toggle = w.find('[data-testid="search-scope"]')
    expect(toggle.exists()).toBe(true)
    // Signed out it keeps its gated accessible name — "sign in to search yours".
    expect(toggle.attributes("aria-label")).toBe("Sign in to search what you've heard")
  })

  it("shows a teaching zero-state with tappable examples before the first search, then runs one", async () => {
    const search = vi
      .spyOn(api, "searchCorpus")
      .mockResolvedValue({ query: "x", error: null, results: [] })
    const { w } = await mountAt("") // no query yet → zero state
    const zero = w.find('[data-testid="search-zero-state"]')
    expect(zero.exists()).toBe(true)
    const chips = zero.findAll("button")
    expect(chips.length).toBeGreaterThanOrEqual(3)
    // Tapping an example runs that search and dismisses the zero state.
    await chips[0].trigger("click")
    await flushPromises()
    expect(search).toHaveBeenCalled()
    expect(w.find('[data-testid="search-zero-state"]').exists()).toBe(false)
  })

  it('signed-out: choosing "My corpus" routes to sign-in instead of searching (#1590)', async () => {
    const search = vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "x",
      error: null,
      results: [],
    })
    const { w, router } = await mountAt("x")
    search.mockClear()

    await w.get('[data-testid="search-scope"]').trigger("click")
    await flushPromises()

    expect(search).not.toHaveBeenCalled()
    expect(router.currentRoute.value.name).toBe("login")
  })

  it("signed-in: My corpus scope searches scope=mine and shows a recall-specific empty message", async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().user = { user_id: "u1", email: "a@b.c", name: "A" }
    const search = vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "sleep",
      error: null,
      results: [],
    })
    const router = makeRouter()
    router.push({ name: "search", query: { q: "sleep" } })
    await router.isReady()
    const w = mount(SearchView, {
      global: { plugins: [i18n, router, pinia], stubs: { teleport: true } },
    })
    await flushPromises()
    // toggle is visible; default scope=all sent no 'mine'
    expect(w.find('[data-testid="search-scope"]').exists()).toBe(true)
    // 4th positional arg is enrich_results=true (#1261-2): the listener always asks the
    // server to decorate hits with related_topics so the "Also about:" chip row can render.
    expect(search).toHaveBeenLastCalledWith("sleep", 12, "all", true)
    // toggle to My listening → searches scope=mine + recall-empty copy
    await w.get('[data-testid="search-scope"]').trigger("click")
    await flushPromises()
    expect(search).toHaveBeenLastCalledWith("sleep", 12, "mine", true)
    expect(w.text()).toContain("Nothing in your listening on this yet")
  })

  // #1261-2: enriched related-topic chips above episode groups
  it('renders "Also about:" chips from server-decorated hits and opens the topic card on tap', async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "ai",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 0.9,
          text: "A grounded passage about AI.",
          source_tier: "insight",
          metadata: {
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
            query_enrichments: {
              related_topics: [
                { topic_id: "topic:ml", topic_label: "Machine Learning", similarity: 0.91 },
                { topic_id: "topic:safety", topic_label: "AI Safety", similarity: 0.83 },
              ],
            },
          },
        },
      ],
    })
    const getTopic = vi.spyOn(api, "getTopicCard").mockResolvedValue({
      id: "topic:ml",
      label: "Machine Learning",
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
    const { w } = await mountAt("ai")
    const chipRow = w.get('[data-testid="related-topic-chips"]')
    expect(chipRow.text()).toContain("Machine Learning")
    expect(chipRow.text()).toContain("AI Safety")
    // Score-desc: ML (0.91) sorts ahead of Safety (0.83).
    const chipButtons = chipRow.findAll("button")
    expect(chipButtons[0].text()).toBe("Machine Learning")
    await chipButtons[0].trigger("click")
    await flushPromises()
    expect(getTopic).toHaveBeenCalledWith("topic:ml")
    expect(w.find('[role="dialog"]').exists()).toBe(true)
  })

  it("hides the chip row entirely when no hit carries related_topics decoration", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "x",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 0.9,
          text: "t",
          source_tier: "insight",
          metadata: { episode_slug: "show-x", episode_title: "Ep X", podcast_title: "Show" },
        },
      ],
    })
    const { w } = await mountAt("x")
    expect(w.find('[data-testid="related-topic-chips"]').exists()).toBe(false)
  })

  // #1261-3: multiple foldable hits on one episode collapse into one summary row
  it('folds N transcript hits per episode into a single "Transcript · N matches" row that expands on tap', async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "ai",
      error: null,
      results: [
        {
          doc_id: "t1",
          score: 0.9,
          text: "First matching chunk.",
          source_tier: "segment",
          metadata: {
            doc_type: "transcript",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
        {
          doc_id: "t2",
          score: 0.7,
          text: "Second matching chunk.",
          source_tier: "segment",
          metadata: {
            doc_type: "transcript",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
        {
          doc_id: "t3",
          score: 0.6,
          text: "Third matching chunk.",
          source_tier: "segment",
          metadata: {
            doc_type: "transcript",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
      ],
    })
    const { w } = await mountAt("ai")
    const clusterRows = w.findAll('[data-testid="folded-cluster-row"]')
    expect(clusterRows).toHaveLength(1)
    // Collapsed state — only the summary row is present, no excerpts yet.
    expect(w.text()).toContain("3 matches")
    expect(w.text()).not.toContain("First matching chunk.")
    // Expand.
    await clusterRows[0].trigger("click")
    expect(w.text()).toContain("First matching chunk.")
    expect(w.text()).toContain("Second matching chunk.")
    expect(w.text()).toContain("Third matching chunk.")
    // Collapse again.
    await clusterRows[0].trigger("click")
    expect(w.text()).not.toContain("First matching chunk.")
  })

  // #1261-5: matched-field kicker on the episode-group header
  it('renders "Matched: Title · Summary ×2 · Transcript" chips on the episode header', async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "x",
      error: null,
      results: [
        {
          doc_id: "t1",
          score: 0.9,
          text: "Title match.",
          source_tier: "segment",
          metadata: {
            doc_type: "episode_title",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
        {
          doc_id: "s1",
          score: 0.8,
          text: "Summary 1.",
          source_tier: "segment",
          metadata: {
            doc_type: "summary_short",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
        {
          doc_id: "s2",
          score: 0.7,
          text: "Summary 2.",
          source_tier: "segment",
          metadata: {
            doc_type: "summary_short",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
        {
          doc_id: "tr1",
          score: 0.6,
          text: "Transcript match.",
          source_tier: "segment",
          metadata: {
            doc_type: "transcript",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
      ],
    })
    const { w } = await mountAt("x")
    const chips = w.get('[data-testid="matched-fields"]')
    expect(chips.text()).toContain("Matched:")
    expect(chips.text()).toContain("Title")
    expect(chips.text()).toContain("Summary ×2")
    expect(chips.text()).toContain("Transcript")
  })

  it("hides the matched-fields kicker when no hit resolves to an episode-level field", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "x",
      error: null,
      results: [
        {
          doc_id: "kg1",
          score: 0.9,
          text: "t",
          source_tier: "kg",
          metadata: {
            doc_type: "kg_topic",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
      ],
    })
    const { w } = await mountAt("x")
    expect(w.find('[data-testid="matched-fields"]').exists()).toBe(false)
  })

  // #1261-7: year-header grouping — mobile-friendly reshape of the timeline chart
  it("shows year section headers when results span multiple publish years", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "ai",
      error: null,
      results: [
        {
          doc_id: "a",
          score: 0.9,
          text: "a",
          source_tier: "insight",
          metadata: {
            doc_type: "insight",
            episode_slug: "ep-2024",
            episode_title: "From 2024",
            publish_date: "2024-04-01",
          },
        },
        {
          doc_id: "b",
          score: 0.8,
          text: "b",
          source_tier: "insight",
          metadata: {
            doc_type: "insight",
            episode_slug: "ep-2023",
            episode_title: "From 2023",
            publish_date: "2023-11-01",
          },
        },
      ],
    })
    const { w } = await mountAt("ai")
    const headers = w.findAll('[data-testid="year-header"]')
    expect(headers).toHaveLength(2)
    expect(headers[0].text()).toContain("2024")
    expect(headers[1].text()).toContain("2023")
  })

  // #1261-8: Save/unsave the current query
  it('save-query button toggles between "Save" and "Saved ✓" and persists via the store', async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "sleep",
      error: null,
      results: [],
    })
    // mockImplementation: Response body is a one-shot stream; multiple hydrate
    // / patch calls consume distinct bodies.
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response(JSON.stringify({ preferences: {} }), { status: 200 })
    )
    const { w } = await mountAt("sleep")
    const auth = useAuthStore() // saving is sign-in gated; ensureLoaded no-ops when loaded
    auth.user = { user_id: "u1", email: "a@b.c", name: "A" }
    auth.loaded = true
    const saveBtn = w.get('[data-testid="save-query-button"]')
    // Initially: not saved.
    expect(saveBtn.text()).toBe("Save")
    await saveBtn.trigger("click")
    await flushPromises()
    const savedQueries = useSavedQueriesStore()
    expect(savedQueries.count).toBe(1)
    expect(savedQueries.list[0].q).toBe("sleep")
    expect(saveBtn.text()).toBe("Saved ✓")
    // A confirmation appears so the save isn't silent.
    expect(w.get('[data-testid="save-query-confirm"]').text()).toContain("Library")
    // Tapping again removes it (toggle).
    await saveBtn.trigger("click")
    await flushPromises()
    expect(savedQueries.count).toBe(0)
    expect(saveBtn.text()).toBe("Save")
  })

  it("routes a signed-out save to sign-in instead of silently not persisting", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "sleep", error: null, results: [] })
    const { w, router } = await mountAt("sleep")
    useAuthStore().loaded = true // signed out but session resolved → gated, not a stray refresh
    const push = vi.spyOn(router, "push")
    await w.get('[data-testid="save-query-button"]').trigger("click")
    await flushPromises()
    // Signed out → gated to sign-in; nothing "saved" locally.
    expect(useSavedQueriesStore().count).toBe(0)
    expect(push).toHaveBeenCalledWith(expect.objectContaining({ name: "login" }))
  })

  it("hides the save button when the query is blank / whitespace", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "", error: null, results: [] })
    const { w } = await mountAt("")
    expect(w.find('[data-testid="save-query-button"]').exists()).toBe(false)
  })

  it("hides year headers when results all fall in a single year", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "ai",
      error: null,
      results: [
        {
          doc_id: "a",
          score: 0.9,
          text: "a",
          source_tier: "insight",
          metadata: {
            doc_type: "insight",
            episode_slug: "ep-2024a",
            episode_title: "Ep A",
            publish_date: "2024-04-01",
          },
        },
        {
          doc_id: "b",
          score: 0.8,
          text: "b",
          source_tier: "insight",
          metadata: {
            doc_type: "insight",
            episode_slug: "ep-2024b",
            episode_title: "Ep B",
            publish_date: "2024-11-01",
          },
        },
      ],
    })
    const { w } = await mountAt("ai")
    expect(w.findAll('[data-testid="year-header"]')).toHaveLength(0)
  })

  it("keeps insight and kg_topic hits out of the fold — they render as standalone rows", async () => {
    vi.spyOn(api, "searchCorpus").mockResolvedValue({
      query: "ai",
      error: null,
      results: [
        {
          doc_id: "i1",
          score: 0.9,
          text: "A grounded insight.",
          source_tier: "insight",
          metadata: {
            doc_type: "insight",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
        {
          doc_id: "t1",
          score: 0.7,
          text: "A transcript chunk.",
          source_tier: "segment",
          metadata: {
            doc_type: "transcript",
            episode_slug: "show-x",
            episode_title: "Ep X",
            podcast_title: "Show",
          },
        },
      ],
    })
    const { w } = await mountAt("ai")
    // The insight text shows up without needing to expand anything.
    expect(w.text()).toContain("A grounded insight.")
    // The single-transcript-hit cluster is present but collapsed by default.
    expect(w.findAll('[data-testid="folded-cluster-row"]')).toHaveLength(1)
    expect(w.text()).not.toContain("A transcript chunk.")
  })

  describe("a result header spends its width on the text (#2004 follow-up)", () => {
    async function resultRow() {
      vi.spyOn(api, "searchCorpus").mockResolvedValue({
        query: "memory",
        error: null,
        results: [
          {
            doc_id: "d1",
            score: 0.9,
            text: "A grounded passage.",
            source_tier: "segment",
            metadata: {
              episode_slug: "show-x",
              episode_title: "A title long enough to want the room",
              podcast_title: "Show",
            },
            lifted: { quote: { timestamp_start_ms: 20000 } },
          },
        ],
      })
      const { w } = await mountAt("memory")
      return w
    }

    // These three invariants predate the shared card and still hold — Search now renders
    // `EpisodeCard` (operator 2026-09-17), so they are asserted against ITS structure rather than
    // against the hand-rolled copy they were written for. The action cluster's testid is the shared
    // `episode-actions`; the narrow column is `lp-media-aside`; the text column is `lp-media-body`.

    it("the match count and the actions sit in ONE column with the artwork", async () => {
      // The header was [artwork + text] | [count + actions], squeezing the text from both sides while
      // the right rail kept a column to itself with empty space under it. Everything that is not the
      // text stacks under the artwork, at one width.
      const w = await resultRow()
      const column = w.get('[data-testid="episode-card"] .lp-media-aside').element as HTMLElement
      expect(
        column.querySelector('[data-testid="episode-actions"]'),
        "the actions are not in the narrow left column"
      ).not.toBeNull()
      expect(column.textContent, "the match count is not in the same column").toMatch(/match/i)
    })

    it("an episode group collapses its matches and starts expanded", async () => {
      // Collapsible on BOTH Search and Revisit through the shared EpisodeGroupCard (operator
      // 2026-09-17). Expanded by default: a listener who just searched must not have to open every
      // group to read the results they asked for. `v-show`, so a folded transcript cluster the user
      // expanded INSIDE the group survives collapsing and re-opening it.
      const w = await resultRow()
      const toggle = w.get('[data-testid="episode-group-toggle"]')
      const body = w.get('[data-testid="episode-group-body"]')
      expect(toggle.attributes("aria-expanded")).toBe("true")
      expect(toggle.text()).toContain("Hide matches")
      await toggle.trigger("click")
      expect(toggle.attributes("aria-expanded")).toBe("false")
      expect(toggle.text()).toContain("Show matches")
      expect(body.attributes("style")).toContain("display: none")
    })

    it("the actions are NOT inside another interactive", async () => {
      // An interactive control inside another interactive control: the whole reason these are
      // siblings, and easy to undo while moving them around. Anchors count as well as buttons —
      // EpisodeCard's title is a stretched <a>, so nesting here would be just as wrong.
      //
      // Checked across EVERY match, not the first one: asserting on `get()` alone once passed while
      // a second, nested cluster existed. One row must yield exactly one cluster.
      const w = await resultRow()
      const all = w.findAll('[data-testid="episode-actions"]')
      expect(all, "expected one action cluster per result row").toHaveLength(1)
      for (const a of all) {
        let n: HTMLElement | null = a.element.parentElement
        while (n) {
          expect(
            ["button", "a"].includes(n.tagName.toLowerCase()),
            "actions nested inside a button or link"
          ).toBe(false)
          n = n.parentElement
        }
      }
    })

    it("the text block is a sibling of that column, free to use the rest of the row", async () => {
      const w = await resultRow()
      const card = w.get('[data-testid="episode-card"]').element as HTMLElement
      const column = card.querySelector(".lp-media-aside") as HTMLElement
      const body = card.querySelector(".lp-media-body") as HTMLElement
      expect(column, "no narrow column").toBeTruthy()
      expect(body, "no text column beside the column").toBeTruthy()
      expect(body.parentElement, "the two are not siblings").toBe(column.parentElement)
      expect(body.textContent).toContain("A title long enough to want the room")
    })

    it("surfaces the listener's own matching notes (SR.1)", async () => {
      vi.spyOn(api, "searchCorpus").mockResolvedValue({ query: "sleep", error: null, results: [] })
      vi.spyOn(api, "getHighlights").mockResolvedValue([])
      vi.spyOn(api, "getNotes").mockResolvedValue([
        {
          id: "n1",
          target: "episode",
          target_id: "ep-1",
          text: "my note on sleep cycles",
          created_at: 1,
          updated_at: 1,
        },
        {
          id: "n2",
          target: "topic",
          target_id: "t1",
          text: "unrelated thought",
          created_at: 2,
          updated_at: 2,
        },
      ])
      const { w } = await mountAt("sleep")
      const sec = w.find('[data-testid="search-note-matches"]')
      expect(sec.exists()).toBe(true)
      expect(sec.text()).toContain("my note on sleep cycles")
      expect(sec.text()).not.toContain("unrelated thought")
    })
  })
})
