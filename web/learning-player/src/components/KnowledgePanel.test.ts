import { flushPromises, mount } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createPinia, setActivePinia } from "pinia"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import * as native from "../services/native"
import en from "../i18n/locales/en.json"
import type { EpisodeDetail, Entity, Highlight, Insight, Topic } from "../services/types"
import { useAuthStore } from "../stores/auth"
import KnowledgePanel from "./KnowledgePanel.vue"
import knowledgePanelSource from "./KnowledgePanel.vue?raw"

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/episode/:slug", name: "player", component: { template: "<div/>" } },
    { path: "/search", name: "search", component: { template: "<div/>" } },
  ],
})

const emptyPage = { items: [], page: 1, page_size: 6, total: 0, has_more: false }

/** A chip's NAME, without the kind label every chip in the mixed group now leads with. */
const chipName = (b: { text(): string }): string => b.text().replace(/^(Topic|Person)/, "").trim()

beforeEach(() => {
  setActivePinia(createPinia()) // FavoriteButton (on insights) resolves the favorites/auth stores
  // Default: no related peers (index unavailable) so the section hides.
  vi.spyOn(api, "getRelated").mockResolvedValue(emptyPage)
  // Keep tests off the network should anything in the panel ask for episode enrichment.
  vi.spyOn(api, "getEpisodeEnrichment").mockResolvedValue({})
})
afterEach(() => {
  vi.restoreAllMocks()
  // The notes viewer teleports to <body>, and these mounts are never unmounted — so without this
  // one test's open viewer is still in the document during the next, and a test asserting the
  // viewer is ABSENT passes or fails on its predecessor's leftovers rather than its own behaviour.
  document.body.innerHTML = ""
})

function episode(): EpisodeDetail {
  return {
    slug: "s1",
    title: "Ep",
    feed_id: "f",
    podcast_title: "Show",
    publish_date: "2024-01-01",
    duration_seconds: 1800,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    summary_title: "Sum",
    summary_bullets: [],
    summary_text: "A short summary.",
    has_transcript: true,
    has_summary: true,
    has_gi: true,
    has_kg: true,
    has_bridge: false,
  }
}

function insight(over: Partial<Insight> = {}): Insight {
  return {
    id: "i1",
    text: "Sleep consolidates memory.",
    grounded: true,
    insight_type: "claim",
    confidence: null,
    position_hint: null,
    quotes: [
      {
        text: "the spindles gate memory",
        speaker: "person:matthew-walker",
        char_start: null,
        char_end: null,
        start_ms: 12000,
        end_ms: 15000,
      },
    ],
    ...over,
  }
}

function mountPanel(props: {
  episode?: EpisodeDetail
  insights?: Insight[]
  topics?: Topic[]
  persons?: Entity[]
  slug?: string
  activeInsightId?: string | null
} = {}) {
  return mount(KnowledgePanel, {
    props: {
      episode: episode(),
      insights: [insight()],
      topics: [{ id: "topic:memory", label: "memory" } as Topic],
      persons: [{ id: "person:matthew-walker", name: "Matthew Walker", kind: "person" } as Entity],
      slug: "s1",
      activeInsightId: null,
      ...props,
    },
    global: { plugins: [i18n, router] },
  })
}

describe("KnowledgePanel", () => {
  it("renders summary, topics, people, and insight cards", () => {
    const w = mountPanel()
    expect(w.text()).toContain("A short summary.")
    expect(w.text()).toContain("memory")
    expect(w.text()).toContain("Matthew Walker")
    expect(w.text()).toContain("Sleep consolidates memory.")
    expect(w.text()).toContain("the spindles gate memory") // verbatim quote
  })

  it("emits seek with the insight quote start (jump-to-moment)", async () => {
    const w = mountPanel()
    // The timestamp button shows 0:12 (12000ms).
    const btn = w.findAll("button").find((b) => b.text().includes("0:12"))
    expect(btn).toBeTruthy()
    await btn!.trigger("click")
    // "▶ Play from" seeks AND plays (operator 2026-10-05) — its own event, so the player plays.
    expect(w.emitted("play-from")?.[0]).toEqual([12])
  })

  it("tapping a person chip opens its entity card (PRD-043)", async () => {
    const getPerson = vi.spyOn(api, "getPersonCard").mockResolvedValue({
      id: "person:matthew-walker",
      label: "Matthew Walker",
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
    })
    const w = mountPanel()
    await w
      .findAll("button")
      .find((b) => chipName(b) === "Matthew Walker")!
      .trigger("click")
    await flushPromises()
    // Replace-in-panel (UXS-014): the card renders INLINE in the panel (no overlay), with a ‹ Back
    // (glyph-only dismiss control, aria-label "Back" when nested).
    expect(getPerson).toHaveBeenCalledWith("person:matthew-walker")
    expect(w.text()).toContain("Matthew Walker")
    expect(w.find('[data-testid="ec-dismiss"]').attributes("aria-label")).toBe("Back")
  })

  it("opened from a note (focusNotes), the panel lands on the notes section", async () => {
    // Put the notes 900px down the panel body (jsdom lays nothing out).
    const rect = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (
      this: HTMLElement
    ) {
      // Like a browser: scrolling the panel body moves the notes up by as much.
      const body = this.closest(".overflow-y-auto") as HTMLElement | null
      return { top: this.id === "notes" ? 900 - (body?.scrollTop ?? 0) : 0 } as DOMRect
    })
    const w = mount(KnowledgePanel, {
      props: {
        episode: episode(),
        insights: [insight()],
        topics: [],
        persons: [],
        slug: "s1",
        activeInsightId: null,
        focusNotes: true,
      },
      global: { plugins: [i18n, router] },
      attachTo: document.body,
    })
    await flushPromises()
    // The landing waits for the notes to stop moving (300ms settle).
    await new Promise((r) => setTimeout(r, 450))
    expect((w.find(".overflow-y-auto").element as HTMLElement).scrollTop).toBe(900)
    rect.mockRestore()
  })

  it("closing the card returns the panel to where it was scrolled, not the top (operator 2026-10-04)", async () => {
    vi.spyOn(api, "getPersonCard").mockResolvedValue({
      id: "person:matthew-walker",
      label: "Matthew Walker",
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
    })
    // jsdom lays nothing out, so every scroller reports zero height; give them room to scroll.
    const height = vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockReturnValue(3000)
    const w = mountPanel()
    const body = () => w.find(".overflow-y-auto").element as HTMLElement
    body().scrollTop = 640 // down at the people row
    await w
      .findAll("button")
      .find((b) => chipName(b) === "Matthew Walker")!
      .trigger("click")
    await flushPromises()
    await w.find('[data-testid="ec-dismiss"]').trigger("click")
    await flushPromises()
    // The panel body is a NEW element after the card closes; it must not start at 0.
    expect(body().scrollTop).toBe(640)
    height.mockRestore()
  })

  it("tapping a topic chip opens its entity card (not a search)", async () => {
    const getTopic = vi.spyOn(api, "getTopicCard").mockResolvedValue({
      id: "topic:memory",
      label: "memory",
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
    const push = vi.spyOn(router, "push")
    const w = mountPanel()
    await w
      .findAll("button")
      .find((b) => chipName(b) === "memory")!
      .trigger("click")
    await flushPromises()
    expect(getTopic).toHaveBeenCalledWith("topic:memory")
    expect(push).not.toHaveBeenCalled() // search now lives inside the card, not on chip-tap
  })

  it("orders topics cluster-first and marks the dominant cluster (RFC-102)", () => {
    const topics: Topic[] = [
      { id: "topic:z", label: "zulu", cluster_id: null, cluster_label: null, cluster_size: 0 },
      {
        id: "topic:ai",
        label: "ai",
        cluster_id: "tc:ml",
        cluster_label: "machine learning",
        cluster_size: 5,
      },
      {
        id: "topic:ml",
        label: "ml",
        cluster_id: "tc:ml",
        cluster_label: "machine learning",
        cluster_size: 5,
      },
    ]
    const w = mountPanel({ topics, persons: [] })
    // The dominant THEME is named again, as a THEME pill beside the storyline (operator
    // 2026-10-04). It was removed on 2026-09-19 because a theme then had no card and the line went
    // nowhere; it has one now (ThemeCard, opened by the theme's own `tc:` id), so it is a real pill.
    expect(w.text()).not.toContain("Similar ·")
    const theme = w.get('[data-testid="kp-theme-link"]')
    expect(theme.text()).toBe("Theme machine learning")
    expect(theme.classes()).toContain("text-theme")
    // Dominant-cluster topics lead (ai, ml), the singleton (zulu) trails.
    const chips = w.findAll("button").filter((b) => ["ai", "ml", "zulu"].includes(chipName(b)))
    expect(chips.map((c) => chipName(c))).toEqual(["ai", "ml", "zulu"])
    // The theme's member topics carry a ring in the THEME colour, tying them to the pill above;
    // the singleton does not.
    expect(chips[0].classes()).toContain("ring-theme/60")
    expect(chips[2].classes()).not.toContain("ring-theme/60")
  })

  it('marks co-occurrence topics with a "Storyline ·" lead-in and theme ring', () => {
    const topics: Topic[] = [
      {
        id: "topic:oil",
        label: "oil",
        cluster_id: null,
        cluster_label: null,
        cluster_size: 0,
        storyline_id: "thc:sanctions",
        storyline_label: "sanctions",
        storyline_size: 3,
      },
      {
        id: "topic:sf",
        label: "shadow fleet",
        cluster_id: null,
        cluster_label: null,
        cluster_size: 0,
        storyline_id: "thc:sanctions",
        storyline_label: "sanctions",
        storyline_size: 3,
      },
      { id: "topic:z", label: "zulu", cluster_id: null, cluster_label: null, cluster_size: 0 },
    ]
    const w = mountPanel({ topics, persons: [] })
    // The storyline is a PILL now, with its kind named separately from its label, so the text is
    // "Storyline" + "sanctions" rather than the old "Storyline · sanctions" lead-in. Asserted via
    // the testid plus its content, which survives a restyle — the previous string assertion would
    // have broken on any punctuation change.
    const pill = w.get('[data-testid="kp-storyline-link"]')
    expect(pill.text()).toContain("Storyline")
    expect(pill.text()).toContain("sanctions")
    // Theme-member chips carry the teal fill (lp-storyline-chip); the non-member does not.
    const oil = w.findAll("button").find((b) => chipName(b) === "oil")!
    const zulu = w.findAll("button").find((b) => chipName(b) === "zulu")!
    expect(oil.classes()).toContain("lp-storyline-chip")
    expect(zulu.classes()).not.toContain("lp-storyline-chip")
  })

  it("runs episode-scoped search and renders grounded results", async () => {
    const api = await import("../services/api")
    vi.spyOn(api, "searchEpisode").mockResolvedValue({
      query: "memory",
      error: null,
      results: [
        {
          doc_id: "d1",
          score: 0.9,
          text: "A grounded passage about memory.",
          metadata: {},
          source_tier: "segment",
          lifted: { quote: { timestamp_start_ms: 20000 } },
        },
      ],
    })
    const w = mountPanel()
    await w.find("input").setValue("memory")
    await w.find("form").trigger("submit")
    await new Promise((r) => setTimeout(r, 0))
    expect(w.text()).toContain("A grounded passage about memory.")
    const jump = w.findAll("button").find((b) => b.text().includes("0:20"))
    await jump!.trigger("click")
    expect(w.emitted("play-from")?.at(-1)).toEqual([20])
  })

  it('renders "More like this" peers with links to the player', async () => {
    vi.spyOn(api, "getRelated").mockResolvedValue({
      items: [
        {
          slug: "peer-1",
          title: "A Related Episode",
          feed_id: "f",
          podcast_title: "Show",
          publish_date: null,
          duration_seconds: null,
          episode_image_url: null,
          feed_image_url: null,
          artwork_url: null,
          status: "ready",
          summary_preview: null,
          summary_text: null,
          summary_bullets: [],
          topics: [],
          has_transcript: true,
          has_summary: false,
          has_gi: false,
          has_kg: false,
          has_bridge: false,
        },
      ],
      page: 1,
      page_size: 6,
      total: 1,
      has_more: false,
    })
    const w = mountPanel()
    await flushPromises()
    expect(w.text()).toContain("More like this")
    expect(w.text()).toContain("A Related Episode")
    expect(w.findAll("a").map((a) => a.attributes("href"))).toContain("/episode/peer-1")
  })

  it("shows the empty message when no intelligence is present", () => {
    const e = episode()
    e.summary_text = null
    e.summary_title = null
    const w = mountPanel({ episode: e, insights: [], topics: [], persons: [] })
    expect(w.text()).toContain("Insights appear once this episode is processed.")
  })

  it("shows the insight save as a sign-in-gated BOOKMARK when signed out", () => {
    const w = mountPanel()
    // Renders signed-out (#1590) but gated: the label is the sign-in prompt.
    //
    // A bookmark, not a heart (operator 2026-09-27). It writes a HIGHLIGHT, the same as a
    // transcript line does, and the heart is the favourite mark for whole objects. It used to
    // announce "Save to favorites" while doing neither.
    const save = w.find('[data-testid="highlight-toggle"]')
    expect(save.exists()).toBe(true)
    expect(save.attributes("aria-label")).toBe("Sign in to mark this moment")
    expect(w.find(".lp-fav").exists(), "the heart does not belong on a fragment").toBe(false)
  })

  it("lets a signed-in user save an insight to highlights (P2 capture)", async () => {
    const auth = useAuthStore()
    auth.user = { user_id: "u1", email: "a@b.c", name: "A" }
    vi.spyOn(api, "getHighlights").mockResolvedValue([])
    vi.spyOn(api, "getNotes").mockResolvedValue([])
    const created: Highlight = {
      id: "h1",
      episode_slug: "s1",
      kind: "insight",
      start_ms: 12000,
      end_ms: null,
      char_start: null,
      char_end: null,
      segment_ids: [],
      quote_text: "Sleep consolidates memory.",
      speaker: null,
      source_insight_id: "i1",
      color: null,
      created_at: 1,
      anchor_status: null,
    }
    const create = vi.spyOn(api, "createHighlight").mockResolvedValue(created)
    const w = mountPanel()
    await flushPromises()
    const save = w.find('[data-testid="highlight-toggle"]')
    expect(save.exists()).toBe(true)
    await save.trigger("click")
    await flushPromises() // the gate resolves the session before acting (#1590)
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "insight", source_insight_id: "i1", start_ms: 12000 })
    )
  })

  it("offers the insight save to signed-out visitors as a teaser (#1590)", async () => {
    // It used to be `v-if="auth.isAuthenticated"`. Saving an insight is the learning loop's payoff;
    // hiding it left signed-out readers with no evidence the product does this at all.
    vi.spyOn(api, "getHighlights").mockResolvedValue([])
    vi.spyOn(api, "getNotes").mockResolvedValue([])
    const create = vi.spyOn(api, "createHighlight")
    const w = mountPanel()
    await flushPromises()

    const save = w.find('[data-testid="highlight-toggle"]')
    expect(save.exists()).toBe(true) // renders signed-out as a teaser
    await save.trigger("click")
    await flushPromises()
    expect(create).not.toHaveBeenCalled() // gated → routes to sign-in, no write
  })

  it("the insight save is the one shared BOOKMARK, writing a highlight not a favorite (RFC-121/#1593)", async () => {
    // An insight is saved by the ONE shared `HighlightToggle` — the same component the transcript
    // line uses, because it is the same action. It writes an insight HIGHLIGHT (capture path) and
    // must NEVER call the favorites path: favourite(insight) is the "same text, two destinations"
    // that #1593 banned.
    //
    // It drew a HEART until 2026-09-27, which is how the ban survived in the data layer while the
    // UI said the opposite out loud. The heart's absence is asserted, not just the bookmark's
    // presence — otherwise reattaching one alongside would keep this green.
    const auth = useAuthStore()
    auth.user = { user_id: "u1", email: "a@b.c", name: "A" }
    vi.spyOn(api, "getHighlights").mockResolvedValue([])
    vi.spyOn(api, "getNotes").mockResolvedValue([])
    const create = vi.spyOn(api, "createHighlight").mockResolvedValue({
      id: "h1",
      episode_slug: "s1",
      kind: "insight",
      start_ms: 0,
      end_ms: null,
      char_start: null,
      char_end: null,
      segment_ids: [],
      quote_text: "",
      speaker: null,
      source_insight_id: "i1",
      color: null,
      created_at: 1,
      anchor_status: null,
    } as Highlight)
    const addFav = vi.spyOn(api, "addFavorite")
    const w = mountPanel()
    await flushPromises()

    const saves = w.findAll('[data-testid="highlight-toggle"]')
    expect(saves.length).toBe(1) // one save per insight, not two
    expect(w.findAll(".lp-fav").length, "no heart on a fragment").toBe(0)
    await saves[0].trigger("click")
    await flushPromises()
    expect(create).toHaveBeenCalled() // → highlights/capture path
    expect(addFav).not.toHaveBeenCalled() // never the favorites path (#1593)
  })
})

describe("KnowledgePanel — #1191 route-and-tag surfacing", () => {
  it("shows surface-tagged and untagged (pre-3.1) insights, hides connect/drop", () => {
    const w = mountPanel({
      insights: [
        insight({ id: "a", text: "AAA surface one", routing_tag: "surface" }),
        insight({ id: "b", text: "BBB connect plumbing", routing_tag: "connect" }),
        insight({ id: "c", text: "CCC dropped filler", routing_tag: "drop" }),
        insight({ id: "d", text: "DDD untagged legacy", routing_tag: null }),
      ],
    })
    expect(w.text()).toContain("AAA surface one")
    expect(w.text()).toContain("DDD untagged legacy") // back-compat: null tag kept
    expect(w.text()).not.toContain("BBB connect plumbing")
    expect(w.text()).not.toContain("CCC dropped filler")
  })

  it("caps at 4 surface insights behind 'Show N more', then reveals the rest (operator 2026-10-07)", async () => {
    const many = Array.from({ length: 10 }, (_, i) =>
      insight({ id: "i" + i, text: "INSIGHT_" + i, routing_tag: "surface" as const })
    )
    const w = mountPanel({ insights: many })
    expect(w.text()).toContain("INSIGHT_3")
    expect(w.text()).not.toContain("INSIGHT_4")
    expect(w.text()).not.toContain("INSIGHT_9")
    const showMore = w.find('[data-testid="kp-insights-show-all"]')
    expect(showMore.text()).toBe("Show 6 more")
    await showMore.trigger("click")
    expect(w.text()).toContain("INSIGHT_8")
    expect(w.text()).toContain("INSIGHT_9")
  })

  it("preserves the server-provided (salience) order and does not re-sort", () => {
    // The server returns insights salience-desc; the panel must render them in THAT order, not
    // re-sort by id/text. Input order (z, a, m) is deliberately not id- or text-sorted, so a panel
    // that re-sorted would reorder them — the DOM order must match the server order.
    const w = mountPanel({
      insights: [
        insight({ id: "z", text: "ZZZ highest salience", routing_tag: "surface" }),
        insight({ id: "a", text: "AAA middle salience", routing_tag: "surface" }),
        insight({ id: "m", text: "MMM lowest salience", routing_tag: "surface" }),
      ],
    })
    const t = w.text()
    expect(t.indexOf("ZZZ highest salience")).toBeLessThan(t.indexOf("AAA middle salience"))
    expect(t.indexOf("AAA middle salience")).toBeLessThan(t.indexOf("MMM lowest salience"))
  })

  it("hides the insights section entirely when none are surface-tagged", () => {
    const w = mountPanel({
      insights: [insight({ id: "x", text: "only connect", routing_tag: "connect" })],
    })
    expect(w.text()).not.toContain("only connect")
  })

  it("#2198: shows an unattributed insight the server sent as the fallback, and says so", () => {
    // ChinaTalk 0df8ed52: 44 grounded insights, naming resolved nobody, the panel was empty.
    const w = mountPanel({
      insights: [
        insight({ id: "u", text: "UUU said by an unnamed voice", routing_tag: "connect", attributed: false }),
      ],
    })
    expect(w.text()).toContain("UUU said by an unnamed voice")
    expect(w.find('[data-testid="insight-unattributed"]').exists()).toBe(true)
  })

  it("#2198: a named insight carries no 'speaker not identified' label", () => {
    const w = mountPanel({
      insights: [insight({ id: "n", text: "NNN named", routing_tag: "surface" })],
    })
    expect(w.text()).toContain("NNN named")
    expect(w.find('[data-testid="insight-unattributed"]').exists()).toBe(false)
  })
})

describe("Topics & People: five, mixed, then '+N more' (operator 2026-10-07)", () => {
  // Reverses #2004 item 15 ("render every chip") on the operator's call: the panel opens on a gist.
  const people = [
    { id: "person:h", name: "Hosty", kind: "person", role: "host" },
    { id: "person:g", name: "Guesty", kind: "person", role: "guest" },
    { id: "person:m", name: "Mentiony", kind: "person", role: "mentioned" },
    { id: "person:m2", name: "Another", kind: "person", role: "mentioned" },
  ] as Entity[]
  const topics = Array.from({ length: 8 }, (_, i) => ({ id: `topic:t${i}`, label: `topic ${i}` })) as Topic[]

  it("shows five: the guest and one other person (never the host), then topics", () => {
    const w = mountPanel({ topics, persons: people })
    const row = w.get('[data-testid="kp-tags-row"]')
    const persons = row.findAll('[data-testid="kp-person-chip"]').map((c) => c.text())
    expect(persons.some((t) => t.includes("Guesty"))).toBe(true)
    expect(persons.some((t) => t.includes("Mentiony"))).toBe(true)
    expect(persons.some((t) => t.includes("Hosty"))).toBe(false)
    expect(row.findAll('[data-testid="kp-topic-chip"]')).toHaveLength(3)
    // 4 people + 8 topics = 12; 5 shown.
    expect(w.get('[data-testid="kp-tags-more"]').text()).toBe("+7 more")
  })

  it("'+N more' reveals every chip, host included", async () => {
    const w = mountPanel({ topics, persons: people })
    await w.get('[data-testid="kp-tags-more"]').trigger("click")
    for (const t of topics) expect(w.text()).toContain(t.label)
    expect(w.text()).toContain("Hosty")
    expect(w.find('[data-testid="kp-tags-more"]').exists()).toBe(false)
  })

  it("leads with the storyline and the theme when the episode has them, and counts them in the five", () => {
    const clustered = topics.map((tp, i) =>
      i < 3 ? { ...tp, cluster_id: "tc:x", cluster_label: "Theme X", storyline_id: "thc:s", storyline_label: "Story S" } : tp,
    ) as Topic[]
    const w = mountPanel({ topics: clustered, persons: [] })
    const row = w.get('[data-testid="kp-tags-row"]')
    expect(row.find('[data-testid="kp-theme-link"]').exists()).toBe(true)
    expect(row.find('[data-testid="kp-storyline-link"]').exists()).toBe(true)
    expect(row.findAll('[data-testid="kp-topic-chip"]')).toHaveLength(3)
    // The header counts what the row holds — the pills too — so "5 + N more" adds up to it.
    expect(w.text()).toContain("· 10")
    expect(w.get('[data-testid="kp-tags-more"]').text()).toBe("+5 more")
  })

  it("no '+N more' when everything already fits", () => {
    const w = mountPanel({ topics: topics.slice(0, 2), persons: [people[1]] })
    expect(w.find('[data-testid="kp-tags-more"]').exists()).toBe(false)
  })
})

describe("Key points: two, then 'Show N more' (operator 2026-10-07)", () => {
  it("shows the first two, then the rest on tap", async () => {
    const bullets = ["P1", "P2", "P3", "P4", "P5"]
    const w = mountPanel({ episode: { ...episode(), summary_bullets: bullets } } as never)
    const list = () => w.get('[data-testid="summary-bullets"]').findAll("li").map((l) => l.text())
    expect(list()).toEqual(["P1", "P2"])
    const more = w.get('[data-testid="kp-key-points-more"]')
    expect(more.text()).toBe("Show 3 more")
    await more.trigger("click")
    expect(list()).toEqual(bullets)
  })
})

describe("insight types are distinguishable (#2004 item 8)", () => {
  const TYPES = ["claim", "observation", "recommendation", "question"]

  /** The mark's shape lives in its path, its identity in the token driving its colour. */
  function markOf(type: string): { path: string; color: string; label: string } {
    const w = mountPanel({ insights: [insight({ insight_type: type })] } as never)
    const el = w.get('[data-testid="insight-type"]')
    return {
      path: el.get("svg path").attributes("d") ?? "",
      color: el.get("svg").attributes("style") ?? "",
      label: el.text(),
    }
  }

  it.each(TYPES)("gives %s a mark and names it", (type) => {
    const m = markOf(type)
    expect(m.path, `${type} rendered no mark`).not.toBe("")
    expect(m.label).toContain(type)
  })

  it("every type is distinguishable from every other, by SHAPE", () => {
    // The actual requirement, and the one the first attempt missed: all four differ from ALL the
    // others, not just from one neighbour. Shape is asserted separately from colour because colour
    // is the second channel — the marks must still be separable in greyscale.
    const paths = TYPES.map((t) => markOf(t).path)
    expect(new Set(paths).size, `two types share a shape: ${paths.join(" | ")}`).toBe(TYPES.length)
  })

  it("every type is distinguishable by COLOUR too, and none of them is the accent", () => {
    // The accent means "you can act on this" (UXS-011). A type mark is not an action, so it must
    // never spend it — that is why these got their own tokens instead of borrowing.
    const colors = TYPES.map((t) => markOf(t).color)
    expect(new Set(colors).size, `two types share a colour: ${colors.join(" | ")}`).toBe(
      TYPES.length
    )
    for (const c of colors) {
      expect(c, "a type mark spends the accent").not.toContain("--lp-accent")
    }
  })

  it("every type explains itself on hover", () => {
    // A symbol nobody can decode is decoration. The visible word says WHICH type; the tooltip says
    // what that type means, which is the part a new reader is missing.
    for (const type of TYPES) {
      const w = mountPanel({ insights: [insight({ insight_type: type })] } as never)
      const title = w.get('[data-testid="insight-type"]').attributes("title") ?? ""
      expect(title, `${type} has no tooltip`).not.toBe("")
      expect(title.toLowerCase(), `${type}'s tooltip does not describe it`).toContain(type)
      expect(title, "the tooltip leaked its i18n key").not.toContain("kp.insightType")
    }
  })

  it("an unrecognised type gets a real sentence, not an empty tooltip", () => {
    // A tooltip that opens blank reads as a broken tooltip.
    const w = mountPanel({ insights: [insight({ insight_type: "speculation" })] } as never)
    const title = w.get('[data-testid="insight-type"]').attributes("title") ?? ""
    expect(title).not.toBe("")
    expect(title).not.toContain("kp.insightType")
  })

  it("the type mark is the FIRST thing in the row — nothing constant precedes it", () => {
    // The regression this replaces: a green "grounded" dot rendered before the type glyph on every
    // grounded row, so the row still opened with an identical mark and the differentiating one had
    // to compete with it. It also duplicated the ▶ timestamp on the same row, which says the same
    // thing more precisely.
    const w = mountPanel({ insights: [insight({ insight_type: "claim" })] } as never)
    const row = w.get('[data-testid="insight-type"]')
    expect(row.element.firstElementChild?.tagName.toLowerCase()).toBe("svg")
    expect(w.html(), "the grounded dot is back in front of the type").not.toContain("●")
  })

  it('renders no type label for "unknown" — it would say nothing', () => {
    const w = mountPanel({ insights: [insight({ insight_type: "unknown" })] } as never)
    expect(w.find('[data-testid="insight-type"]').exists()).toBe(false)
    // the insight itself still renders
    expect(w.text()).toContain("Sleep consolidates memory.")
  })

  it("handles a type it has no glyph for without breaking the row", () => {
    // The vocabulary is closed today, but a new value must degrade to a neutral mark rather than
    // rendering "undefined" beside the label.
    const w = mountPanel({ insights: [insight({ insight_type: "speculation" })] } as never)
    const el = w.get('[data-testid="insight-type"]')
    // A neutral dot, not one of the four identities — and not an empty mark column, which would
    // make the one already-unusual row the only one that does not line up.
    expect(el.get("svg").attributes("style")).toContain("--lp-muted")
    expect(el.text()).toContain("speculation")
    expect(el.text()).not.toContain("undefined")
  })

  describe("the summary spine (#2004 follow-up)", () => {
    it("renders the key points between the summary and the insights", () => {
      // The bullets had NO home: the browse card counts them without showing them, and the Summary
      // panel is the prose alone. The panel's order is its argument — what the episode is about, the
      // shape of it, then the moments it is built from.
      const w = mountPanel({
        episode: { summary_text: "The prose.", summary_bullets: ["First point", "Second point"] },
        insights: [insight({ insight_type: "claim" })],
      } as never)
      const list = w.get('[data-testid="summary-bullets"]')
      expect(list.findAll("li")).toHaveLength(2)

      const html = w.html()
      expect(
        html.indexOf("The prose.") < html.indexOf("First point"),
        "the key points render above the summary"
      ).toBe(true)
      expect(
        html.indexOf("First point") < html.indexOf('data-testid="insight-type"'),
        "the key points render below the insight list"
      ).toBe(true)
    })

    it("does NOT fall back to the thematic headline for the summary", () => {
      // This block sits on the same screen as the Summary panel, which shows prose only. A fallback
      // here meant two panels in one player giving different answers to "what is the summary".
      const w = mountPanel({
        episode: { summary_text: "", summary_title: "A thematic headline", summary_bullets: [] },
      } as never)
      expect(w.text(), "the headline is standing in for a summary").not.toContain(
        "A thematic headline"
      )
    })

    it("shows the key points even when the episode has no prose summary", () => {
      // They are independent fields; withholding the digest because the prose is missing loses
      // something real for no reason.
      const w = mountPanel({
        episode: { summary_text: "", summary_bullets: ["Standalone point"] },
      } as never)
      expect(w.get('[data-testid="summary-bullets"]').text()).toContain("Standalone point")
    })

    it("renders no key-points block when there are none", () => {
      const w = mountPanel({ episode: { summary_text: "Prose.", summary_bullets: [] } } as never)
      expect(w.find('[data-testid="summary-bullets"]').exists()).toBe(false)
    })

    it("renders a localized speaker-role badge on a person chip (BE.4/PL.2)", async () => {
      const w = mountPanel({
        topics: [],
        persons: [{ id: "person:jane", name: "Jane", kind: "person", role: "host" } as Entity],
      })
      // The host is not among the five shown first (they are named in the room line above).
      await w.get('[data-testid="kp-tags-more"]').trigger("click")
      const badge = w.get('[data-testid="kp-person-role"]')
      expect(badge.text()).toBe("Host") // localized via ec.roleHost, not the raw 'host'
      expect(badge.attributes("data-role")).toBe("host")
    })

    it("omits the role badge for a person with no role", () => {
      const w = mountPanel({
        topics: [],
        persons: [{ id: "person:nobody", name: "Nobody", kind: "person" } as Entity],
      })
      expect(w.find('[data-testid="kp-person-role"]').exists()).toBe(false)
    })
  })
})

describe("the people in the room, at the top of the panel", () => {
  const host = {
    id: "person:jane",
    name: "Jane Host",
    kind: "person",
    role: "host",
    image_url: "https://api.example/api/app/persons/person%3Ajane/photo",
  } as Entity
  const guest = { id: "person:bob", name: "Bob Guest", kind: "person", role: "guest" } as Entity
  const mentioned = { id: "person:ann", name: "Ann Mentioned", kind: "person", role: "mentioned" } as Entity

  it("shows host then guest, each with an avatar and a role, and leaves mentioned people out", () => {
    const w = mountPanel({ persons: [guest, mentioned, host] })
    const rows = w.findAll('[data-testid="kp-dossier-person"]')
    expect(rows).toHaveLength(2)
    expect(rows[0].text()).toContain("Jane Host")
    expect(rows[0].get(".lp-kicker").text()).toBe("Host")
    expect(rows[1].text()).toContain("Bob Guest")
    expect(rows[1].get(".lp-kicker").text()).toBe("Guest")
    // The photo when the enricher has one; initials otherwise (ProfileAvatar's own fallback).
    expect(rows[0].find("img").attributes("src")).toBe(host.image_url)
    expect(rows[1].find("img").exists()).toBe(false)
    expect(rows[1].text()).toContain("BG")
  })

  it("opens the person in the panel with a Back, the same way the person chip does", async () => {
    const getPerson = vi.spyOn(api, "getPersonCard").mockResolvedValue({
      id: "person:jane",
      label: "Jane Host",
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
    })
    const w = mountPanel({ persons: [host] })
    await w.get('[data-testid="kp-dossier-person"]').trigger("click")
    await flushPromises()
    expect(getPerson).toHaveBeenCalledWith("person:jane")
    expect(w.find('[data-testid="ec-dismiss"]').attributes("aria-label")).toBe("Back")
    expect(w.find('[data-testid="kp-episode-dossier"]').exists()).toBe(false)
  })

  it("does not make an episode-scoped guest tappable", () => {
    const scoped = { ...guest, id: "person:unresolved-bob-ep1", episode_scoped: true } as Entity
    const w = mountPanel({ persons: [scoped] })
    expect(w.get('[data-testid="kp-dossier-person"]').element.tagName).toBe("SPAN")
  })
})

describe("episode-scoped people (#1685 / #2062)", () => {
  /**
   * A guest identified only within this episode — a single-token name like "Twiggy" — used to be
   * filtered out of the API payload entirely, so the episode showed the interviewer and no guest.
   * She is now sent, flagged `episode_scoped`, because on HER OWN episode the name is not
   * under-specified at all.
   *
   * What must NOT come back with her is the tap target: there is no corpus-wide entity behind the
   * id, so opening the card would show an empty card — the exact thing #1685's filter was
   * protecting against.
   */
  const twiggy = {
    id: "person:unresolved-twiggy-ep1",
    name: "Twiggy",
    kind: "person",
    role: "guest",
    episode_scoped: true,
  } as Entity
  const host = { id: "person:lane-florsheim", name: "Lane Florsheim", kind: "person", role: "host" } as Entity

  it("renders the episode-scoped guest", () => {
    const w = mountPanel({ persons: [host, twiggy] })
    const labels = w.findAll('[data-testid="kp-person-chip"]').map((c) => c.text())
    expect(labels.some((t) => t.includes("Twiggy"))).toBe(true)
  })

  it("every chip in the mixed group names its kind, like the storyline pill does", async () => {
    const w = mountPanel({ persons: [host, twiggy] })
    await w.get('[data-testid="kp-tags-more"]').trigger("click")
    const kinds = w.findAll('[data-testid="kp-person-chip"]').map((c) => c.get('[data-testid="kp-chip-kind"]').text())
    expect(kinds).toEqual(["Person", "Person"])
  })

  it("keeps her role badge", () => {
    const w = mountPanel({ persons: [twiggy] })
    const badge = w.find('[data-testid="kp-person-role"]')
    expect(badge.exists()).toBe(true)
    expect(badge.attributes("data-role")).toBe("guest")
  })

  it("does not render her as a button", () => {
    const w = mountPanel({ persons: [twiggy] })
    const chip = w.find('[data-testid="kp-person-chip"]')
    expect(chip.element.tagName).toBe("SPAN")
    expect(chip.attributes("data-episode-scoped")).toBe("true")
  })

  it("still renders a globally-identified person as a button", async () => {
    const w = mountPanel({ persons: [host] })
    await w.get('[data-testid="kp-tags-more"]').trigger("click")
    const chip = w.find('[data-testid="kp-person-chip"]')
    expect(chip.element.tagName).toBe("BUTTON")
    expect(chip.attributes("data-episode-scoped")).toBeUndefined()
  })

  /**
   * The episode-notes export — the whole-episode document — had NO test at any layer.
   *
   * Both formats were also dead on iOS at ship time, for two different reasons: the Markdown chip
   * was `<a :download>` (ignored by WKWebView) and PDF was `window.open` (a silent no-op there).
   * `native-delivery.test.ts` greps the SOURCE for the right pattern, so it passes on a dead call
   * site or a wrong URL. These exercise the calls (operator review 2026-09-18).
   */
  it("offers ONE notes link, not a chip per format (operator 2026-10-05)", async () => {
    // Markdown and PDF side by side read as two documents; the formats live in the viewer now.
    const w = mountPanel()
    await flushPromises()
    const open = w.get('[data-testid="episode-notes-export"]')
    expect(open.text()).toBe("Download notes")
    expect(w.find('[data-testid="episode-notes-pdf"]').exists()).toBe(false)
    expect(w.find("a[download]").exists()).toBe(false)
  })

  it.each([true, false])(
    "the link OPENS the notes in the app (native=%s) — never an external tab or a save dialog",
    async (isNat) => {
      /*
       * Native: the external browser does not share the cookie jar and lands on the sign-in gate;
       * the share sheet is a SAVE dialog. Web: a tab, where "download" saved HTML, not a PDF
       * (operator 2026-10-05). Both now fetch with the app's own credentials and show the bytes.
       */
      vi.spyOn(native, "isNative").mockReturnValue(isNat)
      const external = vi.spyOn(native, "openExternal").mockResolvedValue(undefined)
      const share = vi.spyOn(native, "saveAndShareText").mockResolvedValue(undefined)
      const fetch = vi
        .spyOn(api, "fetchEpisodeNotes")
        .mockResolvedValue("<html><body>NOTES BODY</body></html>")

      const w = mountPanel()
      await flushPromises()
      await w.get('[data-testid="episode-notes-export"]').trigger("click")
      await flushPromises()

      expect(fetch).toHaveBeenCalledWith(episode().slug, "html") // THIS episode
      const frame = document.querySelector('[data-testid="export-viewer-frame"]')
      expect(frame, "the notes must be ON SCREEN").toBeTruthy()
      // `srcdoc`, not `src`: the bytes already fetched, so there is no second, cookie-less request.
      expect(frame?.getAttribute("srcdoc")).toContain("NOTES BODY")
      expect(frame?.hasAttribute("src")).toBe(false)
      // No scripts inside, ever; same-origin + modals only so the parent can print it.
      expect(frame?.getAttribute("sandbox")).toBe("allow-same-origin allow-modals")
      expect(external).not.toHaveBeenCalled()
      expect(share).not.toHaveBeenCalled()
      document.querySelector<HTMLElement>('[data-testid="export-viewer-close"]')?.click()
    },
  )

  it("search collapses identical hits and drops the keyboard", async () => {
    /*
     * Two defects from one screenshot (operator 2026-09-27). The SAME chunk came back twice,
     * filling a phone screen with room for about one result; and the keyboard covered the results,
     * which render below the input.
     *
     * The dedupe is PRESENTATION, not a fix — duplicate chunks in the index are an index-side
     * defect (same neighbourhood as #2159's chunking bug). Keyed on text, because the doc_ids
     * differ; identical ids would already have been collapsed server-side.
     */
    const hit = (doc_id: string, text: string) =>
      ({ doc_id, text, score: 1, metadata: {}, source_tier: "transcript" }) as never
    vi.spyOn(api, "searchEpisode").mockResolvedValue({
      query: "agents",
      results: [hit("a", "Agentic engineering."), hit("b", "Agentic engineering."), hit("c", "Other.")],
      error: null,
    } as never)

    const w = mountPanel()
    await flushPromises()
    const input = w.get("#kp-ask")
    const blur = vi.spyOn(input.element as HTMLInputElement, "blur")
    await input.setValue("agents")
    await w.get("form").trigger("submit")
    await flushPromises()

    const texts = w.findAll("li p").map((p) => p.text())
    expect(texts.filter((t) => t === "Agentic engineering.")).toHaveLength(1)
    expect(texts).toContain("Other.")
    expect(blur, "the keyboard must get out of the way of the results it just produced").toHaveBeenCalled()
  })

  it("the control says Search, because that is what it does", () => {
    // It was labelled "Ask" while running `searchEpisode()` and rendering raw chunks. There is no
    // synthesis endpoint in the app, so the label promised something nothing could produce.
    const w = mountPanel()
    expect(w.get("#kp-ask").attributes("placeholder")).toBe("Search this episode…")
    expect(w.text()).not.toContain("Ask this episode")
  })

  it("the viewer mounts INSIDE an open dialog, or the top layer hides it", async () => {
    /*
     * The regression this exists for (operator 2026-09-27): "on last deploy nothing happens when I
     * click PDF on insights". Something did happen — the notes fetched and the overlay rendered.
     * It was just invisible, because this panel is `showModal()`'d on mobile and therefore in the
     * TOP LAYER, which paints above the entire normal layer no matter what z-index anything there
     * carries. The viewer was teleported to `body` with `z-[60]` and sat behind the panel.
     *
     * Asserted as the TELEPORT TARGET rather than as visibility, deliberately: jsdom implements
     * neither the top layer nor `showModal` stacking, so an element hidden behind a modal is
     * indistinguishable here from one in front of it. Three tests passed over this bug for exactly
     * that reason. The target is the thing a unit test CAN see, so the target is what gets pinned.
     */
    vi.spyOn(native, "isNative").mockReturnValue(true)
    vi.spyOn(api, "fetchEpisodeNotes").mockResolvedValue("<html><body>NOTES BODY</body></html>")

    const dialog = document.createElement("dialog")
    dialog.setAttribute("open", "")
    document.body.appendChild(dialog)

    const w = mountPanel()
    await flushPromises()
    await w.get('[data-testid="episode-notes-export"]').trigger("click")
    await flushPromises()

    const frame = document.querySelector('[data-testid="export-viewer-frame"]')
    expect(frame, "the viewer did not render at all").toBeTruthy()
    expect(
      dialog.contains(frame),
      "the viewer mounted outside the open <dialog>, so on device the top layer paints over it " +
        "and the control looks dead — use sheetTeleportTarget()",
    ).toBe(true)
  })

  it("the viewer still reaches the share sheet — that is the route to Print → Save as PDF", async () => {
    vi.spyOn(native, "isNative").mockReturnValue(true)
    const share = vi.spyOn(native, "saveAndShareText").mockResolvedValue(undefined)
    vi.spyOn(api, "fetchEpisodeNotes").mockResolvedValue("<html><body>NOTES BODY</body></html>")

    const w = mountPanel()
    await flushPromises()
    await w.get('[data-testid="episode-notes-export"]').trigger("click")
    await flushPromises()

    const shareBtn = document.querySelector<HTMLElement>(
      '[data-testid="export-viewer-share"]',
    )
    expect(shareBtn).toBeTruthy()
    shareBtn!.click()
    await flushPromises()
    expect(share).toHaveBeenCalledTimes(1)
    expect(share.mock.calls[0][2]).toBe("text/html")
  })

  it("a failed export SAYS so rather than doing nothing", async () => {
    // The control it replaced failed in silence on the phone, which reads as a dead button.
    vi.spyOn(native, "isNative").mockReturnValue(true)
    vi.spyOn(api, "fetchEpisodeNotes").mockRejectedValue(new Error("offline"))

    const w = mountPanel()
    await flushPromises()
    await w.get('[data-testid="episode-notes-export"]').trigger("click")
    await flushPromises()

    expect(w.find('[data-testid="episode-notes-error"]').exists()).toBe(true)
    expect(document.querySelector('[data-testid="export-viewer-frame"]')).toBeNull()
  })

  async function openViewer() {
    vi.spyOn(api, "fetchEpisodeNotes").mockImplementation(async (_slug, ext) =>
      ext === "html" ? "<html><body>NOTES BODY</body></html>" : "# Notes\n\nbody",
    )
    const w = mountPanel()
    await flushPromises()
    await w.get('[data-testid="episode-notes-export"]').trigger("click")
    await flushPromises()
    return w
  }
  const inViewer = <T extends HTMLElement>(id: string) =>
    document.querySelector<T>(`[data-testid="export-viewer"] [data-testid="${id}"]`)

  it("on the web the viewer's Markdown is a download link named from the title", async () => {
    vi.spyOn(native, "isNative").mockReturnValue(false)
    await openViewer()
    const a = inViewer<HTMLAnchorElement>("export-viewer-md")
    expect(a?.tagName).toBe("A")
    expect(a?.getAttribute("href")).toContain("/notes.md")
    expect(a?.getAttribute("download")).toBe("ep-notes.md")
    inViewer("export-viewer-close")?.click()
  })

  it("on NATIVE the viewer's Markdown shares a file — <a download> saves nothing in WKWebView", async () => {
    vi.spyOn(native, "isNative").mockReturnValue(true)
    const share = vi.spyOn(native, "saveAndShareText").mockResolvedValue(undefined)
    await openViewer()
    const btn = inViewer("export-viewer-md")
    expect(btn?.tagName).toBe("BUTTON")
    btn!.click()
    await flushPromises()
    expect(share).toHaveBeenCalledTimes(1)
    const [filename, body] = share.mock.calls[0]
    // Named from the TITLE, not the slug (`{feed_slug}-{sha256hex}` was "some crazy name",
    // operator 2026-09-19).
    expect(filename).toBe("ep-notes.md")
    expect(filename).not.toMatch(/[0-9a-f]{8}/)
    expect(body).toContain("# Notes")
    inViewer("export-viewer-close")?.click()
  })

  it("on NATIVE Print or share hands the page to the share sheet", async () => {
    vi.spyOn(native, "isNative").mockReturnValue(true)
    const share = vi.spyOn(native, "saveAndShareText").mockResolvedValue(undefined)
    await openViewer()
    inViewer("export-viewer-share")!.click()
    await flushPromises()
    expect(share).toHaveBeenCalledTimes(1)
    expect(share.mock.calls[0][0]).toBe("ep-notes.html")
    expect(share.mock.calls[0][2]).toBe("text/html")
    inViewer("export-viewer-close")?.click()
  })

  it("on the WEB Print or share uses the browser's file share where it can", async () => {
    vi.spyOn(native, "isNative").mockReturnValue(false)
    const shareFn = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal("navigator", { ...navigator, canShare: () => true, share: shareFn })
    try {
      await openViewer()
      inViewer("export-viewer-share")!.click()
      await flushPromises()
      expect(shareFn).toHaveBeenCalledTimes(1)
      const files = shareFn.mock.calls[0][0].files as File[]
      expect(files[0].name).toBe("ep-notes.html")
      inViewer("export-viewer-close")?.click()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("on the WEB without file share, Print or share opens the print dialog — Save as PDF lives there", async () => {
    vi.spyOn(native, "isNative").mockReturnValue(false)
    vi.stubGlobal("navigator", { ...navigator, canShare: undefined, share: undefined })
    try {
      await openViewer()
      const frame = inViewer<HTMLIFrameElement>("export-viewer-frame")!
      const print = vi.fn()
      Object.defineProperty(frame, "contentWindow", { value: { print }, configurable: true })
      inViewer("export-viewer-share")!.click()
      await flushPromises()
      expect(print).toHaveBeenCalledTimes(1)
      inViewer("export-viewer-close")?.click()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  /**
   * The panel is REUSED across episodes — PlayerView passes a new `slug` prop rather than
   * unmounting — so anything held in a plain `ref` survives the change unless it is cleared.
   *
   * A filter set on episode A carried into episode B. If B had no insights of that type the user
   * saw the "Insights" heading, the chip strip, and nothing under it, with no explanation and no
   * obvious escape (review 2026-09-19).
   */
  it("clears the insight-type filter when the episode changes", async () => {
    // Episode A has claims and predictions; the user filters to predictions.
    const a = [
      insight({ id: "i1", insight_type: "claim", text: "alpha claim" }),
      insight({ id: "i2", insight_type: "prediction", text: "alpha prediction" }),
    ]
    const w = mountPanel({ insights: a, persons: [] })
    await flushPromises()

    const chips = w.findAll('[data-testid="insight-type-filter"] button')
    expect(chips.length, "expected a type filter on a multi-type episode").toBeGreaterThan(1)
    const predictionChip = chips.find((c) => c.text().toLowerCase().includes("prediction"))!
    expect(predictionChip, "no prediction chip to filter by").toBeTruthy()
    await predictionChip.trigger("click")
    await flushPromises()
    expect(w.text()).toContain("alpha prediction")
    expect(w.text()).not.toContain("alpha claim")

    // Episode B has claims and observations — NOTHING of the filtered type. Its insights must show.
    //
    // Asserting on the rendered LIST, not on chip state: with `insights: []` the chip strip does
    // not render at all (`v-if="insightTypeOptions.length > 1"`), so "no chip is active" is
    // trivially true and the test cannot fail. That vacuous version passed with the fix removed.
    const b = [
      insight({ id: "i3", insight_type: "claim", text: "bravo claim" }),
      insight({ id: "i4", insight_type: "observation", text: "bravo observation" }),
    ]
    await w.setProps({ slug: "s2", episode: { ...episode(), slug: "s2" }, insights: b })
    await flushPromises()

    expect(
      w.text(),
      "a filter from the PREVIOUS episode is still applied, so this episode's insights are hidden " +
        "behind a heading and a chip strip with no explanation",
    ).toContain("bravo claim")
    expect(w.text()).toContain("bravo observation")
  })
})
