import { mount } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createPinia, setActivePinia } from "pinia"
import { createI18n } from "vue-i18n"
import { createMemoryHistory, createRouter } from "vue-router"
import * as api from "../services/api"
import en from "../i18n/locales/en.json"
import type { EpisodeSummary, TopicCard } from "../services/types"
import TopicCardContent from "./TopicCardContent.vue"

const i18n = createI18n({ legacy: false, locale: "en", messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: "/", name: "home", component: { template: "<div/>" } },
    { path: "/search", name: "search", component: { template: "<div/>" } },
    { path: "/topic/:id", name: "topic", component: { template: "<div/>" }, props: true },
    { path: "/person/:id", name: "person", component: { template: "<div/>" }, props: true },
    { path: "/podcast/:feedId", name: "podcast", component: { template: "<div/>" }, props: true },
    { path: "/episode/:slug", name: "player", component: { template: "<div/>" }, props: true },
    { path: "/storyline/:id", name: "storyline", component: { template: "<div/>" }, props: true },
  ],
})

// Heavy children with their own data/links aren't under test here — the subject is the activity
// sparkline. Stub them so the mount stays focused (Sparkline + TrendMomentum stay real).
const childStubs = {
  EpisodeRow: true,
  StorylineCard: true,
  NoteComposer: true,
  TopicPerspectives: true,
  TopicConversationArc: true,
  ProfileAvatar: true,
}

beforeEach(() => {
  setActivePinia(createPinia())
  // Trending is fetched by useTrendingIndex on mount; return empty so these topics are NON-trending
  // (no "Rising" pill) — the point is the activity line shows anyway.
  vi.spyOn(api, "getTrending").mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

function ep(slug: string, publish_date: string | null): EpisodeSummary {
  return {
    slug,
    title: slug,
    feed_id: "f1",
    podcast_title: "Show",
    publish_date,
    duration_seconds: 60,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    status: "ready",
    summary_preview: null,
    summary_bullets: [],
    topics: [],
    has_transcript: true,
    has_summary: false,
    has_gi: false,
    has_kg: false,
    has_bridge: false,
  } as EpisodeSummary
}

function topic(over: Partial<TopicCard> = {}): TopicCard {
  return {
    id: "t1",
    label: "Interest Rates",
    cluster_id: null,
    cluster_label: null,
    cluster_size: 0,
    sibling_topics: [],
    episode_count: 0,
    episodes: [],
    related_people: [],
    ...over,
  } as TopicCard
}

const mountIt = (t: TopicCard) =>
  mount(TopicCardContent, {
    props: { topic: t },
    global: { plugins: [i18n, router], stubs: childStubs },
  })

describe("TopicCardContent — universal activity sparkline (operator 2026-09-14)", () => {
  it("shows a 'discussed over time' sparkline for a NON-trending topic, from its own episodes", () => {
    const w = mountIt(
      topic({
        episodes: [ep("a", "2026-01-10"), ep("b", "2026-02-14"), ep("c", "2026-04-02")],
      })
    )
    // No trending data → no "Rising" pill, but the activity line still renders.
    expect(w.find('[data-testid="ec-topic-momentum"]').exists()).toBe(false)
    expect(w.find('[data-testid="ec-topic-activity"]').exists()).toBe(true)
    expect(w.find('[data-testid="sparkline"]').exists()).toBe(true)
    expect(w.text()).toContain("Discussed over time")
  })

  it("omits the activity line when the topic spans a single month (a flat bar is not a trend)", () => {
    const w = mountIt(topic({ episodes: [ep("a", "2026-03-01"), ep("b", "2026-03-20")] }))
    expect(w.find('[data-testid="ec-topic-activity"]').exists()).toBe(false)
  })

  it("omits the activity line when there are too few dated episodes", () => {
    const w = mountIt(topic({ episodes: [ep("a", null), ep("b", "2026-03-20")] }))
    expect(w.find('[data-testid="ec-topic-activity"]').exists()).toBe(false)
  })
})
