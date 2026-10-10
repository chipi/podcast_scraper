<script setup lang="ts">
/**
 * Discover 2 — a PREVIEW of the reworked Discover (operator 2026-10-10), next to the current one
 * so the two can be compared on a device. The design and its reasoning:
 * docs/wip/discovery-competitive-analysis-2026-10-09.md and the "Discover Redesign Sketch".
 *
 * Built only from existing APIs, so some sections are approximations of the final design:
 * - "From what you captured": your newest highlights, each searched as a query; the first grounded
 *   passage from ANOTHER episode, outside the shows you follow, is the recommendation.
 * - "Trending episodes": the server's mix (GET /trending/episodes) — rising-topic takes and the
 *   most listened-to and saved, interleaved.
 * - "Continue the thread": storylines you follow × the episodes you have finished.
 * - "People you keep hearing": people from your own listening, with how many shows they are on.
 * - "Latest from shows you follow": your Library follows, newest first.
 * Every section loads on its own and hides when it has nothing. No tests: a preview to compare.
 */
import { computed, onMounted, ref, watch } from "vue"
import { RouterLink } from "vue-router"
import MomentsLink from "../components/MomentsLink.vue"
import {
  getCompleted,
  getDerivedInterests,
  getHighlightsPage,
  getLibrary,
  getPersonCard,
  getStorylineCard,
  getTrendingEpisodes,
  getUserInterests,
  listPodcastEpisodes,
  searchCorpus,
} from "../services/api"
import type { EpisodeSummary, SearchHit } from "../services/types"
import type { TrendingEpisodeCard } from "../services/api"
import { hitStartSeconds } from "../player/insights"
import { formatTime } from "../player/transcriptSync"
import { resolveMediaUrl } from "../services/tier"
import { episodeArtwork } from "../utils/episode"
import { useTrendingScope } from "../composables/useTrendingScope"
import { useAuthStore } from "../stores/auth"
import SearchSection from "../components/SearchSection.vue"
import TrendingShowsRail from "../components/TrendingShowsRail.vue"
import TrendsSection from "../components/TrendsSection.vue"
import TrendingScopeButton from "../components/TrendingScopeButton.vue"

interface QuoteCard {
  key: string
  why: string
  yours?: string
  quote: string
  speaker?: string | null
  slug: string
  episodeTitle: string
  show: string
  art: string | null
  startSeconds: number | null
}

const auth = useAuthStore()
const { scope, setScope } = useTrendingScope()
const mine = computed(() => scope.value === "mine")

const followedFeeds = ref<Set<string>>(new Set())

// ---------------------------------------------------------------- From what you captured
const captured = ref<QuoteCard[]>([])
async function loadCaptured(): Promise<void> {
  captured.value = []
  if (!auth.isAuthenticated) return
  const page = await getHighlightsPage({ limit: 4, perEpisode: 1, sort: "recent" }).catch(() => null)
  const highlights = (page?.items ?? []).filter((h) => (h.quote_text ?? "").trim().length > 20)
  const cards: QuoteCard[] = []
  for (const h of highlights.slice(0, 4)) {
    const res = await searchCorpus(h.quote_text!.slice(0, 300), 10, "all").catch(() => null)
    const hit = (res?.results ?? []).find((r: SearchHit) => {
      const md = r.metadata as Record<string, unknown>
      return md.episode_slug && md.episode_slug !== h.episode_slug && !followedFeeds.value.has(String(md.feed_id ?? ""))
    })
    if (!hit) continue
    const md = hit.metadata as Record<string, unknown>
    cards.push({
      key: `${h.id}:${hit.doc_id}`,
      why: "Because you highlighted",
      yours: h.quote_text!.length > 110 ? h.quote_text!.slice(0, 110) + "…" : h.quote_text!,
      quote: quoteOf(hit),
      speaker: (md.speaker_name as string) ?? null,
      slug: String(md.episode_slug),
      episodeTitle: String(md.episode_title ?? ""),
      show: String(md.podcast_title ?? ""),
      art: resolveArt((md.episode_artwork as string) ?? null),
      startSeconds: hitStartSeconds(hit),
    })
  }
  captured.value = cards
}
function quoteOf(hit: SearchHit): string {
  const lifted = (hit.lifted as { quote?: { text?: string } } | null)?.quote?.text
  const sq = (hit.supporting_quotes ?? [])[0] as { text?: string } | undefined
  const text = (lifted || sq?.text || hit.text || "").trim()
  return text.length > 220 ? text.slice(0, 220) + "…" : text
}

// ---------------------------------------------------------------- Trending episodes
// The server's mix (GET /trending/episodes): episodes on a rising topic, interleaved with the most
// listened-to and saved over four weeks; each card says which.
const trendingEpisodes = ref<QuoteCard[]>([])
async function loadTrendingEpisodes(): Promise<void> {
  const cards = await getTrendingEpisodes(scope.value, 8).catch(() => [] as TrendingEpisodeCard[])
  trendingEpisodes.value = cards.map((c) => ({
    key: `${c.reason}:${c.episode.slug}`,
    why:
      c.reason === "rising_topic"
        ? `↑ Rising topic · ${c.topic_label ?? ""}`
        : `Most listened to and saved · ${c.events ?? 0} this month`,
    quote: c.quote,
    speaker: c.speaker,
    slug: c.episode.slug,
    episodeTitle: c.episode.title,
    show: c.episode.podcast_title ?? "",
    art: episodeArtwork(c.episode),
    startSeconds: c.start_ms != null ? c.start_ms / 1000 : null,
  }))
}

// ---------------------------------------------------------------- Continue the thread
interface Thread {
  id: string
  label: string
  episodes: EpisodeSummary[]
  heard: Set<string>
  next: EpisodeSummary | null
  fresh: number
}
const threads = ref<Thread[]>([])
async function loadThreads(): Promise<void> {
  threads.value = []
  if (!auth.isAuthenticated) return
  const interests = await getUserInterests().catch(() => [] as string[])
  const storylines = interests.filter((t) => t.startsWith("thc:")).slice(0, 3)
  if (!storylines.length) return
  const done = new Set(await getCompleted().catch(() => [] as string[]))
  const out: Thread[] = []
  for (const id of storylines) {
    const card = await getStorylineCard(id, { limit: 30 }).catch(() => null)
    if (!card?.episodes.length) continue
    const ordered = [...card.episodes].sort((a, b) => (a.publish_date ?? "").localeCompare(b.publish_date ?? ""))
    const heard = new Set(ordered.filter((e) => done.has(e.slug)).map((e) => e.slug))
    const lastHeardIdx = ordered.reduce((acc, e, i) => (heard.has(e.slug) ? i : acc), -1)
    const fresh = ordered.slice(lastHeardIdx + 1).length
    out.push({
      id,
      label: card.label,
      episodes: ordered.slice(-10),
      heard,
      next: ordered.find((e) => !heard.has(e.slug)) ?? null,
      fresh,
    })
  }
  threads.value = out
}

// ---------------------------------------------------------------- People you keep hearing
interface PersonRow {
  id: string
  label: string
  count: number
  shows: number
  image: string | null
}
const people = ref<PersonRow[]>([])
async function loadPeople(): Promise<void> {
  people.value = []
  if (!auth.isAuthenticated) return
  const derived = await getDerivedInterests().catch(() => [])
  const top = derived.filter((d) => d.kind === "person" && d.count >= 2).slice(0, 6)
  const rows: PersonRow[] = []
  for (const d of top) {
    const card = await getPersonCard(d.token, undefined, { limit: 1 }).catch(() => null)
    rows.push({
      id: d.token,
      label: d.label,
      count: d.count,
      shows: card?.shows?.length ?? 0,
      image: (card as { image_url?: string | null } | null)?.image_url ?? null,
    })
  }
  people.value = rows
}

// ---------------------------------------------------------------- Latest from shows you follow
const latest = ref<EpisodeSummary[]>([])
const latestTotal = ref(0)
async function loadLatest(): Promise<void> {
  latest.value = []
  if (!auth.isAuthenticated) return
  const feeds = [...followedFeeds.value].slice(0, 25)
  const pages = await Promise.all(feeds.map((f) => listPodcastEpisodes(f, { pageSize: 3 }).catch(() => null)))
  const all = pages.flatMap((p) => p?.items ?? [])
  all.sort((a, b) => (b.publish_date ?? "").localeCompare(a.publish_date ?? ""))
  latestTotal.value = all.length
  latest.value = all.slice(0, 4)
}

function resolveArt(url: string | null): string | null {
  return url ? resolveMediaUrl(url) : null
}
function playTo(card: QuoteCard): Record<string, unknown> {
  return {
    name: "player",
    params: { slug: card.slug },
    query: card.startSeconds != null ? { t: String(Math.floor(card.startSeconds)) } : {},
  }
}
function dateLabel(iso: string | null): string {
  if (!iso) return ""
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

async function loadAll(): Promise<void> {
  if (auth.isAuthenticated) {
    const lib = await getLibrary().catch(() => [])
    followedFeeds.value = new Set(lib.map((l) => l.feed_id))
  }
  void loadCaptured()
  void loadTrendingEpisodes()
  void loadThreads()
  void loadPeople()
  void loadLatest()
}
onMounted(loadAll)
watch(scope, () => void loadTrendingEpisodes())
</script>

<template>
  <section class="lp-page pb-8" data-testid="discover2-view">
    <div class="mb-1 flex items-center justify-between gap-3">
      <h1 class="font-display text-3xl font-extrabold tracking-tight">Discover 2</h1>
      <TrendingScopeButton testid="discover2-scope" />
    </div>
    <p class="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
      <span class="rounded-full border border-border px-2 py-0.5 font-mono uppercase tracking-wider">Preview</span>
      <span>{{ mine ? "Mine · your world" : "Everyone" }}</span>
      <RouterLink :to="{ name: 'browse' }" class="font-bold text-accent">Compare with Discover ›</RouterLink>
    </p>

    <SearchSection prefix="browse" />

    <!-- From what you captured -->
    <section v-if="mine && captured.length" class="mt-7" data-testid="d2-captured">
      <h2 class="mb-3 text-lg font-bold">From what you captured</h2>
      <div class="d2-rail">
        <article v-for="c in captured" :key="c.key" class="d2-card">
          <p class="font-mono text-[11px] tracking-wide text-accent">{{ c.why }}</p>
          <p class="border-l-2 border-border pl-2 text-xs italic text-muted">“{{ c.yours }}”</p>
          <p class="d2-quote">“{{ c.quote }}”</p>
          <p v-if="c.speaker" class="text-xs text-muted">{{ c.speaker }}</p>
          <div class="flex min-w-0 items-center gap-2">
            <img v-if="c.art" :src="c.art" alt="" class="h-9 w-9 shrink-0 rounded-md object-cover" />
            <div class="min-w-0">
              <p class="truncate text-sm font-bold">{{ c.episodeTitle }}</p>
              <p class="font-mono text-[10.5px] uppercase tracking-wider text-muted">{{ c.show }}</p>
            </div>
          </div>
          <RouterLink :to="playTo(c)" class="d2-play">▶ {{ c.startSeconds != null ? formatTime(c.startSeconds) : "Play" }}</RouterLink>
        </article>
      </div>
    </section>

    <!-- Trending episodes -->
    <section v-if="trendingEpisodes.length" class="mt-7" data-testid="d2-trending-episodes">
      <h2 class="mb-1 text-lg font-bold">Trending episodes</h2>
      <p class="mb-3 text-xs text-muted">Rising topics and the most listened to and saved, {{ mine ? "in your world" : "across everyone" }}.</p>
      <div class="d2-rail">
        <article v-for="c in trendingEpisodes" :key="c.key" class="d2-card">
          <p class="font-mono text-[11px] tracking-wide text-accent">{{ c.why }}</p>
          <p class="d2-quote">“{{ c.quote }}”</p>
          <p v-if="c.speaker" class="text-xs text-muted">{{ c.speaker }}</p>
          <div class="flex min-w-0 items-center gap-2">
            <img v-if="c.art" :src="c.art" alt="" class="h-9 w-9 shrink-0 rounded-md object-cover" />
            <div class="min-w-0">
              <p class="truncate text-sm font-bold">{{ c.episodeTitle }}</p>
              <p class="font-mono text-[10.5px] uppercase tracking-wider text-muted">{{ c.show }}</p>
            </div>
          </div>
          <!-- The pill plays the quoted moment; "▶ Moments" plays the whole reel (2026-10-10). -->
          <div class="flex items-center justify-between gap-2">
            <RouterLink :to="playTo(c)" class="d2-play">▶ {{ c.startSeconds != null ? formatTime(c.startSeconds) : "Play" }}</RouterLink>
            <MomentsLink :slug="c.slug" />
          </div>
        </article>
      </div>
    </section>

    <!-- Continue the thread -->
    <section v-if="threads.length" class="mt-7" data-testid="d2-threads">
      <h2 class="mb-3 text-lg font-bold">Continue the thread</h2>
      <div class="grid gap-3">
        <article v-for="th in threads" :key="th.id" class="d2-card">
          <p class="font-mono text-[10.5px] uppercase tracking-widest text-muted">Storyline you follow</p>
          <p class="text-base font-bold">{{ th.label }}</p>
          <div class="flex flex-wrap items-center gap-1.5">
            <span
              v-for="e in th.episodes"
              :key="e.slug"
              class="inline-block h-2.5 w-2.5 rounded-full"
              :class="th.heard.has(e.slug) ? 'bg-muted' : 'bg-accent'"
              :title="e.title"
            />
            <span class="ml-1 font-mono text-[11px] text-muted">{{ th.fresh }} not heard yet</span>
          </div>
          <RouterLink v-if="th.next" :to="{ name: 'player', params: { slug: th.next.slug } }" class="d2-play">▶ Next: {{ th.next.title }}</RouterLink>
        </article>
      </div>
    </section>

    <!-- People you keep hearing -->
    <section v-if="people.length" class="mt-7" data-testid="d2-people">
      <h2 class="mb-3 text-lg font-bold">People you keep hearing</h2>
      <div class="flex gap-4 overflow-x-auto pb-2">
        <RouterLink
          v-for="p in people"
          :key="p.id"
          :to="{ name: 'person', params: { id: p.id } }"
          class="grid w-24 shrink-0 justify-items-center gap-1 text-center"
        >
          <img v-if="p.image" :src="resolveArt(p.image) ?? ''" alt="" class="h-14 w-14 rounded-full object-cover" />
          <span v-else class="grid h-14 w-14 place-items-center rounded-full border border-border bg-surface text-lg font-bold text-accent">{{ p.label.slice(0, 2) }}</span>
          <span class="text-sm font-bold leading-tight">{{ p.label }}</span>
          <span class="font-mono text-[10.5px] leading-tight text-muted">in {{ p.count }} of yours<template v-if="p.shows"> · on {{ p.shows }} show{{ p.shows === 1 ? "" : "s" }}</template></span>
        </RouterLink>
      </div>
    </section>

    <!-- Latest from shows you follow -->
    <section v-if="latest.length" class="mt-7" data-testid="d2-latest">
      <div class="mb-3 flex items-baseline justify-between gap-3">
        <h2 class="text-lg font-bold">Latest from shows you follow</h2>
        <RouterLink :to="{ name: 'browse', query: { tab: 'episodes', from: 'following', state: 'unplayed' }, hash: '#catalog' }" class="text-sm font-bold text-accent">all ›</RouterLink>
      </div>
      <ul class="grid">
        <li v-for="e in latest" :key="e.slug" class="border-b border-border last:border-b-0">
          <RouterLink :to="{ name: 'player', params: { slug: e.slug } }" class="flex min-w-0 items-center gap-3 py-2">
            <img v-if="episodeArtwork(e)" :src="episodeArtwork(e) ?? ''" alt="" class="h-10 w-10 shrink-0 rounded-md object-cover" />
            <span class="min-w-0 flex-1">
              <span class="block truncate text-sm font-bold">{{ e.title }}</span>
              <span class="block font-mono text-[10.5px] uppercase tracking-wider text-muted">{{ e.podcast_title }} · {{ dateLabel(e.publish_date) }}</span>
            </span>
          </RouterLink>
        </li>
      </ul>
    </section>

    <TrendingShowsRail :title="'Trending shows'" :scope="scope" :top="5" @show-everyone="setScope('corpus')" />

    <div class="lg:w-1/2 lg:pr-4">
      <TrendsSection />
    </div>
  </section>
</template>

<style scoped>
.d2-rail {
  display: flex;
  gap: 0.75rem;
  overflow-x: auto;
  padding-bottom: 0.5rem;
  scroll-snap-type: x mandatory;
}
.d2-rail > .d2-card {
  flex: 0 0 82%;
  scroll-snap-align: start;
}
@media (min-width: 640px) {
  .d2-rail > .d2-card { flex-basis: 20rem; }
}
.d2-card {
  display: grid;
  gap: 0.5rem;
  align-content: start;
  min-width: 0;
  border: 1px solid var(--lp-border);
  border-radius: 14px;
  padding: 0.75rem;
  background: var(--lp-surface);
}
.d2-quote {
  font-family: Georgia, "Iowan Old Style", serif;
  font-size: 15px;
  line-height: 1.45;
}
.d2-play {
  justify-self: start;
  border-radius: 999px;
  background: var(--lp-accent);
  color: var(--lp-accent-foreground);
  padding: 0.3rem 0.8rem;
  font-family: var(--lp-font-mono, ui-monospace, monospace);
  font-size: 13px;
  font-weight: 700;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
