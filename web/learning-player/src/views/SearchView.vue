<script setup lang="ts">
/**
 * Corpus-wide grounded search (PRD-042 FR5 / RFC-099 §Home). Searches the whole library via
 * GET /api/app/search (extractive, no request-time LLM). Rather than a flat wall of mixed
 * passages, results are **grouped by episode** (ranked by their best hit) and each passage is
 * labelled by kind (Insight / Transcript / Topic). A "Play from …" jump appears only when the
 * passage carries a real timestamp — otherwise we open the episode rather than fake a 0:00.
 */
import { computed, ref, watch } from "vue"
import { toCountBucket, toRankBucket, track } from "../services/analytics"
import { useI18n } from "vue-i18n"
import { noteRoute as resolveNoteRoute, noteTargetLabel } from "../composables/noteTarget"
defineOptions({ name: "SearchView" }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRoute, useRouter } from "vue-router"
import {
  getPodcastsPage,
  resolveEntity,
  searchBoards,
  searchCorpus,
  searchInterests,
} from "../services/api"
import { useCorpusLanguages } from "../composables/useCorpusLanguages"
import { resolveMediaUrl } from "../services/tier"
import type {
  CollectionMatch,
  EntityRef,
  EpisodeSummary,
  Note,
  Podcast,
  SearchHit,
} from "../services/types"
import { hitStartSeconds } from "../player/insights"
import { formatTime } from "../player/transcriptSync"
import { formatPublishDate } from "../utils/format"
import { aggregateRelatedTopics } from "../utils/relatedTopics"
import {
  collapseFoldableHits,
  isFoldedCluster,
  type CollapsedRow,
  type FoldedHitCluster,
} from "../utils/collapseFoldableHits"
import { summarizeMatchedFields } from "../utils/matchedFields"
import { groupEpisodesByYear, type YearSection } from "../utils/yearGrouping"
import { useSignInGate } from "../composables/useSignInGate"
import { useTrendingScope } from "../composables/useTrendingScope"
import { useSavedQueriesStore } from "../stores/savedQueries"
import { useUserPreferencesStore } from "../stores/userPreferences"
import { kindPill } from "../utils/interests"
import { useCaptureStore } from "../stores/capture"
import EntityCard from "../components/EntityCard.vue"
import ShowAllToggle from "../components/ShowAllToggle.vue"
import { useNotesPage } from "../composables/useNotesPage"
import EpisodeGroupCard from "../components/EpisodeGroupCard.vue"
import PlayFrom from "../components/PlayFrom.vue"
import AddToCollectionButton from "../components/AddToCollectionButton.vue"
import SectionStatus from "../components/SectionStatus.vue"
import TypeFilterBar from "../components/TypeFilterBar.vue"
import ShowRow from "../components/ShowRow.vue"
import ShowMenu from "../components/ShowMenu.vue"

const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const { isGated, gated } = useSignInGate()
const savedQueries = useSavedQueriesStore()
const capture = useCaptureStore()
// Shared rule (composables/noteTarget) — was a second, drifted copy of the same function.
const noteRoute = (target: string, id: string) => resolveNoteRoute(target, id, capture.highlights)
/**
 * Episode and show titles for the note rows, built from the hits ALREADY on screen.
 *
 * Every hit's metadata carries `episode_slug`/`episode_title` and `feed_id`/`feed_title`, so a note
 * about something the search also matched gets its title for free. Deliberately no extra fetch: the
 * note list is a secondary section, and a note pointing at an episode outside these results simply
 * shows no subtitle rather than costing a request per note.
 */
const hitTitles = computed(() => {
  const episodes = new Map<string, string>()
  const shows = new Map<string, string>()
  for (const h of results.value) {
    const m = h.metadata as Record<string, unknown>
    const slug = m.episode_slug as string | undefined
    const epTitle = m.episode_title as string | undefined
    if (slug && epTitle) episodes.set(slug, epTitle)
    const feed = m.feed_id as string | undefined
    const feedTitle = (m.feed_title ?? m.podcast_title) as string | undefined
    if (feed && feedTitle) shows.set(feed, feedTitle)
  }
  return { episodes, shows }
})
const noteLabel = (target: string, id: string) =>
  noteTargetLabel(target, id, hitTitles.value, capture.highlights)
// SR.1 — search the listener's OWN notes alongside the corpus. Notes are per-user and client-side,
// so this is a local text match, shown as its own "Your notes" section rather than interleaved with
// the corpus passages (a note is not a transcript hit).
/** The term the results on screen were found for — the box can be edited without re-running. */
const ranTerm = ref("")

/**
 * SHOWS matching the search (operator 2026-10-04: search should include shows wherever there is a
 * search). The corpus search returns passages, never a show, so "pragmatic engineer" found the
 * show's episodes but not the show — while Library's own filter did find it. Matched client-side
 * over the catalogue, by title and host names (every word, any order — utils/textMatch). Not by
 * description: a word like "engineer" is in many shows' blurbs, and that would bury the one show
 * the listener named.
 */
/** The matching shows — asked of the server per search since 2026-10-08 (same every-word rule). */
const showMatches = ref<Podcast[]>([])
const { languageOfFeed } = useCorpusLanguages()

// USERPREFS-1 hydrate fires once at app init in main.ts; the savedQueries
// watch reacts when the payload arrives so the Save button flips to
// "Saved ✓" if the current query was already persisted. No per-view
// hydrate call — the tests demonstrated that adding one creates
// test-order flakiness without changing user-visible behaviour.

// Scope (P3 Recall, #1124): 'all' = whole library; 'mine' = the listener's own world (heard,
// captured, saved, followed shows and interests; ADR-162). The toggle only shows when signed in.
// One "mine" switch everywhere (operator 2026-10-09): with no `scope` in the URL the results open in
// the shared Mine ⇄ Everyone choice, and flipping it here flips it on Discover too.
const { scope: sharedScope, setScope: setSharedScope } = useTrendingScope()
const scope = ref<"all" | "mine">(
  route.query.scope === "mine" || route.query.scope === "all"
    ? route.query.scope
    : sharedScope.value === "mine"
      ? "mine"
      : "all"
)

const query = ref(String(route.query.q ?? ""))

// #1261-8: reactive save-state on the current query+scope pair so the button
// toggles between "Save" and "Saved ✓" as the listener switches scope tabs.
const currentIsSaved = computed(() => savedQueries.isSaved(query.value, scope.value))

// Brief "Saved — see Library" confirmation; saving is otherwise silent, so a listener had no sign
// it worked or where it went.
const saveMsg = ref("")
let saveMsgTimer: ReturnType<typeof setTimeout> | undefined
async function toggleSaveQuery(): Promise<void> {
  const q = query.value.trim()
  if (!q) return
  if (savedQueries.isSaved(q, scope.value)) {
    await savedQueries.remove(q, scope.value)
    saveMsg.value = ""
  } else {
    await savedQueries.save(q, scope.value)
    saveMsg.value = t("search.savedConfirm")
    if (saveMsgTimer) clearTimeout(saveMsgTimer)
    saveMsgTimer = setTimeout(() => (saveMsg.value = ""), 3000)
  }
}
// Saving is per-account → gate it: signed-out taps route to sign-in instead of a silent no-op that
// looked saved but never persisted. gated(fn) must be a stable handler (calling gated() inline in
// @click discards the returned handler).
const onSaveClick = gated(toggleSaveQuery)
const results = ref<SearchHit[]>([])
/**
 * Category mode (operator 2026-10-13): each kind of result is its own capped search, three each,
 * with "Show more" under a kind's last item as on the rails. A topic row exists per episode per
 * topic, so "risk" filled every slot with topic rows — which are listed by name above and dropped
 * from the episodes — and the search showed no episode at all.
 */
const PER_CATEGORY = 3
const CATEGORY_ORDER = ["insights", "transcript", "episodes", "mentions"] as const
/** Whether each category has more than is shown (from the server's `categories`). */
const categoryMore = ref<Record<string, boolean>>({})
const categoryLoading = ref<string | null>(null)
const entity = ref<EntityRef | null>(null)
const cardTarget = ref<{ kind: "person" | "topic" | "organization"; id: string } | null>(null)
const searching = ref(false)
const error = ref(false)
const ran = ref(false)
// Shows for the term that RAN, from the server (`q`: every word over title and hosts).
let showSeq = 0
watch(
  () => (ran.value ? ranTerm.value.trim() : ""),
  async (term) => {
    const mine = ++showSeq
    if (!term) {
      showMatches.value = []
      return
    }
    const page = await getPodcastsPage({ q: term, limit: 50, sort: "az" }).catch(() => null)
    if (mine === showSeq) showMatches.value = page?.items ?? []
  },
  { immediate: true },
)
// The listener's own boards for the term that ran (#2341): by name first, then by what is saved on
// them, every word in any order (the shows' rule). Signed out there are none; a failure leaves the
// section empty and never fails the search.
const boardMatches = ref<CollectionMatch[]>([])
let boardSeq = 0
watch(
  () => (ran.value && !isGated.value ? ranTerm.value.trim() : ""),
  async (term) => {
    const mine = ++boardSeq
    if (!term) {
      boardMatches.value = []
      return
    }
    const items = await searchBoards(term).catch(() => [])
    if (mine === boardSeq) boardMatches.value = items
  },
  { immediate: true },
)
function boardWhy(m: CollectionMatch): string {
  if (m.matched === "name" || !m.item_kind) return t("search.boardMatchedName")
  return t("search.boardMatchedItem", {
    kind: t(`collections.kind.${m.item_kind}`),
    label: m.item_label ?? "",
  })
}
function openBoard(rank: number): void {
  track("search_result_click", { result_kind: "board", rank: toRankBucket(rank) })
}
// Matched on the SERVER since 2026-10-08 (every word, any order — the same rule as before), five
// at a time, newest first: the store no longer holds every note to search through.
const notesTerm = computed(() => (ran.value ? ranTerm.value.trim() : ""))
const notesPage = useNotesPage(notesTerm, ref<string[]>([]), {
  words: true,
  enabled: () => notesTerm.value !== "",
  debounceMs: 0,
})
const noteMatches = computed<Note[]>(() => (notesTerm.value ? notesPage.items.value : []))
// The `term::scope` currently on screen, so a kept-alive re-entry can tell "same results" from a
// genuinely new query and skip a redundant re-fetch.
const lastRunSig = ref("")
// The `term::scope` of a search that is STILL resolving. `lastRunSig` only settles in `finally`, so
// navigating away and back while a search is in flight would otherwise fire a second identical
// fetch (`ran` not yet true) — this guards that concurrent duplicate without blocking a retry once
// the first has settled.
let inFlightSig = ""
// Monotonic run counter — see the generation guard in `run()`.
let runSeq = 0
// The term of the search currently live/on screen (NOT the input box, which may hold an unsubmitted
// edit) — the tab-return restore rebuilds `?q` from THIS so it never fires a fetch the user didn't ask
// for.
let lastRunTerm = ""

type Kind = "insight" | "transcript" | "topic" | "passage"
interface EpisodeGroup {
  slug: string | null
  title: string
  show: string | null
  date: string | null
  art: string | null
  hits: SearchHit[]
  /** #1261-3: hits collapsed so multiple same-kind foldable rows (transcript,
   *  title, description, summary) render as one expandable summary row. */
  rows: CollapsedRow[]
}

/** An episode result offers its Moments reel when the episode has insights (operator 2026-10-10). */
function hasMoments(g: EpisodeGroup): boolean {
  return !!g.slug && g.hits.some((h) => (h.metadata as Record<string, unknown> | undefined)?.episode_has_moments === true)
}

/**
 * An `EpisodeGroup` as the `EpisodeSummary` that `EpisodeCard` takes.
 *
 * Search renders the SAME card as Discover's and Library's episode lists rather than a
 * near-identical hand-rolled one (operator 2026-09-17). The previous markup had drifted to its own
 * artwork radius, title size, centred action row and bordered container — recognisably a different
 * component for the same object.
 *
 * `artwork_url` is the already-resolved `g.art` (search absolutises it when grouping, so it works on
 * native). No summary fields: a search group carries passages, not a summary, so the card renders no
 * prose and offers no "Read more" — the match context goes in the card's `#meta` and `#aside` slots
 * instead. `status: 'ready'` because a hit can only exist for an indexed episode.
 */
function groupAsEpisode(g: EpisodeGroup): EpisodeSummary {
  const feedId = (g.hits[0]?.metadata as Record<string, unknown> | undefined)?.feed_id ?? null
  return {
    slug: g.slug ?? "",
    title: g.title,
    podcast_title: g.show,
    // REQUIRED, not decorative: the card's show-name kicker is a RouterLink to
    // `{ name: 'podcast', params: { feedId } }`, and vue-router THROWS on resolving that route with
    // an empty param. Search never tracked the feed id, so it comes off the group's first hit —
    // every hit in a group belongs to one episode, hence to one feed.
    feed_id: feedId,
    language: groupLanguage(g, feedId),
    publish_date: g.date,
    artwork_url: g.art,
    episode_image_url: null,
    feed_image_url: null,
    duration_seconds: null,
    status: "ready",
  } as unknown as EpisodeSummary
}

/**
 * The language for a search group's card badge (V2-C.1). A hit's own `language` is stamped only
 * on SOURCE-layer chunks of a translated episode — the English analysis chunks and every native
 * English chunk carry none (hybrid_search never defaults it) — so hits alone would badge some
 * Spanish episodes and no English ones. A source-layer hit is the episode's own evidence and wins;
 * otherwise the show's language from the catalogue, which is also what the API falls back to for an
 * episode with no language of its own.
 */
function groupLanguage(g: EpisodeGroup, feedId: unknown): string | null {
  const fromHit = g.hits.map((h) => md(h).language).find((l): l is string => typeof l === "string" && !!l)
  if (fromHit) return fromHit
  return languageOfFeed(feedId)
}

const md = (h: SearchHit) => h.metadata as Record<string, unknown>
const hitSlug = (h: SearchHit) => (md(h).episode_slug as string | undefined) ?? null
const hitEpisode = (h: SearchHit) => (md(h).episode_title as string | undefined) ?? null
const hitShow = (h: SearchHit) => (md(h).podcast_title as string | undefined) ?? null
const hitDate = (h: SearchHit) => (md(h).publish_date as string | undefined) ?? null
// Absolutised: search-hit metadata carries the same relative artwork url as the catalog.
const hitArt = (h: SearchHit) => resolveMediaUrl(md(h).episode_artwork as string | undefined)

function hitKind(h: SearchHit): Kind {
  const dt = md(h).doc_type
  if (dt === "insight") return "insight"
  if (dt === "transcript") return "transcript"
  if (dt === "kg_topic") return "topic"
  return "passage"
}

// #1261-2: aggregate per-hit related_topics into a chip row above the episode
// groups. Tapping a chip opens the Topic EntityCard modal — deeper exploration
// without expanding the search surface itself.
const relatedTopicChips = computed(() => aggregateRelatedTopics(results.value, 8))

function openTopicChip(topicId: string): void {
  // Two events, because they answer different questions (#2267). `search_result_click` says WHICH
  // KIND of result search surfaced usefully — the related-topic chips exist precisely to turn a
  // text search into a pivot, and whether anyone takes them is the thing to find out.
  // `entity_open` is the pivot itself and feeds Pivot rate.
  //
  // `rank: '1'` rather than a position: the chip row is a short unranked set, not a results list,
  // and inventing an ordinal would make a bucket mean two different things across events.
  track("search_result_click", { result_kind: "topic", rank: "1" })
  track("entity_open", { kind: "topic", presentation: "card", source: "search" })
  cardTarget.value = { kind: "topic", id: topicId }
}

// #1261-5: shim so the template's inline v-for can call the helper by name.
// Wrapped so summarizeMatchedFields can be swapped in unit tests independently.
function matchedFieldChips(hits: SearchHit[]) {
  return summarizeMatchedFields(hits)
}

/** Every hit of a group's episode, across the category sections: "Matched: Title · Transcript"
 * says why the EPISODE surfaced, not why it is in this one section. */
function episodeHits(g: EpisodeGroup): SearchHit[] {
  return g.slug ? passageResults.value.filter((h) => hitSlug(h) === g.slug) : g.hits
}

// #1261-7: bucket the episode groups by publish year so the results read as
// "2026 · 4 episodes" / "2025 · 12 episodes" sections — the mobile-friendly
// reshape of the operator viewer's timeline chart concept. Sections are only
// shown when the search spans multiple years; single-year results skip the
// header to keep the page short.
const yearSections = computed<YearSection<EpisodeGroup>[]>(() =>
  groupEpisodesByYear<EpisodeGroup>(groups.value, (g) => g.date)
)

function yearLabel(year: number | "unknown"): string {
  return year === "unknown" ? t("search.yearUnknown") : String(year)
}

// Type-safe key for the collapsed-rows v-for: plain hits carry doc_id,
// folded clusters use their foldedKind + index (unique within a group).
function rowKey(row: CollapsedRow, i: number): string {
  return isFoldedCluster(row) ? `c:${row.foldedKind}:${i}` : `${row.doc_id}:${i}`
}

// Group passages under their source episode, preserving rank order (results arrive best-first,
// so an episode's rank is its first appearance).
/**
 * A storyline, topic or person hit names something ACROSS episodes, not a passage of one (operator
 * 2026-10-11): grouped under the episode it was indexed from, "TOPIC · systems thinking
 * integration" read as a fact about that episode. They are lifted out, like storylines, into their
 * own lists at the top. Organizations stay where they were (they have no page or list here).
 */
function isLiftedEntityHit(h: SearchHit): boolean {
  const m = md(h)
  if (m.doc_type === "kg_topic") return true
  return m.doc_type === "kg_entity" && String(m.source_id ?? "").startsWith("person:")
}
/** Everything the episode grouping should see — passages only. */
const passageResults = computed(() =>
  results.value.filter((h) => md(h).doc_type !== "storyline" && !isLiftedEntityHit(h))
)

type SubjectKind = "person" | "topic" | "theme" | "storyline"
type EntityRow = { key: string; id: string; label: string; routeId: string }

/**
 * The subjects at the top are matched BY NAME, on the words typed (operator 2026-10-12: "I want a
 * word match … exact matches"). They used to come from the passage search's own hits, which rank by
 * meaning as well as words, so "safety" listed "Second-order effects" and "incident prevention" —
 * close in meaning, without the word. Episodes keep word + meaning; this section is names you open,
 * and a name either has the word or it does not.
 *
 * Rule: every word typed starts a word of the name ("risk" finds "risk management" and "discounted
 * risks"; "art" does not find "start"). Fetched per kind from the followables' name search
 * (`/interests/search`), corpus-wide in both scopes.
 */
const nameMatches = ref<Record<SubjectKind, { id: string; label: string; anchor_topic_id?: string | null }[]>>({
  person: [],
  topic: [],
  theme: [],
  storyline: [],
})
function words(text: string): string[] {
  return text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean)
}
function nameHasWords(label: string, typed: string[]): boolean {
  const own = words(label)
  return typed.length > 0 && typed.every((w) => own.some((o) => o.startsWith(w)))
}
/** One kind's rows: the resolver's exact-name answer first, then the name matches, one per subject. */
function subjectRows(kind: SubjectKind): EntityRow[] {
  const typed = words(ranTerm.value)
  const out: EntityRow[] = []
  const seen = new Set<string>()
  const add = (id: string, label: string, routeId: string): void => {
    const k = label.trim().toLowerCase()
    if (!label.trim() || seen.has(id) || seen.has(k)) return
    seen.add(id)
    seen.add(k)
    out.push({ key: id, id, label: label.trim(), routeId })
  }
  const e = entity.value
  // The resolver answers a storyline with its ANCHOR topic id, which is what the storyline page routes on.
  if (e?.kind === kind) add(e.id, e.label, e.id)
  for (const m of nameMatches.value[kind]) {
    if (!nameHasWords(m.label, typed)) continue
    // A storyline routes on its anchor topic (there is no `thc:` page); without one it cannot open.
    const routeId = kind === "storyline" ? String(m.anchor_topic_id ?? "") : m.id
    if (!routeId) continue
    add(m.id, m.label, routeId)
  }
  return out
}
const peopleResults = computed(() => subjectRows("person"))
const topicResults = computed(() => subjectRows("topic"))
const themeResults = computed(() => subjectRows("theme"))
const storylineResults = computed(() => subjectRows("storyline"))
/** A long list folds behind "Show all" (a broad word like "risk" reaches twenty topics). */
const ENTITY_ROWS_SHOWN = 8
const expandedEntityLists = ref(new Set<string>())
function shownRows<T>(key: string, rows: T[]): T[] {
  return expandedEntityLists.value.has(key) ? rows : rows.slice(0, ENTITY_ROWS_SHOWN)
}
function toggleEntityList(key: string): void {
  const next = new Set(expandedEntityLists.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  expandedEntityLists.value = next
}
/**
 * Every subject shown ONE way (operator 2026-10-11): a section per kind — People, Topics, Themes,
 * Storylines — each the same kind-coloured pills. A person or topic opens its card; a theme or
 * storyline its page (neither has a card). Empty kinds are left out.
 */
type SubjectPill = { key: string; label: string; open: { card: { kind: "person" | "topic"; id: string } } | { to: { name: string; params: { id: string } } } }
const subjectSections = computed<{ key: string; kind: SubjectKind; rows: SubjectPill[] }[]>(() => {
  return [
    { key: "people", kind: "person" as const, rows: peopleResults.value.map((r) => ({ key: r.key, label: r.label, open: { card: { kind: "person" as const, id: r.id } } })) },
    { key: "topics", kind: "topic" as const, rows: topicResults.value.map((r) => ({ key: r.key, label: r.label, open: { card: { kind: "topic" as const, id: r.id } } })) },
    { key: "themes", kind: "theme" as const, rows: themeResults.value.map((r) => ({ key: r.key, label: r.label, open: { to: { name: "theme", params: { id: r.routeId } } } })) },
    { key: "storylines", kind: "storyline" as const, rows: storylineResults.value.map((r) => ({ key: r.key, label: r.label, open: { to: { name: "storyline", params: { id: r.routeId } } } })) },
  ].filter((sec) => sec.rows.length > 0)
})
function trackSubjectPage(kind: SubjectKind): void {
  track("entity_open", { kind, presentation: "page", source: "search" })
}
function openEntityRow(kind: "topic" | "person", id: string): void {
  track("search_result_click", { result_kind: kind, rank: "1" })
  track("entity_open", { kind, presentation: "card", source: "search" })
  cardTarget.value = { kind, id }
}

const groups = computed<EpisodeGroup[]>(() => groupHits(passageResults.value))

/** Hits grouped by episode, in rank order (the first hit of an episode places it). */
function groupHits(hits: readonly SearchHit[]): EpisodeGroup[] {
  const byKey = new Map<string, EpisodeGroup>()
  const order: string[] = []
  for (const h of hits) {
    const slug = hitSlug(h)
    const key = slug ?? `doc:${h.doc_id}`
    let g = byKey.get(key)
    if (!g) {
      g = {
        slug,
        title: hitEpisode(h) ?? t("player.notFound"),
        show: hitShow(h),
        date: hitDate(h),
        art: hitArt(h),
        hits: [],
        rows: [],
      }
      byKey.set(key, g)
      order.push(key)
    }
    g.hits.push(h)
  }
  const built = order.map((k) => byKey.get(k)!)
  for (const g of built) g.rows = collapseFoldableHits(g.hits)
  return built
}

/** The server's `SEARCH_CATEGORIES`, for a hit that does not say its own category. */
const CATEGORY_OF_DOC_TYPE: Record<string, (typeof CATEGORY_ORDER)[number]> = {
  insight: "insights",
  transcript: "transcript",
  episode_title: "episodes",
  episode_description: "episodes",
  summary_short: "episodes",
  summary: "episodes",
  kg_entity: "mentions",
}
function hitCategory(h: SearchHit): string {
  const own = md(h).search_category
  if (typeof own === "string" && own) return own
  return CATEGORY_OF_DOC_TYPE[String(md(h).doc_type ?? "")] ?? "episodes"
}

/** The episode results, one section per category with hits, each grouped by episode. */
const categorySections = computed(() =>
  CATEGORY_ORDER.map((id) => {
    const hits = passageResults.value.filter((h) => hitCategory(h) === id)
    const years = groupEpisodesByYear<EpisodeGroup>(groupHits(hits), (g) => g.date)
    return {
      id,
      count: hits.length,
      yearSections: years,
      showYearHeaders: years.length > 1,
      more: categoryMore.value[id] === true,
    }
  }).filter((c) => c.count > 0)
)

/** "Show more" under a category: its next page, appended; "Show less" once it is all out. */
async function showMoreOrLess(id: string): Promise<void> {
  if (categoryLoading.value) return
  const mine = results.value.filter((h) => hitCategory(h) === id)
  if (!categoryMore.value[id]) {
    // "Show less": back to the first three; what was hidden is there to show again.
    const keep = new Set(mine.slice(0, PER_CATEGORY))
    results.value = results.value.filter((h) => hitCategory(h) !== id || keep.has(h))
    categoryMore.value = { ...categoryMore.value, [id]: true }
    return
  }
  const term = ranTerm.value
  categoryLoading.value = id
  try {
    const resp = await searchCorpus(term, PER_CATEGORY, scope.value, true, {
      perCategory: PER_CATEGORY,
      category: id,
      offset: mine.length,
    })
    if (ranTerm.value !== term) return
    const seen = new Set(results.value.map((h) => h.doc_id))
    results.value = [...results.value, ...resp.results.filter((h) => !seen.has(h.doc_id))]
    const cat = (resp.categories ?? []).find((c) => c.id === id)
    categoryMore.value = { ...categoryMore.value, [id]: cat?.has_more === true }
  } catch {
    // The page stays as it was; the control still offers more.
  } finally {
    categoryLoading.value = null
  }
}

// #1261-3: expand/collapse state per folded-cluster row, keyed by "<slug>|<kind>".
// A Set of expanded keys — reactive because the ref is a plain Set instance and
// the template mutates via ``clusterExpanded.value = new Set(...)`` on toggle.
const clusterExpanded = ref<Set<string>>(new Set())
function clusterKey(groupKey: string | null, cluster: FoldedHitCluster): string {
  return `${groupKey ?? "nogroup"}|${cluster.foldedKind}`
}
function toggleCluster(groupKey: string | null, cluster: FoldedHitCluster): void {
  const key = clusterKey(groupKey, cluster)
  const next = new Set(clusterExpanded.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  clusterExpanded.value = next
}
function isClusterOpen(groupKey: string | null, cluster: FoldedHitCluster): boolean {
  return clusterExpanded.value.has(clusterKey(groupKey, cluster))
}

// Recent searches (SR.3) — the last few terms, shown under the box, cap 8. The ACCOUNT's, synced
// through preferences (2026-10-11): kept in device storage, one account's searches showed to the
// next account signed in on the same phone, and never followed the listener to another device.
const RECENTS_PREF = "lp.search.recents"
const RECENTS_MAX = 8
const searchPrefs = useUserPreferencesStore()
void searchPrefs.hydrate()
const recents = computed<string[]>(() => {
  const v = searchPrefs.get<unknown>(RECENTS_PREF)
  return Array.isArray(v) ? v.filter((r): r is string => typeof r === "string").slice(0, RECENTS_MAX) : []
})
function recordRecent(term: string): void {
  void searchPrefs.set(RECENTS_PREF, [term, ...recents.value.filter((r) => r !== term)].slice(0, RECENTS_MAX))
}
function runRecent(q: string): void {
  query.value = q
  void router.push({ name: "search", query: { q } })
}

async function run(q: string): Promise<void> {
  const term = q.trim()
  if (!term) {
    // Invalidate any in-flight run so its late response can't repaint over this deliberate clear.
    runSeq++
    searching.value = false
    inFlightSig = ""
    lastRunTerm = ""
    results.value = []
    categoryMore.value = {}
    entity.value = null
    nameMatches.value = { person: [], topic: [], theme: [], storyline: [] }
    ran.value = false
    lastRunSig.value = ""
    ranTerm.value = ""
    return
  }
  // Already showing these exact results (e.g. this kept-alive view was re-activated on a tab
  // return) — don't re-fetch and flash a loading state over what's on screen.
  const sig = `${term}::${scope.value}`
  if (ran.value && lastRunSig.value === sig) return
  if (searching.value && inFlightSig === sig) return
  recordRecent(term)
  inFlightSig = sig
  lastRunTerm = term
  ranTerm.value = term
  // Generation token: two different queries can be in flight at once (no request cancellation), and
  // without this the slower-OLDER response wins — "cats" landing after "dogs" would paint cat
  // results over the dogs query. Every mutation below is gated on still being the current run.
  const mySeq = ++runSeq
  const current = (): boolean => mySeq === runSeq
  searching.value = true
  error.value = false
  const recall = scope.value === "mine"
  // Resolve a person/topic entity match in parallel with the passage search (3.4) — but only in the
  // whole-library scope; recall is about the user's own passages, not a global entity card.
  const entityP = recall
    ? Promise.resolve((entity.value = null))
    : resolveEntity(term).then(
        (r) => {
          // ORGANISATIONS are not surfaced as a result (operator 2026-09-17). They stay indexed and
          // still do their work upstream — resolution, graph edges, the entity pages that link them
          // — this only declines to offer one as a destination from search, where an org card is a
          // dead end next to the episodes and people a listener came for. Dropped at RENDER time,
          // deliberately: nothing about the index or the resolver changes, so re-enabling it is one
          // condition.
          if (current()) entity.value = r.entity?.kind === "organization" ? null : r.entity
        },
        () => {
          if (current()) entity.value = null
        }
      )
  // The subjects at the top, BY NAME (see `nameMatches`): each kind's name search, in both scopes.
  // A failure leaves that kind empty; it never fails the search.
  nameMatches.value = { person: [], topic: [], theme: [], storyline: [] }
  for (const kind of ["person", "topic", "theme", "storyline"] as const) {
    void searchInterests(kind, term, 50).then(
      (items) => {
        if (current()) nameMatches.value = { ...nameMatches.value, [kind]: items }
      },
      () => undefined
    )
  }
  try {
    // #1261-1: always ask the server for related-topic decoration; a broken
    // enricher chain degrades to plain hits and the chip row simply disappears.
    const resp = await searchCorpus(term, 12, recall ? "mine" : "all", true, {
      perCategory: PER_CATEGORY,
    })
    if (!current()) return
    results.value = resp.results
    categoryMore.value = Object.fromEntries((resp.categories ?? []).map((c) => [c.id, c.has_more]))
    error.value = Boolean(resp.error)
    // #2267 — NO QUERY TEXT, ever. Only the scope and a BUCKETED hit count: "did search answer
    // them" is the question, and `results: '0'` is what makes an empty search visible to the
    // beta's week-one check without recording what they were looking for.
    //
    // Reported after the response, not at submit, because the count is the point. A submit-time
    // event could not carry it, and `search_submitted` with no result count cannot distinguish a
    // search that worked from one that found nothing.
    track("search_submitted", {
      scope: recall ? "recall" : "corpus",
      results: toCountBucket(resp.results.length),
    })
    // #2267 friction. The beta's day-7 check reads these by name — "any errors or empty states,
    // especially not in corpus" — and an empty search is the single most common way a listener
    // discovers the corpus does not have what they wanted. `not_in_corpus` rather than
    // `no_results` when the search succeeded and simply found nothing: the distinction is whether
    // the SEARCH failed or the CONTENT is missing, and only the second is a corpus priority.
    if (resp.error) {
      track("error_shown", { surface: "search", kind: "server" })
    } else if (resp.results.length === 0) {
      track("empty_state_shown", { surface: "search", reason: "not_in_corpus" })
    }
  } catch {
    if (current()) error.value = true
  } finally {
    // NOT `await entityP`. The entity card fills in when its lookup lands (it guards itself with
    // current()); the passages must not wait for it. Awaiting it held the whole result list
    // behind a spinner for as long as the entity lookup took — 12-102 s on prod while the
    // entity-id map warmed after a deploy (2026-10-07), with the passages already in hand.
    void entityP
    // A newer run has taken over — leave its searching/ran/sig state alone.
    if (current()) {
      searching.value = false
      ran.value = true
      // Only remember a SUCCESSFUL run as "already on screen". Latching the sig on an error would
      // make the retry button (which re-runs the identical term+scope) early-return forever.
      lastRunSig.value = error.value ? "" : sig
    }
  }
}

function setScope(s: "all" | "mine"): void {
  // "my corpus" needs an account; signed-out it is a teaser that routes to sign-in (#1590).
  if (s === "mine" && isGated.value) {
    gated(() => {})()
    return
  }
  if (scope.value === s) return
  scope.value = s
  setSharedScope(s === "mine" ? "mine" : "corpus")
  void router.replace({ name: "search", query: { q: query.value.trim() || undefined, scope: s } })
  void run(query.value)
}

function submit(): void {
  const q = query.value.trim() || undefined
  // Emptying the box and pressing Enter must CLEAR results. `run("")` resets `lastRunTerm` first, so
  // the tab-return restore branch (which keys off `lastRunTerm`) won't bounce the URL back to the
  // old term — otherwise a deliberate empty submit is silently reverted.
  if (!q) void run("")
  void router.replace({ name: "search", query: { q, scope: scope.value } })
}

// Example searches for the zero state (before the first query) — tapping one runs it. Kept broad so
// they land on most corpora; the point is to TEACH that a search jumps you to the exact spoken
// moment, not to be exhaustive.
const EXAMPLES = ["how memory works", "artificial intelligence", "the future of work"]
function runExample(ex: string): void {
  query.value = ex
  submit()
}

function openEpisode(slug: string | null, hit?: SearchHit, rank?: number): void {
  if (!slug) return
  // WHICH result they took, and how far down (#2267). `episode_open` fires separately from the
  // router with `source: 'search'`, so this is not a duplicate: that one measures discovery, this
  // one measures whether search RANKS well — if every taken result is rank 1, ranking is doing the
  // work; if taps are spread down the list, people are hunting.
  track("search_result_click", { result_kind: "episode", rank: toRankBucket(rank ?? 1) })
  const s = hit ? hitStartSeconds(hit) : null
  // Reached only from "▶ Play from", and an explicit ▶ PLAYS (operator 2026-10-05).
  void router.push({
    name: "player",
    params: { slug },
    query: s != null ? { t: String(Math.floor(s)), play: "1" } : { play: "1" },
  })
}

// Search state PERSISTS across tab switches (this view is kept-alive). Two things made results
// vanish before: navigating AWAY changed `route.query.q` to undefined and re-ran an empty search,
// and the bottom-nav Search link returns to `{ name: 'search' }` with no `?q`. So: ignore the
// watcher while another tab is active, and on return with a prior search still live, restore the
// URL to it rather than clearing (operator 2026-09-13).
watch(
  () => [route.name, route.query.q] as const,
  ([name, q]) => {
    if (name !== "search") return
    const qs = String(q ?? "")
    // A prior search is live (settled OR still resolving) — restore ITS term (not an unsubmitted box
    // edit) so a tab return keeps the on-screen results rather than clearing or fetching a term the
    // user never ran.
    if (!qs && lastRunTerm && (ran.value || searching.value)) {
      void router.replace({
        name: "search",
        query: { q: lastRunTerm, scope: scope.value },
      })
      return
    }
    // The box follows the URL (operator 2026-10-04). This view is kept alive, so the box was filled
    // from `?q` ONCE, at setup; a later search from Home or Discover ran the new term while the box
    // still showed the old one.
    if (qs) query.value = qs
    void run(qs)
  },
  { immediate: true }
)

/**
 * Which RESULT KINDS to show — the same multi-select chips Library uses, none selected = all
 * (operator 2026-09-17).
 *
 * Only chips for kinds actually present, so the filter can never empty the page on its own — the
 * presence rule Library's bar follows. Not persisted: a filter is about the query you are reading
 * now, and carrying it into the next search would silently hide results.
 */
const resultTypes = ref<string[]>([])
const typeVisible = (key: string) => resultTypes.value.length === 0 || resultTypes.value.includes(key)
const availableResultTypes = computed(() => {
  const out: { key: string; label: string }[] = []
  // A chip per KIND, not one lumped "entity" chip (operator 2026-09-17): the resolver returns at
  // most one entity, so whichever kind it is gets its own chip and its own section.
  // People and topics are lists now (the exact-name answer plus the index hits), so their chips
  // follow the lists; the single entity card is left for a theme or a storyline.
  if (peopleResults.value.length) out.push({ key: "people", label: t("search.type_people") })
  if (topicResults.value.length) out.push({ key: "topics", label: t("search.type_topics") })
  if (themeResults.value.length) out.push({ key: "themes", label: t("search.type_themes") })
  // Storylines: index hits plus the resolver's exact-name answer, one list (storylineResults).
  if (storylineResults.value.length)
    out.push({ key: "storylines", label: t("search.type_storylines") })
  if (showMatches.value.length) out.push({ key: "shows", label: t("search.type_shows") })
  if (noteMatches.value.length) out.push({ key: "notes", label: t("notes.title") })
  if (boardMatches.value.length) out.push({ key: "boards", label: t("search.type_boards") })
  if (yearSections.value.length) out.push({ key: "episodes", label: t("search.typeEpisodes") })
  return out
})
// A chip for a kind that has just disappeared would stay selected and hide everything, so drop any
// selection the current results can no longer offer.
watch(availableResultTypes, (types) => {
  const keys = new Set(types.map((x) => x.key))
  const kept = resultTypes.value.filter((k) => keys.has(k))
  if (kept.length !== resultTypes.value.length) resultTypes.value = kept
})

const showEmpty = computed(
  () =>
    ran.value &&
    !searching.value &&
    !error.value &&
    results.value.length === 0 &&
    noteMatches.value.length === 0 &&
    boardMatches.value.length === 0 &&
    storylineResults.value.length === 0 &&
    topicResults.value.length === 0 &&
    peopleResults.value.length === 0 &&
    themeResults.value.length === 0 &&
    entity.value === null
)
/**
 * Mine found no EPISODES. Said whatever the subjects at the top show: those are matched by name
 * across the library (2026-10-12), so a new listener searching "index" still sees a topic named so,
 * and that must not hide "nothing in your listening" or the way to everyone's.
 */
const showMineEmpty = computed(
  () =>
    scope.value === "mine" &&
    ran.value &&
    !searching.value &&
    !error.value &&
    passageResults.value.length === 0 &&
    noteMatches.value.length === 0 &&
    boardMatches.value.length === 0
)
</script>

<template>
  <section>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t("search.title") }}
    </h1>

    <form class="lp-search flex flex-wrap items-center gap-2" @submit.prevent="submit">
      <label class="sr-only" for="search-q">{{ t("search.title") }}</label>
      <input
        id="search-q"
        v-model="query"
        type="search"
        :placeholder="t('search.placeholder')"
        class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-3 text-sm"
      />
      <!-- Scope: a compact boxed-pill toggle (matches Home's) riding the search row instead of a
           full second row below (operator). Everything ⇄ My listening. Auth-gated by setScope. -->
      <div class="inline-flex shrink-0 items-center rounded-full border border-border bg-surface p-1">
        <button
          type="button"
          data-testid="search-scope"
          class="inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-xs font-bold transition"
          :class="
            scope === 'mine'
              ? 'bg-accent text-accent-foreground'
              : 'text-muted hover:text-canvas-foreground'
          "
          :aria-pressed="scope === 'mine'"
          :aria-label="isGated ? t('auth.signInToSearchMine') : t('search.scopeLabel')"
          :title="scope === 'mine' ? t('search.scopeMine') : t('search.scopeAll')"
          @click="setScope(scope === 'mine' ? 'all' : 'mine')"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="h-3.5 w-3.5"
            aria-hidden="true"
          >
            <circle cx="12" cy="8" r="4" />
            <path d="M4 21a8 8 0 0 1 16 0" />
          </svg>
          {{ t("search.scopeMineShort") }}
        </button>
      </div>
      <!-- No standalone Search button (#1966). The field IS the button — the form already submits
           on return, and a `type=search` input carries that affordance natively. It was costing
           ~200px of a 412px row, collapsing the input to under half the width, in a row that also
           held Save and a bookmark. Four controls of four different weights for one action.
           The submit stays reachable for keyboard and assistive tech: the form submits on Enter,
           and the input is labelled. -->
      <!-- #1261-8: save the current query+scope to the listener's saved list.
           Only surfaces once the query is non-empty. -->
      <!-- The same size as the Mine toggle beside it (operator 2026-10-08: Save stood taller than its
           neighbours) — text-xs and py-2.5 make the same ~38px as Mine's padded pill. -->
      <button
        v-if="query.trim()"
        type="button"
        class="shrink-0 rounded-full border border-border px-3.5 py-2.5 text-xs font-bold text-muted transition hover:text-canvas-foreground"
        :aria-label="
          isGated
            ? t('auth.signInToSave')
            : currentIsSaved
            ? t('search.unsaveQuery')
            : t('search.saveQuery')
        "
        data-testid="save-query-button"
        @click="onSaveClick"
      >
        {{ currentIsSaved ? t("search.saved") : t("search.save") }}
      </button>
      <!-- Pin this search into a collection (RFC-119) — a live "more like this" seed for a board. -->
      <AddToCollectionButton
        v-if="query.trim()"
        :item="{ kind: 'search', ref: query.trim(), scope }"
      />
    </form>

    <!-- Recent searches (SR.3): per-device history, shown only when the box is empty. -->
    <div
      v-if="recents.length && !query.trim()"
      class="mt-3 flex flex-wrap items-center gap-1.5"
      data-testid="search-recents"
    >
      <span class="lp-kicker mr-1">{{ t("search.recent") }}</span>
      <button
        v-for="r in recents"
        :key="r"
        type="button"
        class="rounded-full bg-overlay px-3 py-1 text-sm text-canvas-foreground transition hover:bg-elevated"
        @click="runRecent(r)"
      >
        {{ r }}
      </button>
    </div>

    <!-- Confirmation: saving is otherwise silent, so this says it worked + where to find it (#saved-searches). -->
    <p
      v-if="saveMsg"
      class="mt-2 text-sm font-semibold text-grounded"
      aria-live="polite"
      data-testid="save-query-confirm"
    >
      {{ saveMsg }}
    </p>

    <!-- Result-kind filter — the same multi-select chips as Library's, "All" first so clearing is
         one tap. Only renders once there is more than one kind to choose between; with a single kind
         a filter is decoration. -->
    <TypeFilterBar
      v-if="availableResultTypes.length > 1 && !searching"
      v-model="resultTypes"
      :options="availableResultTypes"
      testid-prefix="search-type"
      class="mt-3"
    />

    <!-- Every result section carries a HEADING (operator 2026-09-17). "Your notes" had one and the
         other two did not, so the page read as one labelled section followed by loose content —
         with nothing saying where the notes ended and the corpus began. The headings use the same
         words as the filter chips, so a chip and the block it governs name the same thing. -->
    <!-- People / Topics / Storylines are three separate sections (operator 2026-09-17), each with
         its own heading, subtitle and filter chip. The resolver returns at most ONE entity, so
         exactly one of them renders for a given query — but which one is now named, instead of a
         lumped heading that made a person and a storyline look like the same kind of result. -->
    <!-- PEOPLE · TOPICS · THEMES · STORYLINES — one way (operator 2026-10-11): a section per kind,
         each the same kind-coloured pills, one per subject however many episodes it was found in,
         listed above the episodes (a topic or person connects episodes, so it is not a row under
         one). The exact-name answer leads its kind; eight, then "Show all (N)". A person or topic
         opens its card, a theme or storyline its page. -->
    <template v-for="sec in subjectSections" :key="sec.key">
      <template v-if="!searching && typeVisible(sec.key)">
        <h2 class="lp-section mb-1 mt-4" :data-testid="`search-section-${sec.key}`">
          {{ t(`search.type_${sec.key}`) }}
        </h2>
        <ul class="mt-2 flex flex-wrap gap-2" :data-testid="`search-${sec.key}`">
          <li v-for="row in shownRows(sec.key, sec.rows)" :key="row.key">
            <RouterLink
              v-if="'to' in row.open"
              :to="row.open.to"
              class="lp-tap inline-flex items-center rounded-full px-3 py-1.5 text-xs font-semibold no-underline transition hover:opacity-80"
              :class="kindPill(sec.kind)"
              :data-kind="sec.kind"
              :data-testid="`search-${sec.kind}-result`"
              :aria-label="t('kp.openEntity', { term: row.label })"
              @click="trackSubjectPage(sec.kind)"
            >{{ row.label }}</RouterLink>
            <button
              v-else
              type="button"
              class="lp-tap inline-flex items-center rounded-full px-3 py-1.5 text-xs font-semibold transition hover:opacity-80"
              :class="kindPill(sec.kind)"
              :data-kind="sec.kind"
              :data-testid="`search-${sec.kind}-result`"
              :aria-label="t('kp.openEntity', { term: row.label })"
              @click="openEntityRow(row.open.card.kind, row.open.card.id)"
            >{{ row.label }}</button>
          </li>
        </ul>
        <button
          v-if="sec.rows.length > ENTITY_ROWS_SHOWN"
          type="button"
          class="lp-tap mt-2 text-sm font-bold text-accent"
          :aria-expanded="expandedEntityLists.has(sec.key)"
          :data-testid="`search-${sec.key}-more`"
          @click="toggleEntityList(sec.key)"
        >{{ expandedEntityLists.has(sec.key) ? t("home.showLess") : t("search.showAllCount", { count: sec.rows.length }) }}</button>
      </template>
    </template>

    <!-- SHOWS — matched by title / host, every word in any order (see `showMatches`). -->
    <template v-if="showMatches.length && !searching && typeVisible('shows')">
      <h2 class="lp-section mb-1 mt-4" data-testid="search-section-shows">
        {{ t("search.type_shows") }}
      </h2>
      <p class="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
        {{ t("search.showsSummary", { count: showMatches.length }, showMatches.length) }}
      </p>
      <ul class="flex flex-col gap-2" data-testid="search-shows">
        <!-- Every show action behind one ⋯ right of the name — where the episode cards below put
             theirs (operator 2026-10-05). -->
        <li v-for="p in showMatches" :key="p.feed_id">
          <ShowRow :show="p"><template #menu><ShowMenu :show="p" /></template></ShowRow>
        </li>
      </ul>
    </template>

    <!-- SR.1: the listener's OWN notes matching the query, as their own section above the corpus
         passages (a note is not a transcript hit). Independent of the results chain below, so notes
         and corpus hits can both show. Client-side text match on the capture store. -->
    <section
      v-if="noteMatches.length && typeVisible('notes')"
      class="mt-4"
      data-testid="search-note-matches"
    >
      <h2 class="lp-section mb-1">{{ t("notes.title") }}</h2>
      <p class="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
        {{ t("search.notesSummary", { count: notesPage.total.value }, notesPage.total.value) }}
      </p>
      <ul class="flex flex-col gap-2">
        <li
          v-for="n in noteMatches"
          :key="n.id"
          class="rounded-xl border border-border p-3"
          data-testid="search-note"
        >
          <p class="whitespace-pre-wrap text-sm leading-relaxed text-canvas-foreground">
            {{ n.text }}
          </p>
          <!-- Kind AND what it is attached to: "EPISODE · Risk Is a Systems Property". The kind
               alone left a list reading "episode, episode, topic" with no way to tell WHICH without
               following every link (operator 2026-09-17). `min-w-0 truncate` so a long episode title
               shortens instead of pushing "Open" off the row. -->
          <div class="mt-1.5 flex items-center gap-2 text-xs">
            <span class="lp-kicker shrink-0">{{ n.target }}</span>
            <span
              v-if="noteLabel(n.target, n.target_id)"
              class="min-w-0 flex-1 truncate font-semibold text-canvas-foreground"
              data-testid="search-note-target"
              >{{ noteLabel(n.target, n.target_id) }}</span
            >
            <RouterLink
              v-if="noteRoute(n.target, n.target_id)"
              :to="noteRoute(n.target, n.target_id)!"
              class="ml-auto shrink-0 font-semibold text-accent no-underline"
              >{{ t("notes.open") }}</RouterLink
            >
          </div>
        </li>
      </ul>
      <ShowAllToggle
        v-if="notesPage.total.value > 5"
        :expanded="notesPage.remaining.value === 0"
        :count="notesPage.total.value"
        :remaining="notesPage.remaining.value"
        data-testid="search-notes-more"
        @toggle="notesPage.toggle()"
      />
    </section>

    <!-- The listener's own BOARDS (#2341), beside their notes: a board whose name matches, then one
         holding something that does. Each row says what matched and opens the board in Library. -->
    <section
      v-if="boardMatches.length && typeVisible('boards')"
      class="mt-4"
      data-testid="search-board-matches"
    >
      <h2 class="lp-section mb-1">{{ t("search.type_boards") }}</h2>
      <p class="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
        {{ t("search.boardsSummary", { count: boardMatches.length }, boardMatches.length) }}
      </p>
      <ul class="flex flex-col gap-2">
        <li v-for="(m, i) in boardMatches" :key="m.collection.id">
          <RouterLink
            :to="{ name: 'library', query: { tab: 'collections', board: m.collection.id } }"
            class="flex items-center gap-3 rounded-xl border border-border p-3 no-underline"
            data-testid="search-board"
            @click="openBoard(i + 1)"
          >
            <img
              v-if="m.collection.cover_url"
              :src="m.collection.cover_url"
              alt=""
              loading="lazy"
              class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
            />
            <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" aria-hidden="true" />
            <span class="min-w-0 flex-1">
              <span class="block truncate text-sm font-semibold text-canvas-foreground">{{
                m.collection.name
              }}</span>
              <span class="block truncate text-xs text-muted" data-testid="search-board-why">{{
                boardWhy(m)
              }}</span>
            </span>
            <span class="shrink-0 text-xs text-muted">{{
              t("collections.count", m.collection.count, { named: { count: m.collection.count } })
            }}</span>
          </RouterLink>
        </li>
      </ul>
    </section>

    <!-- F1.3/F1.4: reserve the results shape while searching (no jump when they land) and offer a
         retry on failure, instead of a bare "Searching…"/error line. -->
    <SectionStatus
      v-if="searching || error"
      class="mt-4"
      :phase="searching ? 'loading' : 'error'"
      :rows="4"
      @retry="run(query)"
    />
    <!-- An empty Mine says so AND offers everyone's (ADR-162): Mine is the default, so this is a
         new listener's first search — it must not end on an empty page. -->
    <div v-else-if="showMineEmpty" class="mt-4 text-muted" data-testid="search-mine-empty">
      <p>{{ t("search.recallEmpty") }}</p>
      <button
        type="button"
        class="mt-2 font-bold text-accent"
        data-testid="search-show-everyone"
        @click="setScope('all')"
      >{{ t("search.searchEveryone") }}</button>
    </div>
    <p v-else-if="showEmpty" class="mt-4 text-muted">{{ t("search.noResults") }}</p>

    <!-- `typeVisible('episodes')` gates the whole corpus block — its count line and topic chips
         describe the episode results, so they go when the episodes do. -->
    <template v-else-if="passageResults.length && typeVisible('episodes')">
      <!-- The episodes block gets a heading like the notes block has, with the passage/episode count
           as its SUBTITLE rather than standing in for a title (operator 2026-09-17). The count was
           doing both jobs and neither well: it never said what the section was. -->
      <h2 class="lp-section mb-1 mt-4" data-testid="search-section-episodes">
        {{ t("search.typeEpisodes") }}
      </h2>
      <p class="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
        <!-- The term is confirmed IN the count row (operator 2026-10-04) — the box can be edited after
             a search, so the results needed to say what they are results for. -->
        {{ t("search.summaryFor", { passages: passageResults.length, episodes: groups.length, term: ranTerm }) }}
      </p>

      <!-- #1261-2: related-topic chip row above the episode groups. Silent
           when the QueryEnricher chain returned nothing (broken corpus, no
           topic_similarity.json, or no hits carried topic decorations). -->
      <div
        v-if="relatedTopicChips.length"
        class="mt-3 flex flex-wrap items-center gap-1.5"
        data-testid="related-topic-chips"
      >
        <span class="lp-kicker mr-1">{{ t("search.alsoAbout") }}</span>
        <button
          v-for="chip in relatedTopicChips"
          :key="chip.topicId"
          type="button"
          class="rounded-full border border-border bg-surface px-2.5 py-1 text-xs font-semibold text-canvas-foreground transition hover:bg-overlay"
          :aria-label="t('search.openTopicChip', { label: chip.label })"
          @click="openTopicChip(chip.topicId)"
        >
          {{ chip.label }}
        </button>
      </div>

      <!-- #1261-7: year sections wrap the episode-group list. When the
           results span a single year, the header is suppressed so the
           results still read as one flat list. -->
      <section
        v-for="cat in categorySections"
        :key="cat.id"
        class="mt-5"
        :data-testid="`search-category-${cat.id}`"
      >
      <h3 class="lp-kicker" data-testid="search-category-title">{{ t(`search.category.${cat.id}`) }}</h3>
      <template v-for="section in cat.yearSections" :key="section.year">
        <h2
          v-if="cat.showYearHeaders"
          class="mt-6 mb-2 font-display text-sm font-bold uppercase tracking-wider text-muted"
          data-testid="year-header"
        >
          {{ yearLabel(section.year) }}
          <span class="ml-1 font-normal">
            · {{ t("search.yearEpisodes", section.groups.length) }}
          </span>
        </h2>
        <ul class="mt-3 flex flex-col gap-6">
          <!-- The SHARED episode-group header (`EpisodeGroupCard`), also used by Saved and Revisit
               (operator 2026-10-05): artwork, show, title, one meta line, the fold as a chevron.
               Search's own content rides its slots — when + how many in `#meta`, the matched-field
               breakdown under the title in `#extra` — and the passages are the body. -->
          <EpisodeGroupCard
            v-for="(g, gi) in section.groups"
            :key="g.slug ?? g.title"
            :episode="groupAsEpisode(g)"
            :item-count="g.rows.length"
            :moments="hasMoments(g)"
            moments-link
          >
            <template #meta>
              <template v-if="formatPublishDate(g.date, locale)">{{ formatPublishDate(g.date, locale) }} · </template>{{ t("search.matchCount", g.hits.length) }}
            </template>
            <!-- #1261-5: matched-field breakdown ("Matched: Title · Summary ×2 · Transcript") —
                 so the listener knows WHY this episode surfaced without tapping through. Hidden
                 when nothing resolved to an episode-level field. -->
            <template v-if="matchedFieldChips(episodeHits(g)).length" #extra>
              <span class="lp-kicker mt-1 block" data-testid="matched-fields">
                {{ t("search.matchedPrefix") }}
                <template v-for="(m, mi) in matchedFieldChips(episodeHits(g))" :key="m.label">
                  <template v-if="mi > 0"> · </template>
                  <span class="font-semibold text-canvas-foreground">
                    {{ m.label }}<template v-if="m.count > 1"> ×{{ m.count }}</template>
                  </span>
                </template>
              </span>
            </template>

            <!-- Matching passages (#1261-3: foldable rows collapse to one
               expandable summary per (episode, source-kind)). -->
            <ul class="flex flex-col">
              <template v-for="(row, i) in g.rows" :key="rowKey(row, i)">
                <!-- FoldedHitCluster: N hits of the same foldable kind (transcript /
                   title / description / summary) collapsed into one expandable row. -->
                <li v-if="isFoldedCluster(row)" class="border-b border-border">
                  <button
                    type="button"
                    class="flex w-full items-center gap-2 py-3 text-left"
                    :aria-expanded="isClusterOpen(g.slug, row)"
                    :aria-label="
                      t('search.expandCluster', {
                        kind: t(`search.foldedKind.${row.foldedKind}`),
                        count: row.members.length,
                      })
                    "
                    data-testid="folded-cluster-row"
                    @click="toggleCluster(g.slug, row)"
                  >
                    <span
                      class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-canvas-foreground"
                    >
                      {{ t(`search.foldedKind.${row.foldedKind}`) }}
                    </span>
                    <span class="text-xs font-semibold text-muted">
                      {{ t("search.foldedCount", row.members.length) }}
                    </span>
                    <span class="ml-auto text-xs font-bold text-accent">
                      {{ isClusterOpen(g.slug, row) ? "▲" : "▼" }}
                    </span>
                  </button>
                  <ul v-if="isClusterOpen(g.slug, row)" class="flex flex-col">
                    <li
                      v-for="(m, mi) in row.members"
                      :key="m.doc_id + mi"
                      class="border-t border-border py-2 pl-4"
                    >
                      <div class="flex items-center gap-2">
                        <PlayFrom
                          v-if="hitStartSeconds(m) != null && g.slug"
                          class="ml-auto"
                          :seconds="hitStartSeconds(m)"
                          :aria-label="
                            t('search.jumpTo', {
                              time: formatTime(hitStartSeconds(m) ?? 0),
                              episode: g.title,
                            })
                          "
                          @click="openEpisode(g.slug, m, gi + 1)"
                        />
                      </div>
                      <p class="mt-1 line-clamp-2 text-sm leading-relaxed text-surface-foreground">
                        {{ m.text }}
                      </p>
                    </li>
                  </ul>
                </li>
                <!-- Plain hit (insight / kg_topic / kg_entity / lifted transcript). -->
                <li v-else class="border-b border-border py-3">
                  <div class="flex items-center gap-2">
                    <span
                      class="rounded bg-overlay px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
                      :class="{
                        'text-grounded': hitKind(row) === 'insight',
                        'text-canvas-foreground':
                          hitKind(row) === 'transcript' || hitKind(row) === 'passage',
                        'text-topic': hitKind(row) === 'topic',
                      }"
                    >
                      {{ t(`search.kind.${hitKind(row)}`) }}
                    </span>
                    <PlayFrom
                      v-if="hitStartSeconds(row) != null && g.slug"
                      class="ml-auto"
                      :seconds="hitStartSeconds(row)"
                      :aria-label="
                        t('search.jumpTo', {
                          time: formatTime(hitStartSeconds(row) ?? 0),
                          episode: g.title,
                        })
                      "
                      @click="openEpisode(g.slug, row, gi + 1)"
                    />
                  </div>
                  <p
                    class="mt-1.5 line-clamp-2 text-sm leading-relaxed"
                    :class="
                      hitKind(row) === 'topic' ? 'italic text-muted' : 'text-surface-foreground'
                    "
                  >
                    {{ row.text }}
                  </p>
                </li>
              </template>
            </ul>
          </EpisodeGroupCard>
        </ul>
      </template>
      <!-- Under the category's last item, as on the rails: its next three, then "Show less". -->
      <ShowAllToggle
        v-if="cat.more || cat.count > PER_CATEGORY"
        :expanded="cat.count > PER_CATEGORY"
        :count="cat.count"
        :remaining="cat.more ? 1 : 0"
        :more-label="t('search.categoryMore')"
        :disabled="categoryLoading === cat.id"
        :data-testid="`search-category-more-${cat.id}`"
        @toggle="showMoreOrLess(cat.id)"
      />
      </section>
    </template>

    <!-- Zero state (before the first search): teach the feature instead of a blank page. Search is
         the differentiator (jump-to-moment), and you can arrive here without passing Home's selling
         hero — via Discovery's box (the Discover tab, from any screen) or the desktop masthead
         magnifier. A first-timer landing here needs a nudge.
         Tapping an example runs it. -->
    <!-- Zero state sits directly under the search row (operator): the examples + any saved searches
         follow the field with normal spacing. The prior version centred the chips in a 58dvh box,
         which read as a big blank gap on a tall phone. -->
    <div
      v-else-if="!ran"
      class="mt-6"
      data-testid="search-zero-state"
    >
      <p class="text-sm text-muted">{{ t("search.tryPrompt") }}</p>
      <div class="mt-2 flex flex-wrap gap-2">
        <button
          v-for="ex in EXAMPLES"
          :key="ex"
          type="button"
          class="rounded-full border border-border bg-surface px-3 py-1.5 text-sm font-semibold text-canvas-foreground transition hover:bg-overlay"
          @click="runExample(ex)"
        >
          {{ ex }}
        </button>
      </div>

      <!-- The listener's own saved searches, when they have any (#1966).
           The zero state was three example chips above ~1,200px of unbroken black — not confident
           negative space, an unfinished page. Their own searches are the most useful thing that can
           occupy it, and they cost nothing: the store is already hydrated for the Save control in
           the row above. Absent for anyone who has saved none, which is the honest empty state
           rather than filler. -->
      <div v-if="savedQueries.list.length" class="mt-8">
        <h2 class="lp-section mb-2 text-base">{{ t("search.savedTitle") }}</h2>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="sq in savedQueries.list.slice(0, 8)"
            :key="`${sq.q}-${sq.scope}`"
            type="button"
            class="rounded-full border border-border px-3 py-1.5 text-sm font-semibold text-muted transition hover:text-canvas-foreground"
            @click="runExample(sq.q)"
          >
            {{ sq.q }}
          </button>
        </div>
      </div>
    </div>

    <EntityCard
      v-if="cardTarget"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
  </section>
</template>
