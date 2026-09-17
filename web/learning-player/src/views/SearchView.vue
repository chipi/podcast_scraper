<script setup lang="ts">
/**
 * Corpus-wide grounded search (PRD-042 FR5 / RFC-099 §Home). Searches the whole library via
 * GET /api/app/search (extractive, no request-time LLM). Rather than a flat wall of mixed
 * passages, results are **grouped by episode** (ranked by their best hit) and each passage is
 * labelled by kind (Insight / Transcript / Topic). A "Play from …" jump appears only when the
 * passage carries a real timestamp — otherwise we open the episode rather than fake a 0:00.
 */
import { computed, onMounted, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { noteRoute as resolveNoteRoute } from "../composables/noteTarget"
defineOptions({ name: "SearchView" }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRoute, useRouter } from "vue-router"
import { resolveEntity, searchCorpus } from "../services/api"
import { resolveMediaUrl } from "../services/tier"
import type { EntityRef, Note, SearchHit } from "../services/types"
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
import { useSavedQueriesStore } from "../stores/savedQueries"
import { useCaptureStore } from "../stores/capture"
import EntityCard from "../components/EntityCard.vue"
import EpisodeActions from "../components/EpisodeActions.vue"
import AddToCollectionButton from "../components/AddToCollectionButton.vue"
import SectionStatus from "../components/SectionStatus.vue"

const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const { isGated, gated } = useSignInGate()
const savedQueries = useSavedQueriesStore()
const capture = useCaptureStore()
// Shared rule (composables/noteTarget) — was a second, drifted copy of the same function.
const noteRoute = (target: string, id: string) => resolveNoteRoute(target, id, capture.highlights)
// SR.1 — search the listener's OWN notes alongside the corpus. Notes are per-user and client-side,
// so this is a local text match, shown as its own "Your notes" section rather than interleaved with
// the corpus passages (a note is not a transcript hit).
onMounted(() => void capture.ensureLoaded().catch(() => {}))
const noteMatches = computed<Note[]>(() => {
  const q = query.value.trim().toLowerCase()
  if (!ran.value || !q) return []
  // `?? []`: the async ensureLoaded() from onMounted can resolve after the store is disposed (test
  // teardown), re-running this computed against a torn-down store whose `notes` is undefined. A
  // computed must be total, so read defensively rather than throw into Vue's flush.
  return (capture.notes ?? [])
    .filter((n) => n.text.toLowerCase().includes(q))
    .sort((a, b) => b.created_at - a.created_at)
})
// USERPREFS-1 hydrate fires once at app init in main.ts; the savedQueries
// watch reacts when the payload arrives so the Save button flips to
// "Saved ✓" if the current query was already persisted. No per-view
// hydrate call — the tests demonstrated that adding one creates
// test-order flakiness without changing user-visible behaviour.

// Scope (P3 Recall, #1124): 'all' = whole library; 'mine' = grounded recall over the user's
// heard∪captured corpus ("what have I learned about X"). The toggle only shows when signed in.
const scope = ref<"all" | "mine">(route.query.scope === "mine" ? "mine" : "all")

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
const entity = ref<EntityRef | null>(null)
const cardTarget = ref<{ kind: "person" | "topic" | "organization"; id: string } | null>(null)
const searching = ref(false)
const error = ref(false)
const ran = ref(false)
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
  cardTarget.value = { kind: "topic", id: topicId }
}

// #1261-5: shim so the template's inline v-for can call the helper by name.
// Wrapped so summarizeMatchedFields can be swapped in unit tests independently.
function matchedFieldChips(hits: SearchHit[]) {
  return summarizeMatchedFields(hits)
}

// #1261-7: bucket the episode groups by publish year so the results read as
// "2026 · 4 episodes" / "2025 · 12 episodes" sections — the mobile-friendly
// reshape of the operator viewer's timeline chart concept. Sections are only
// shown when the search spans multiple years; single-year results skip the
// header to keep the page short.
const yearSections = computed<YearSection<EpisodeGroup>[]>(() =>
  groupEpisodesByYear<EpisodeGroup>(groups.value, (g) => g.date)
)
const showYearHeaders = computed(() => yearSections.value.length > 1)

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
const groups = computed<EpisodeGroup[]>(() => {
  const byKey = new Map<string, EpisodeGroup>()
  const order: string[] = []
  for (const h of results.value) {
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
})

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

// Recent searches (SR.3) — a small per-device history shown under the box; localStorage, cap 8.
const RECENTS_KEY = "lp.search.recents"
const RECENTS_MAX = 8
const recents = ref<string[]>([])
try {
  const raw = localStorage.getItem(RECENTS_KEY)
  recents.value = raw ? (JSON.parse(raw) as string[]).slice(0, RECENTS_MAX) : []
} catch {
  recents.value = []
}
function recordRecent(term: string): void {
  recents.value = [term, ...recents.value.filter((r) => r !== term)].slice(0, RECENTS_MAX)
  try {
    localStorage.setItem(RECENTS_KEY, JSON.stringify(recents.value))
  } catch {
    /* storage blocked — recents are a convenience, not critical */
  }
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
    entity.value = null
    ran.value = false
    lastRunSig.value = ""
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
          if (current()) entity.value = r.entity
        },
        () => {
          if (current()) entity.value = null
        }
      )
  try {
    // #1261-1: always ask the server for related-topic decoration; a broken
    // enricher chain degrades to plain hits and the chip row simply disappears.
    const resp = await searchCorpus(term, 12, recall ? "mine" : "all", true)
    if (!current()) return
    results.value = resp.results
    error.value = Boolean(resp.error)
  } catch {
    if (current()) error.value = true
  } finally {
    await entityP
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
  void router.replace({ name: "search", query: { q: query.value.trim() || undefined, scope: s } })
  void run(query.value)
}

function openEntity(): void {
  if (entity.value) cardTarget.value = { kind: entity.value.kind, id: entity.value.id }
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

function openEpisode(slug: string | null, hit?: SearchHit): void {
  if (!slug) return
  const s = hit ? hitStartSeconds(hit) : null
  void router.push({
    name: "player",
    params: { slug },
    query: s != null ? { t: String(Math.floor(s)) } : {},
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
    void run(qs)
  },
  { immediate: true }
)

const showEmpty = computed(
  () =>
    ran.value &&
    !searching.value &&
    !error.value &&
    results.value.length === 0 &&
    noteMatches.value.length === 0 &&
    entity.value === null
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
      <button
        v-if="query.trim()"
        type="button"
        class="shrink-0 rounded-full border border-border px-4 py-3 text-sm font-bold text-muted transition hover:text-canvas-foreground"
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

    <!-- Entity match (3.4): a person/topic card above the passages, opening the full card on tap. -->
    <button
      v-if="entity && !searching"
      type="button"
      class="mt-4 flex w-full items-center gap-3 rounded-xl border border-border bg-surface p-4 text-left transition hover:bg-overlay"
      :aria-label="t('kp.openEntity', { term: entity.label })"
      @click="openEntity"
    >
      <span class="min-w-0 flex-1">
        <span class="lp-kicker block">{{
          entity.kind === "person"
            ? t("ec.person")
            : entity.kind === "organization"
              ? t("ec.organization")
              : t("ec.topic")
        }}</span>
        <span class="block font-display text-lg font-bold text-canvas-foreground">{{
          entity.label
        }}</span>
      </span>
      <span class="shrink-0 text-sm font-semibold text-accent">{{ t("search.viewEntity") }} ›</span>
    </button>

    <!-- SR.1: the listener's OWN notes matching the query, as their own section above the corpus
         passages (a note is not a transcript hit). Independent of the results chain below, so notes
         and corpus hits can both show. Client-side text match on the capture store. -->
    <section v-if="noteMatches.length" class="mt-4" data-testid="search-note-matches">
      <h2 class="lp-section mb-2">{{ t("notes.title") }}</h2>
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
          <div class="mt-1.5 flex items-center gap-2 text-xs">
            <span class="lp-kicker">{{ n.target }}</span>
            <RouterLink
              v-if="noteRoute(n.target, n.target_id)"
              :to="noteRoute(n.target, n.target_id)!"
              class="font-semibold text-accent no-underline"
              >{{ t("notes.open") }}</RouterLink
            >
          </div>
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
    <p v-else-if="showEmpty && scope === 'mine'" class="mt-4 text-muted">
      {{ t("search.recallEmpty") }}
    </p>
    <p v-else-if="showEmpty" class="mt-4 text-muted">{{ t("search.noResults") }}</p>

    <template v-else-if="results.length">
      <p class="mt-4 text-xs font-semibold uppercase tracking-wider text-muted">
        {{ t("search.summary", { passages: results.length, episodes: groups.length }) }}
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
      <template v-for="section in yearSections" :key="section.year">
        <h2
          v-if="showYearHeaders"
          class="mt-6 mb-2 font-display text-sm font-bold uppercase tracking-wider text-muted"
          data-testid="year-header"
        >
          {{ yearLabel(section.year) }}
          <span class="ml-1 font-normal">
            · {{ t("search.yearEpisodes", section.groups.length) }}
          </span>
        </h2>
        <ul class="mt-3 flex flex-col gap-3">
          <li
            v-for="g in section.groups"
            :key="g.slug ?? g.title"
            class="overflow-hidden rounded-xl border border-border bg-surface"
          >
            <!-- Episode header: tapping the row opens/plays the episode; a quick-action cluster
               (favorite + queue) sits alongside, like a Library row (#2). The actions are siblings
               of the open button, never nested inside it (no interactive-in-interactive). -->
            <!--
            ONE narrow left column, not a left artwork AND a right rail. Everything that is not the
            text — artwork, match count, action row — stacks under the artwork at one width, so the
            centre text is squeezed from one side only.

            The column is `w-32` (128px), matching the Browse card's artwork (operator 2026-09-14):
            it is the width the shared `EpisodeActions` row needs to sit on ONE line (favourite +
            queue + ⋯ ≈ 120px), and a bigger cover reads like Browse rather than a cramped thumbnail.
            At the old 76px the third control (⋯) wrapped to a second row.

            `items-start` on the row: the artwork sits at the TOP of a multi-line title rather than
            floating against its middle.
          -->
            <div class="flex w-full items-start gap-3 px-4 pt-4">
              <div class="flex w-32 shrink-0 flex-col items-center gap-1.5">
                <button
                  v-if="g.art"
                  type="button"
                  class="w-full"
                  :aria-label="t('search.openEpisode', { title: g.title })"
                  @click="openEpisode(g.slug)"
                >
                  <img
                    :src="g.art"
                    alt=""
                    loading="lazy"
                    class="h-32 w-32 rounded-md bg-elevated object-cover"
                  />
                </button>
                <span class="text-center text-xs font-semibold text-muted">
                  {{ t("search.matchCount", g.hits.length) }}
                </span>
                <!-- Sibling of the open button, never nested inside it (no interactive-in-
                   interactive). The shared EpisodeActions row (favourite/queue/download/collect). -->
                <EpisodeActions v-if="g.slug" :slug="g.slug" data-testid="search-result-actions" />
              </div>
              <button
                type="button"
                class="flex min-w-0 flex-1 text-left"
                @click="openEpisode(g.slug)"
              >
                <span class="min-w-0 flex-1">
                  <span
                    class="block font-display text-base font-bold leading-snug text-canvas-foreground"
                  >
                    {{ g.title }}
                  </span>
                  <span v-if="g.show || g.date" class="lp-kicker mt-0.5 block">
                    {{ g.show }}<template v-if="g.show && g.date"> · </template
                    >{{ g.date ? formatPublishDate(g.date, locale) : "" }}
                  </span>
                  <!-- #1261-5: matched-field breakdown ("Matched: Title · Summary
                   ×2 · Transcript") — small kicker line so the listener knows
                   why this episode surfaced without tapping through. Hidden
                   when nothing resolved to an episode-level field. -->
                  <span
                    v-if="matchedFieldChips(g.hits).length"
                    class="lp-kicker mt-0.5 block"
                    data-testid="matched-fields"
                  >
                    {{ t("search.matchedPrefix") }}
                    <template v-for="(m, mi) in matchedFieldChips(g.hits)" :key="m.label">
                      <template v-if="mi > 0"> · </template>
                      <span class="font-semibold text-canvas-foreground">
                        {{ m.label }}<template v-if="m.count > 1"> ×{{ m.count }}</template>
                      </span>
                    </template>
                  </span>
                </span>
              </button>
            </div>

            <!-- Matching passages (#1261-3: foldable rows collapse to one
               expandable summary per (episode, source-kind)). -->
            <ul class="mt-3 flex flex-col">
              <template v-for="(row, i) in g.rows" :key="rowKey(row, i)">
                <!-- FoldedHitCluster: N hits of the same foldable kind (transcript /
                   title / description / summary) collapsed into one expandable row. -->
                <li v-if="isFoldedCluster(row)" class="border-t border-border">
                  <button
                    type="button"
                    class="flex w-full items-center gap-2 px-4 py-3 text-left"
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
                      class="border-t border-border px-6 py-2"
                    >
                      <div class="flex items-center gap-2">
                        <button
                          v-if="hitStartSeconds(m) != null && g.slug"
                          type="button"
                          class="ml-auto font-mono text-xs font-bold text-accent"
                          :aria-label="
                            t('search.jumpTo', {
                              time: formatTime(hitStartSeconds(m) ?? 0),
                              episode: g.title,
                            })
                          "
                          @click="openEpisode(g.slug, m)"
                        >
                          ▶
                          {{ t("search.playHere", { time: formatTime(hitStartSeconds(m) ?? 0) }) }}
                        </button>
                      </div>
                      <p class="mt-1 line-clamp-2 text-sm leading-relaxed text-surface-foreground">
                        {{ m.text }}
                      </p>
                    </li>
                  </ul>
                </li>
                <!-- Plain hit (insight / kg_topic / kg_entity / lifted transcript). -->
                <li v-else class="border-t border-border px-4 py-3">
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
                    <button
                      v-if="hitStartSeconds(row) != null && g.slug"
                      type="button"
                      class="ml-auto font-mono text-xs font-bold text-accent"
                      :aria-label="
                        t('search.jumpTo', {
                          time: formatTime(hitStartSeconds(row) ?? 0),
                          episode: g.title,
                        })
                      "
                      @click="openEpisode(g.slug, row)"
                    >
                      ▶ {{ t("search.playHere", { time: formatTime(hitStartSeconds(row) ?? 0) }) }}
                    </button>
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
          </li>
        </ul>
      </template>
    </template>

    <!-- Zero state (before the first search): teach the feature instead of a blank page. Search is
         the differentiator (jump-to-moment), and the phone Search tab skips Home's selling hero, so
         a first-timer landing here needs a nudge. Tapping an example runs it. -->
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
