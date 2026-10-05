<script setup lang="ts">
/**
 * Knowledge Panel (PRD-039 FR4 / RFC-099 §5) — the learning surface beside the player.
 * Sections (each independently hidden when its artifact is absent):
 *   Ask (extractive grounded search, no LLM) · Summary · Topics · People · Insights.
 * Every timestamp + Ask result emits `seek` for jump-to-moment. Tapping a person filters
 * the insight list. The "surfacing now" insight (driven by playback) is highlighted.
 *
 * "More like this" surfaces semantic peer episodes (vector similarity) at the foot of the
 * panel — the consolidation loop: finish here, keep learning next.
 */
import { computed, nextTick, onMounted, ref, watch } from "vue"
import CloseIcon from "./CloseIcon.vue"
import { useI18n } from "vue-i18n"
import { episodeNotesUrl, fetchEpisodeNotes, getRelated, searchEpisode } from "../services/api"
import type {
  EpisodeDetail,
  EpisodeSummary,
  Entity,
  Insight,
  SearchHit,
  Topic,
} from "../services/types"
import { formatTime } from "../player/transcriptSync"
import { hitStartSeconds, insightStartSeconds } from "../player/insights"
import { speakerLabel } from "../utils/format"
import CardRail from "./CardRail.vue"
import EpisodeTile from "./EpisodeTile.vue"
import PlayFrom from "./PlayFrom.vue"
import { useAuthStore } from "../stores/auth"
import { sheetTeleportTarget } from "../composables/sheetStack"
import { useSignInGate } from "../composables/useSignInGate"
import { scrollBehavior } from "../utils/motion"
import { holdScroll, offsetWithin, restoreScroll, waitForSettledElement } from "../utils/scrollRestore"
import { NOTES_ANCHOR } from "../composables/noteTarget"
import { useCaptureStore } from "../stores/capture"
import CollapsibleSection from "./CollapsibleSection.vue"
import HighlightToggle from "./HighlightToggle.vue"
import InsightTypeMark from "./InsightTypeMark.vue"
import NoteComposer from "./NoteComposer.vue"
import EntityCardBody from "./EntityCardBody.vue"
import { personName } from "../utils/personName"
import ProfileAvatar from "./ProfileAvatar.vue"
import StorylineCard from "./StorylineCard.vue"
import ThemeCard from "./ThemeCard.vue"
import EpisodeDensity from "./EpisodeDensity.vue"
import { isNative, openExternal, saveAndShareText } from "../services/native"
import { exportFilename } from "../utils/exportFilename"

const props = withDefaults(
  defineProps<{
    episode: EpisodeDetail
    insights: Insight[]
    topics: Topic[]
    persons: Entity[]
    slug: string
    activeInsightId: string | null
    /** An insight tapped from the transcript — scroll it into view + highlight it. */
    focusInsightId?: string | null
    /** Opened from a note's "Open" (operator 2026-10-04) — land on the notes, not the panel top. */
    focusNotes?: boolean
  }>(),
  { focusInsightId: null, focusNotes: false }
)
const emit = defineEmits<{
  (e: "seek", seconds: number): void
  (e: "close"): void
  /**
   * Announce a capture outcome through the parent's live region (S8).
   *
   * Saving an insight announced NOTHING — the bookmark filling was the only feedback, and on
   * failure it does not fill, so a screen-reader user got silence either way. Emitting rather than
   * adding a second `aria-live` region: two live regions on one page compete, and PlayerView
   * already owns one.
   */
  (e: "announce", message: string): void
}>()

const { t } = useI18n()

/**
 * The summary is `summary_text`. No fallback to `summary_title`.
 *
 * This block sits on the SAME screen as the Summary panel, which shows the prose. With the fallback
 * an episode carrying only a headline showed the headline here and nothing there — two panels, one
 * player, two different answers to "what is the summary". `summary_title` is a headline; it is not
 * a short summary.
 */
/**
 * The episode-notes export — the whole episode as a document, not the Library's capture export.
 *
 * Built from the same API base as every other call so it follows the configured backend rather
 * than assuming an origin.
 */
function notesUrl(ext: 'md' | 'html'): string {
  return episodeNotesUrl(props.episode.slug, ext)
}

/**
 * The print-styled notes.
 *
 * On NATIVE this now OPENS them, in the app (operator 2026-09-27: "when I click a PDF, he offers
 * me to download HTML rather than opening me PDF in a new browser window"). It used to go straight
 * to the share sheet — which is a SAVE dialog. Reasonable if you wanted the file; wrong as the
 * answer to a control the reader takes to mean "show me the document".
 *
 * Note what this is NOT: it is not the external-browser route, which was tried and failed.
 * `openExternal` opens SFSafariViewController, which does not share the app's cookie jar, so the
 * export arrived unauthenticated and rendered the sign-in gate — the operator's tell at the time
 * was "when I copy the link and open it in a normal browser, it works fine", because that browser
 * had a session. The document here is FETCHED by the app (`apiFetch`, carrying the shell's bearer
 * token) and then displayed from memory. There is no second request, so there is nothing to
 * authenticate twice.
 *
 * Sharing stays one tap away INSIDE the viewer, because the share sheet is the route to iOS's
 * Print -> Save as PDF, and that is the real print-to-PDF path on the platform. Bundling a PDF
 * library to re-implement a renderer the OS already has would still be the wrong trade.
 *
 * Web keeps opening a tab: the cookie travels there, and a browser tab is already the thing the
 * native side is approximating.
 */
const printingNotes = ref(false)
const notesHtml = ref<string | null>(null)
const notesError = ref(false)
/**
 * Where the viewer mounts — the OPEN DIALOG when there is one, else `body`.
 *
 * This panel is `showModal()`'d on mobile, so it lives in the top layer, and the top layer paints
 * above everything in the normal layer regardless of z-index. Teleporting the viewer to `body` put
 * it behind the panel: rendered, correct, invisible. Resolved at open rather than at setup, because
 * whether a dialog is up depends on the route the reader took to get here.
 */
const notesTeleportTarget = ref<HTMLElement | string>('body')

async function openPrintableNotes(): Promise<void> {
  if (!isNative()) {
    await openExternal(notesUrl('html'))
    return
  }
  if (printingNotes.value) return
  printingNotes.value = true
  notesError.value = false
  notesTeleportTarget.value = sheetTeleportTarget()
  try {
    notesHtml.value = await fetchEpisodeNotes(props.episode.slug, 'html')
  } catch {
    // A failed export has to SAY so. Silence reads as a dead control — the same failure mode as
    // the `<a download>` this button replaced, which did nothing at all on the phone.
    notesError.value = true
  } finally {
    printingNotes.value = false
  }
}

/** Hand the already-fetched document to the share sheet — the way to iOS Print -> Save as PDF. */
async function shareOpenNotes(): Promise<void> {
  if (!notesHtml.value) return
  await saveAndShareText(
    exportFilename(`${props.episode.title} notes`, 'html', 'episode-notes'),
    notesHtml.value,
    'text/html',
  )
}

/**
 * Markdown notes on native: fetch and hand to the share sheet.
 *
 * The template's `<a download>` is ignored by WKWebView, so tapping "Markdown" on the phone
 * produced nothing — found alongside the PDF defect above. Both formats of the episode-notes export
 * were unreachable on the only build the operator actually uses.
 */
const savingNotes = ref(false)
async function saveNotesNative(): Promise<void> {
  if (savingNotes.value) return
  savingNotes.value = true
  try {
    const md = await fetchEpisodeNotes(props.episode.slug)
    await saveAndShareText(exportFilename(`${props.episode.title} notes`, 'md', 'episode-notes'), md)
  } finally {
    savingNotes.value = false
  }
}

const summary = computed(() => props.episode.summary_text || null)

/**
 * The episode-level digest, and the ONE place it renders (#2004 follow-up).
 *
 * The bullets are valuable and had no home: the browse card counts them without showing them, and
 * the Summary panel is the prose alone. They belong here, between the summary and the insights,
 * because that is the order of the panel's argument — what the episode is about, the shape of it,
 * then the moments it is built from. A digest next to its evidence.
 */
const summaryBullets = computed(() => props.episode.summary_bullets ?? [])
const hasAnything = computed(
  () =>
    Boolean(summary.value) ||
    props.topics.length > 0 ||
    props.persons.length > 0 ||
    props.insights.length > 0
)

// --- Topics + People as one compact, expandable row; topics cluster-first (RFC-102) ---
type Tag = {
  key: string
  label: string
  kind: "topic" | "person"
  dominant: boolean
  storylineMember: boolean
  /** Person only: aggregate speaker role (host/guest/mentioned), raw; localized at render. */
  role?: string
  /**
   * Person only: this identity is scoped to ONE episode (#2062), e.g. a guest known to the
   * transcript only by a bare first name. The chip renders as a <span>, not a button: there is
   * no corpus-wide entity behind it to open, so it is shown but not followable.
   */
  episodeScoped: boolean
}

// Tapping a chip opens its entity card (PRD-043; library search now lives inside the card).
const cardTarget = ref<{ kind: "person" | "topic"; id: string } | null>(null)
// The card REPLACES the panel body, so closing it rebuilt the panel at the top — far from the
// people or topics row the card was opened from. Remember the offset; put it back on close
// (operator 2026-10-04).
const panelBodyEl = ref<HTMLElement | null>(null)
let panelScrollBeforeCard = 0
function showCard(kind: "person" | "topic", id: string): void {
  panelScrollBeforeCard = panelBodyEl.value?.scrollTop ?? 0
  cardTarget.value = { kind, id }
}
function openCard(tag: Tag): void {
  showCard(tag.kind, tag.key)
}
function closeCard(): void {
  cardTarget.value = null
  void nextTick(() => restoreScroll(panelBodyEl.value, panelScrollBeforeCard))
}

// How many of THIS episode's topics fall in each corpus cluster (intra-episode dominance).
const topicClusterCounts = computed<Record<string, number>>(() => {
  const c: Record<string, number> = {}
  for (const t of props.topics) if (t.cluster_id) c[t.cluster_id] = (c[t.cluster_id] ?? 0) + 1
  return c
})
// The dominant cluster = the one with the most of this episode's topics (≥2), tie → larger corpus
// cluster; null when no topic is clustered or none reaches 2 (then it's a flat list).
const dominantClusterId = computed<string | null>(() => {
  const counts = topicClusterCounts.value
  let best: string | null = null
  let bestCount = 1
  let bestSize = -1
  for (const t of props.topics) {
    if (!t.cluster_id) continue
    const n = counts[t.cluster_id] ?? 0
    if (n > bestCount || (n === bestCount && t.cluster_size > bestSize)) {
      best = t.cluster_id
      bestCount = n
      bestSize = t.cluster_size
    }
  }
  return best
})

// Theme clusters (co-occurrence "discussed together") — parallel to the semantic dominant above.
// Marked on the pills (theme ring) + a "Storyline ·" lead-in. No-op when topics carry no storyline_id.
const storylineCounts = computed<Record<string, number>>(() => {
  const c: Record<string, number> = {}
  for (const t of props.topics)
    if (t.storyline_id) c[t.storyline_id] = (c[t.storyline_id] ?? 0) + 1
  return c
})
const storylineDominantId = computed<string | null>(() => {
  const counts = storylineCounts.value
  let best: string | null = null
  let bestCount = 1
  let bestSize = -1
  for (const t of props.topics) {
    if (!t.storyline_id) continue
    const n = counts[t.storyline_id] ?? 0
    if (n > bestCount || (n === bestCount && (t.storyline_size ?? 0) > bestSize)) {
      best = t.storyline_id
      bestCount = n
      bestSize = t.storyline_size ?? 0
    }
  }
  return best
})
const storylineDominantLabel = computed(
  () =>
    props.topics.find((t) => t.storyline_id === storylineDominantId.value)?.storyline_label ??
    null
)
/**
 * A topic id that opens the storyline named by the lead-in (operator 2026-09-19).
 *
 * ANY MEMBER topic works, which is why this does not need the anchor id the theme-cluster artifact
 * carries: "a storyline has no dedicated endpoint, so StorylineView reconstructs the whole theme
 * cluster from any member topic's card" (StorylineCard). Routing with the `thc:` cluster id instead
 * is what 404s, so it is deliberately not used here.
 */
const storylineDominantTopicId = computed<string | null>(
  () => props.topics.find((t) => t.storyline_id === storylineDominantId.value)?.id ?? null
)
const storylineOpen = ref(false)
/**
 * The dominant THEME, named beside the storyline (operator 2026-10-04: "don't forget THEME").
 * The panel already ranked topics by it and ringed its members, but never said which theme that
 * was — so the ring explained nothing. Opens as a sheet ON TOP, like the storyline, because
 * `router.push` from inside this top-layer dialog changes the page underneath and looks dead.
 */
const dominantThemeLabel = computed(
  () => props.topics.find((t) => t.cluster_id === dominantClusterId.value)?.cluster_label ?? null
)
const themeOpen = ref(false)
// Speaker-role badge on person chips (BE.4/PL.2) — same host/guest/mentioned vocabulary and i18n
// keys as EntityCardBody, so the label reads identically wherever a person appears.
const ROLE_LABEL_KEYS: Record<string, string> = {
  host: "ec.roleHost",
  guest: "ec.roleGuest",
  mentioned: "ec.roleMentioned",
}
function roleLabel(role: string | undefined): string {
  if (!role) return ""
  // Known role → localized; an unrecognized one falls back to its raw string (same idiom as
  // TrendingSparkChips), so a new server role still shows something rather than vanishing.
  const key = ROLE_LABEL_KEYS[role.toLowerCase()]
  return key ? t(key) : role
}
/** Chip order for people: host, guest, then anyone else (mentioned, or an unlabelled role). */
const PERSON_ROLE_ORDER: Record<string, number> = { host: 0, guest: 1 }
const personRank = (p: { role?: string | null }): number =>
  PERSON_ROLE_ORDER[(p.role ?? "").toLowerCase()] ?? 2

/** The people in the room, for the dossier line — host and guest only, in that order. */
const dossierPeople = computed(() =>
  props.persons
    .filter((p) => personRank(p) < 2)
    .slice()
    .sort((a, b) => personRank(a) - personRank(b))
)

const allTags = computed<Tag[]>(() => {
  const counts = topicClusterCounts.value
  const dom = dominantClusterId.value
  // Rank: dominant cluster first, then other clustered (larger intra-episode groups earlier),
  // then singletons. Stable sort keeps original order within a rank.
  const rank = (t: { cluster_id: string | null }): number =>
    t.cluster_id === dom && dom ? 0 : t.cluster_id ? 100 - (counts[t.cluster_id] ?? 0) : 1000
  const topics = [...props.topics].sort((a, b) => rank(a) - rank(b))
  // People read in conversation order — host, then guest, then everyone merely mentioned
  // (operator 2026-09-19). The server returns them in graph order, which put the guest first as
  // often as not; "who is this episode" is answered by the two people actually in the room.
  const persons = [...props.persons].sort((a, b) => personRank(a) - personRank(b))
  return [
    ...topics.map((tp) => ({
      key: tp.id,
      label: tp.label,
      kind: "topic" as const,
      dominant: Boolean(dom) && tp.cluster_id === dom,
      storylineMember: Boolean(tp.storyline_id),
      episodeScoped: false,
    })),
    ...persons.map((p) => ({
      key: p.id,
      label: p.name,
      kind: "person" as const,
      dominant: false,
      storylineMember: false,
      role: p.role ?? undefined,
      // #1685/#2062: a person identified only within this episode has no corpus-wide entity, so
      // the chip shows (she IS the guest) but does not offer a tap into an empty card.
      episodeScoped: p.episode_scoped === true,
    })),
  ]
})
/**
 * Topics & People render in full — no collapse (#2004 item 15).
 *
 * They used to clip at 6 behind a `+N …` expander. Tags are CHIPS: they wrap, so twenty of them
 * cost a few rows, and collapsing at six bought a little vertical space in exchange for hiding most
 * of the list on a panel whose whole job is showing what an episode is about.
 *
 * The insight list below still collapses (`INSIGHT_COLLAPSED`), and deliberately so — those are full
 * cards, and an episode with 36 of them would bury everything under it. The two are not the same
 * shape and are not made "consistent" with each other.
 */
const visibleTags = computed(() => allTags.value)

/**
 * `unknown` renders NO type label at all.
 *
 * A row labelled "UNKNOWN" spends a line to tell the reader nothing, and it is the one value that
 * carries no meaning to convey — it exists because the classifier could not decide. The insight
 * itself still renders; only the empty label is dropped.
 */
/**
 * What the type MEANS, for the hover tooltip.
 *
 * Falls back to a generic line rather than an empty title: a tooltip that opens blank reads as a
 * broken tooltip, and the vocabulary can legitimately carry a value this build predates.
 */
function insightTypeHint(ins: { insight_type?: string | null }): string {
  const type = insightTypeLabel(ins)
  const key = `kp.insightType.${type}`
  const hint = t(key)
  return hint === key ? t("kp.insightType.other") : hint
}

function insightTypeLabel(ins: { insight_type?: string | null }): string {
  const t = (ins.insight_type ?? "").toLowerCase()
  return t && t !== "unknown" ? t : ""
}

// ADR-135/#1191: the player shows `surface`-tagged insights — attributed to a named speaker. The
// server's `surfaceable` gate already excludes `connect` (UNATTRIBUTED insights, no named speaker),
// so this client filter is defensive, NOT a quality call: the 2026-07-29 eval found `connect`
// insights score HIGHER than `surface` (3.12 vs 2.99), so routing_tag is not a quality signal — it's
// about attribution. `drop` is excluded server-side; a null tag = pre-3.1 corpus, kept for back-
// compat. Server returns them salience-sorted; we preserve order and cap at gi_surface_default_limit
// (8) — the eval showed ranks 6-8 are as good as the top-5, so 8 (not 6) is the fold.
const INSIGHT_COLLAPSED = 8
// #2198: `attributed === false` insights are routed `connect` (no named speaker) yet shown — the
// server sends them only as the fallback for an episode whose speakers were never named, so they
// never appear beside named ones. Without this an episode with 44 grounded insights showed none.
const surfaceInsights = computed(() =>
  props.insights.filter(
    (i) => i.routing_tag == null || i.routing_tag === "surface" || i.attributed === false
  )
)
// Per-type filter (IN.3): null = all. Chips render only for the types actually present.
const insightTypeFilter = ref<string | null>(null)
/**
 * The types present, each with how many insights carry it (operator 2026-09-19).
 *
 * Counted off `surfaceInsights` — the same list the filter narrows — so a chip's number is exactly
 * what tapping it yields. Note this is a different cut of the same total from the INSIGHT DENSITY
 * bars above, which split by POSITION (early/mid/late); both sum to the same count.
 */
const insightTypeOptions = computed<{ type: string; count: number }[]>(() => {
  const counts = new Map<string, number>()
  for (const i of surfaceInsights.value) {
    const ty = insightTypeLabel(i)
    if (ty) counts.set(ty, (counts.get(ty) ?? 0) + 1)
  }
  return [...counts].map(([type, count]) => ({ type, count }))
})
const typeFilteredInsights = computed(() =>
  insightTypeFilter.value
    ? surfaceInsights.value.filter((i) => insightTypeLabel(i) === insightTypeFilter.value)
    : surfaceInsights.value
)
const showAll = ref(false)
const visibleInsights = computed(() =>
  showAll.value
    ? typeFilteredInsights.value
    : typeFilteredInsights.value.slice(0, INSIGHT_COLLAPSED)
)

// Scroll a transcript-tapped insight into view (and reveal it past the 5-item fold).
const insightEls = ref<Record<string, HTMLElement>>({})
watch(
  () => props.focusInsightId,
  async (id) => {
    if (!id) return
    showAll.value = true
    await nextTick()
    // rAF so the panel (and on mobile, its open transition) has laid out before we centre —
    // scrollIntoView walks every scroll ancestor, bringing the claim into the viewport too.
    requestAnimationFrame(() => {
      insightEls.value[id]?.scrollIntoView({ behavior: scrollBehavior(), block: "center" })
    })
  }
)

// The notes sit at the very bottom of the panel, below every rail; scroll the panel's own body to
// them once they are rendered.
watch(
  () => props.focusNotes,
  async (on) => {
    if (!on) return
    await nextTick()
    const notes = await waitForSettledElement(NOTES_ANCHOR)
    if (!notes) return
    // Instant, then held: rails above the notes can still arrive and push them down, and a smooth
    // scroll would animate toward where they WERE (see the router's anchor branch).
    const body = panelBodyEl.value
    if (!body) return notes.scrollIntoView({ block: "start" })
    body.scrollTop = offsetWithin(body, notes)
    holdScroll(body, () => offsetWithin(body, notes))
  },
  { immediate: true }
)

// --- ask (extractive grounded search) ---
const q = ref("")
const results = ref<SearchHit[]>([])
const searching = ref(false)
const askError = ref(false)
const searchInput = ref<HTMLInputElement | null>(null)

/**
 * Collapse hits whose text is identical.
 *
 * The operator's screenshot showed the SAME chunk returned twice, filling a phone screen that only
 * has room for about one result. This is presentation, NOT a fix: duplicate chunks in the index are
 * an index-side defect, in the same neighbourhood as the chunking bug in #2159 (a boundaryless
 * transcript became one 44,924-char chunk, and overlapping chunks return near-identical text).
 * Collapsing them here stops the reader paying for it; it does not stop it happening, and the
 * duplicate is still in the index for whoever picks that up.
 *
 * Keyed on trimmed text rather than `doc_id`, precisely because the ids differ — identical ids
 * would have been deduped by the backend already.
 */
function dedupeByText(hits: SearchHit[]): SearchHit[] {
  const seen = new Set<string>()
  return hits.filter((h) => {
    const key = (h.text ?? '').trim()
    if (!key || seen.has(key)) return seen.has(key) ? false : true
    seen.add(key)
    return true
  })
}

async function runSearch(): Promise<void> {
  const query = q.value.trim()
  if (!query) {
    results.value = []
    return
  }
  /*
   * Drop the keyboard before the results land (operator 2026-09-27).
   *
   * Results render BELOW the input, and on a phone the keyboard covers most of the panel — so a
   * search you had just run showed you roughly one hit of however many it found. Blurring on
   * submit dismisses it, which is also what the platform expects once a query is committed: the
   * text field has done its job.
   */
  searchInput.value?.blur()
  searching.value = true
  askError.value = false
  try {
    const resp = await searchEpisode(props.slug, query)
    results.value = dedupeByText(resp.results)
    askError.value = Boolean(resp.error)
  } catch {
    askError.value = true
  } finally {
    searching.value = false
  }
}

// --- capture (P2, PRD-040): save a grounded insight to the personal highlights corpus ---
const capture = useCaptureStore()
const savedInsightIds = computed(() => capture.savedInsightIds)
/** Auth-gated: a signed-out tap routes to sign-in rather than POSTing a 401 (#1590). */
const captureInsight = (ins: Insight) =>
  gated(async () => {
    const secs = insightStartSeconds(ins)
    const saved = savedInsightIds.value.has(ins.id)
    const ok = await capture.captureInsight(props.slug, {
      id: ins.id,
      text: ins.text,
      start_ms: secs != null ? Math.round(secs * 1000) : null,
    })
    emit(
      "announce",
      ok ? t(saved ? "capture.removed" : "capture.savedInsight") : t("capture.saveFailed")
    )
  })()

// --- related ("more like this") ---
const auth = useAuthStore()
const { isGated, gated } = useSignInGate()

const related = ref<EpisodeSummary[]>([])
async function loadRelated(slug: string): Promise<void> {
  try {
    related.value = (await getRelated(slug)).items.filter((e) => e.slug !== slug)
  } catch {
    related.value = []
  }
}
// Same floating-promise leak as PlayerView's: a failing GET /highlights became an unhandled
// rejection in the browser. Nothing here awaits it — the panel renders from an empty store — so
// catch and leave it un-loaded, which lets the next call retry.
const loadCaptures = (): void => {
  if (auth.isAuthenticated) void capture.ensureLoaded().catch(() => {})
}
onMounted(() => {
  loadRelated(props.slug)
  loadCaptures()
})
watch(
  () => props.slug,
  (s) => {
    // Clear the insight-type filter when the EPISODE changes.
    //
    // The panel is not remounted between episodes — PlayerView passes a new `slug` prop — so a
    // filter set on episode A survived into episode B. If B had no insights of that type the user
    // got the "Insights" heading, the chip strip, and an empty list with no explanation, and the
    // only escape was tapping "All", which nobody would think to do. The filter is a property of
    // the episode you are reading, not of the session (review 2026-09-19).
    insightTypeFilter.value = null
    return loadRelated(s)
  }
)
watch(() => auth.isAuthenticated, loadCaptures)
</script>

<template>
  <aside class="flex h-full flex-col bg-surface" :aria-label="t('kp.title')">
    <!-- Mobile bottom-sheet grab handle (signals the player sits behind; hidden on desktop rail). -->
    <div class="flex shrink-0 justify-center pt-2 lg:hidden" aria-hidden="true">
      <span class="h-1.5 w-10 rounded-full bg-border"></span>
    </div>
    <!-- Replace-in-panel (UXS-014): a tapped chip swaps the panel content to the entity card with a
         ‹ Back — no overlay, no second backdrop.
         `can-layer` is nonetheless TRUE, and the two are not in conflict. Replace-in-panel governs
         what a tapped CHIP does inside this panel; it never meant nothing may sit above the panel.
         This panel is itself a full-height bottom sheet, so a storyline or person opened from the
         card it is showing stacks ON TOP and leaves that card's kicker + title visible behind —
         topic underneath, storyline on it, person on that (operator 2026-09-16). Deriving it from
         `dismissAtRoot` instead sent the storyline to a PAGE, which loses the topic altogether. -->
    <EntityCardBody
      v-if="cardTarget"
      variant="inline"
      can-layer
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="closeCard"
    />
    <template v-else>
      <header class="flex items-center justify-between border-b border-border px-4 py-3">
        <span class="font-display text-lg font-bold">{{ t("kp.title") }}</span>
        <!-- Same ✕ idiom as the topic/person cards (lp-nav) — the bare button showed the default
             accent focus outline ("yellow frame") the others don't (operator, IMG_7093). -->
        <button
          type="button"
          class="lp-nav shrink-0"
          :aria-label="t('kp.close')"
          @click="emit('close')"
        >
          <CloseIcon />
        </button>
      </header>

      <div ref="panelBodyEl" class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <!-- WHICH EPISODE THIS IS (operator 2026-09-19).

             The panel carries the summary, the topics, the people and every insight — it is the
             episode's dossier — but it opened with a generic "Insights" header and an Ask box, so
             the document never said what it was about. The title leads; the show is a kicker above
             it (smaller, the same subordinate relationship the episode rows use); the people in the
             room follow, host first.

             In the body rather than the sticky header: the header is a fixed-height strip shared
             with the ✕, and a two-line episode title in it would either clip or push the close
             control around. -->
        <section class="mb-5" data-testid="kp-episode-dossier">
          <p v-if="episode.podcast_title" class="lp-kicker lp-show-name mb-0.5 text-muted" :title="episode.podcast_title">
            {{ episode.podcast_title }}
          </p>
          <h2 class="font-display text-xl font-bold leading-tight text-canvas-foreground">
            {{ episode.title }}
          </h2>
          <!-- Host + guest only. "Mentioned" people are already in the Topics & People chips and
               would turn the room into a crowd.

               Each with their photo (operator 2026-09-30) — the same ProfileAvatar, and so the same
               crop, as Top voices; initials when the enricher has no photo. A tap opens the person
               exactly as their chip below does (replace-in-panel, ‹ Back), so the two ways in stack
               identically. An episode-scoped person has no corpus-wide card: shown, not tappable. -->
          <ul v-if="dossierPeople.length" class="mt-3 flex flex-wrap gap-x-4 gap-y-2">
            <li v-for="p in dossierPeople" :key="p.id">
              <component
                :is="p.episode_scoped ? 'span' : 'button'"
                :type="p.episode_scoped ? undefined : 'button'"
                class="flex items-center gap-2 text-left"
                :aria-label="p.episode_scoped ? undefined : t('kp.openEntity', { term: personName(p.name) })"
                data-testid="kp-dossier-person"
                @click="p.episode_scoped ? undefined : showCard('person', p.id)"
              >
                <ProfileAvatar :name="personName(p.name)" :src="p.image_url" :size="32" />
                <span class="flex flex-col leading-tight">
                  <span class="text-sm font-medium text-canvas-foreground">{{ personName(p.name) }}</span>
                  <span class="lp-kicker text-muted">{{ roleLabel(p.role ?? undefined) }}</span>
                </span>
              </component>
            </li>
          </ul>
        </section>

        <p v-if="!hasAnything" class="text-sm text-muted">{{ t("kp.empty") }}</p>

        <!--
        The SUMMARY is not collapsible.

        It is the reason the panel was opened and it is a paragraph, not a list — folding it would
        save almost nothing and hide the one thing everybody wants. The sections below it are long,
        repetitive, or both, which is what makes folding them worth a tap.
      -->
        <section v-if="summary" class="mb-5">
          <h3 class="lp-section mb-1">{{ t("kp.summary") }}</h3>
          <p class="text-sm leading-relaxed text-surface-foreground">{{ summary }}</p>
        </section>

        <!-- Export the whole episode as notes (operator 2026-09-18).

             This panel IS the document — summary, key points, topics, everything said — so the
             export belongs here rather than in the Library, which exports captures across every
             episode. Two formats, matching the Library's chips: Markdown to keep, and a
             print-styled page the browser saves as PDF. -->
        <div class="mb-5 flex items-center gap-2" data-testid="episode-notes-export">
          <span class="text-xs text-muted">{{ t("kp.exportKicker") }}</span>
          <!-- Native shell: fetch + share sheet (WKWebView ignores `<a download>`, so this chip
               did nothing at all on the phone); web: plain download link. Mirrors the Library's
               highlights export, which already had the native branch this one was missing. -->
          <button
            v-if="isNative()"
            type="button"
            :disabled="savingNotes"
            :aria-label="t('kp.exportNotesMarkdown')"
            class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
            @click="saveNotesNative"
          >{{ t("kp.exportMarkdownShort") }}</button>
          <a
            v-else
            :href="notesUrl('md')"
            :download="`${episode.slug}-notes.md`"
            :aria-label="t('kp.exportNotesMarkdown')"
            class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent no-underline transition hover:bg-overlay"
          >{{ t("kp.exportMarkdownShort") }}</a>
          <button
            type="button"
            :disabled="printingNotes"
            :aria-label="t('kp.exportNotesPdf')"
            data-testid="episode-notes-pdf"
            class="whitespace-nowrap rounded-full border border-border px-2.5 py-1 text-xs font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
            @click="openPrintableNotes"
          >{{ t("kp.exportPdfShort") }}</button>
          <!-- The export can fail (offline, a dead session), and it used to fail in silence. -->
          <span v-if="notesError" class="text-xs text-danger" data-testid="episode-notes-error">
            {{ t("kp.exportFailed") }}
          </span>
        </div>

        <!--
          The notes, OPEN, on native (operator 2026-09-27).

          TELEPORTED INTO THE OPEN DIALOG, NOT INTO `body`. This panel is a `<dialog>` opened with
          `showModal()` on mobile (`PlayerView.vue`), which puts it in the TOP LAYER — and the top
          layer paints above the whole normal layer no matter what z-index anything there carries.
          The first version of this teleported to `body` with `z-[60]`; the notes fetched, the
          overlay rendered, and it sat invisible BEHIND the panel. The operator's report was "on
          last deploy nothing happens when I click PDF on insights", and nothing is exactly what it
          looked like. My own comment here named the hazard and then did the opposite of what it
          said.

          `sheetTeleportTarget()` is the existing answer to this — it returns `dialog[open]` when
          there is one and `body` otherwise, and `EntityCard` already uses it for the same reason.
          Resolved per open, because whether a dialog is up depends on how you got here.

          None of my three tests caught it: jsdom implements neither the top layer nor `showModal`
          stacking, so an element hidden behind a modal is indistinguishable there from one on top
          of it. The device tier is the only place this is observable, and it does not run in CI.

          An `<iframe srcdoc>` is what makes this work without a second request. The export is a
          COMPLETE standalone document — its own `<html>`, its own print stylesheet — so injecting
          it into this page would both break the page's styling and lose the print styling that is
          the entire point of the .html format. An iframe gives it its own document, and `srcdoc`
          means the bytes we already fetched with the shell's bearer token are the bytes rendered:
          no URL for SFSafariViewController to re-request without a cookie, which is exactly how
          the previous attempt at "open it" ended up on the sign-in gate.

          `sandbox` with nothing granted: the document is ours, but it is assembled from episode
          content, and a viewer has no reason to run script or navigate anywhere.
        -->
        <Teleport :to="notesTeleportTarget">
          <div
            v-if="notesHtml"
            class="fixed inset-0 z-[60] flex flex-col bg-canvas"
            role="dialog"
            aria-modal="true"
            :aria-label="t('kp.exportNotesPdf')"
            data-testid="episode-notes-viewer"
          >
            <div
              class="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 pb-2 pt-[max(0.5rem,env(safe-area-inset-top))]"
            >
              <button
                type="button"
                class="rounded-full border border-border px-3 py-1.5 text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
                data-testid="episode-notes-viewer-close"
                @click="notesHtml = null"
              >
                {{ t("kp.exportClose") }}
              </button>
              <!-- The share sheet is still here, because it is the route to Print -> Save as PDF.
                   It is an action WITHIN the document now, not the whole answer to opening it. -->
              <button
                type="button"
                class="rounded-full border border-border px-3 py-1.5 text-sm font-bold text-accent transition hover:bg-overlay"
                data-testid="episode-notes-viewer-share"
                @click="shareOpenNotes"
              >
                {{ t("kp.exportShare") }}
              </button>
            </div>
            <iframe
              :srcdoc="notesHtml"
              sandbox=""
              class="min-h-0 flex-1 w-full border-0 bg-white"
              :title="t('kp.exportNotesPdf')"
              data-testid="episode-notes-frame"
            />
          </div>
        </Teleport>

        <!--
          SEARCH, not "Ask" (operator 2026-09-27: "Ask episode doesn't feel right here").

          It was labelled Ask and it runs `searchEpisode()`, rendering ranked transcript chunks —
          `hit.text`, with `kp.noResults` when empty. There is no synthesis endpoint in the app:
          `app_search.py` exposes `GET /search` and nothing else. So the label promised an answer
          that nothing in the stack could produce, which is the whole of why it "didn't feel right".

          Renamed rather than built, deliberately and on the operator's call. Making it answer means
          a new backend route, per-question gateway cost, and deterministic fixtures to keep CI
          airgapped (LLMs in CI are banned). That is its own piece of work, not a label fix.

          The three sibling keys — `searching`, `searchError`, `noResults` — already said "search".
          Only the two user-facing ones lied, and the i18n KEYS were renamed too so the code stops
          carrying the fiction.

          Placed AFTER the summary and the download row, above the key points (operator
          2026-09-30): the panel opens on what the episode is — who is in it and what it says —
          and search is for digging into it once you know.
        -->
        <form class="mb-5" @submit.prevent="runSearch">
          <label class="sr-only" for="kp-ask">{{ t("kp.searchAction") }}</label>
          <div class="lp-search flex gap-2">
            <input
              id="kp-ask"
              ref="searchInput"
              v-model="q"
              type="search"
              :placeholder="t('kp.searchPlaceholder')"
              class="min-w-0 flex-1 rounded-full border border-border bg-canvas px-4 py-2 text-sm"
            />
            <button
              type="submit"
              class="rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground"
            >
              {{ t("kp.searchAction") }}
            </button>
          </div>
          <p v-if="searching" class="mt-2 text-sm text-muted">{{ t("kp.searching") }}</p>
          <p v-else-if="askError" class="mt-2 text-sm text-danger">{{ t("kp.searchError") }}</p>
          <ul v-else-if="results.length" class="mt-3 flex flex-col gap-2">
            <li
              v-for="hit in results"
              :key="hit.doc_id"
              class="rounded-xl border border-border p-3"
            >
              <p class="text-sm text-surface-foreground">{{ hit.text }}</p>
              <PlayFrom
                v-if="hitStartSeconds(hit) != null"
                class="mt-1 inline-block"
                :seconds="hitStartSeconds(hit)"
                @click="emit('seek', hitStartSeconds(hit) as number)"
              />
            </li>
          </ul>
          <p v-else-if="q.trim() && !searching" class="mt-2 text-sm text-muted">
            {{ t("kp.noResults") }}
          </p>
        </form>

        <!--
        The digest, under the summary and above the insights. Its own labelled block rather than
        loose text: ~8 sentences of ~200 characters on a real episode, which without a heading read
        as a second summary that disagrees with the first.
      -->
        <CollapsibleSection
          v-if="summaryBullets.length"
          :title="t('kp.keyPoints')"
          :count="summaryBullets.length"
          section-key="key-points"
          class="mb-5"
        >
          <!-- Key points as accent-ruled rows (IN.1): the old 1px grey dot read as faint noise; a
             short left rule gives each point weight and scans as a structured list. -->
          <ul data-testid="summary-bullets" class="flex flex-col gap-2.5">
            <li
              v-for="(b, i) in summaryBullets"
              :key="i"
              class="border-l-2 border-accent/50 pl-3 text-sm leading-relaxed text-surface-foreground"
            >
              {{ b }}
            </li>
          </ul>
        </CollapsibleSection>

        <!-- Topics & People — one compact, expandable row; topics cluster-first (RFC-102) -->
        <CollapsibleSection
          v-if="allTags.length"
          :title="t('kp.tags')"
          :count="allTags.length"
          section-key="tags"
          class="mb-5"
        >
          <!-- Storyline + similar context (IN.2): promoted from a cramped, right-aligned `text-xs`
             column to a clear left-aligned block, so the storyline (theme cluster) this episode's
             topics belong to reads at a glance rather than as fine print. -->
          <div
            v-if="storylineDominantLabel || (dominantClusterId && dominantThemeLabel)"
            class="mb-2 flex flex-wrap items-center gap-2"
          >
            <button
              v-if="dominantClusterId && dominantThemeLabel"
              type="button"
              data-testid="kp-theme-link"
              class="lp-tap inline-flex items-center gap-1.5 rounded-full bg-overlay px-2.5 py-1 text-xs font-semibold text-theme ring-1 ring-inset ring-theme/40 transition hover:bg-elevated"
              :aria-label="t('kp.openTheme', { label: dominantThemeLabel })"
              @click="themeOpen = true"
            >
              <span class="font-mono text-[10px] uppercase tracking-wide opacity-80">{{ t("kp.themeKind") }}</span>
              {{ dominantThemeLabel }}
            </button>
            <!-- The storyline OPENS (operator 2026-09-19): it is a real destination with its own
                 sheet, and reading its name without being able to go there was the gap. Falls back
                 to a plain <span> when no member topic id is available to route with. -->
            <!-- A PILL, in the accent, with its kind named (operator 2026-09-19).
                 It used to be an underlined text link stacked above an inert "Similar ·" line, in a
                 wall of identical grey topic pills — the one tappable thing in the section did not
                 look tappable, and the line above it looked equally tappable and was not.
                 It wears the storyline's OWN hue and tint (`--lp-storyline`, 2026-10-04), the same
                 pill a storyline wears everywhere — not the accent, which is not a kind colour.
                 The word STORYLINE rides along so the kind is NAMED, not inferred from colour —
                 colour alone reaches neither a colour-blind reader nor VoiceOver. -->
            <button
              v-if="storylineDominantLabel && storylineDominantTopicId"
              type="button"
              data-testid="kp-storyline-link"
              class="lp-tap lp-storyline-chip inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold text-storyline transition"
              :aria-label="t('kp.openStoryline', { label: storylineDominantLabel })"
              @click="storylineOpen = true"
            >
              <span class="font-mono text-[10px] uppercase tracking-wide opacity-80">{{ t("kp.storylineKind") }}</span>
              {{ storylineDominantLabel }}
            </button>
            <span
              v-else-if="storylineDominantLabel"
              class="inline-flex items-center gap-1.5 rounded-full bg-overlay px-2.5 py-1 text-xs font-semibold text-muted"
            >
              <span class="font-mono text-[10px] uppercase tracking-wide opacity-80">{{ t("kp.storylineKind") }}</span>
              {{ storylineDominantLabel }}
            </span>
          </div>
          <div class="flex flex-wrap gap-1.5">
            <!-- data-testid, not the colour class: specs used to select these with
               `button.text-topic`, which couples the test suite to styling — a restyle would break
               them for reasons unrelated to behaviour, and it was the cause of two flaky specs
               (consolidation, perspectives). Flagged in #1612. -->
            <component
              :is="tag.episodeScoped ? 'span' : 'button'"
              v-for="tag in visibleTags"
              :key="tag.key"
              :type="tag.episodeScoped ? undefined : 'button'"
              :data-episode-scoped="tag.episodeScoped ? 'true' : undefined"
              :data-testid="tag.kind === 'topic' ? 'kp-topic-chip' : 'kp-person-chip'"
              class="rounded-full px-2.5 py-1 text-xs transition"
              :class="[
                tag.kind === 'topic' ? 'text-topic' : 'text-person',
                tag.storylineMember
                  ? 'lp-storyline-chip'
                  : tag.dominant
                  ? 'bg-overlay ring-1 ring-inset ring-theme/60 hover:bg-elevated'
                  : 'bg-overlay hover:bg-elevated',
              ]"
              :aria-label="tag.episodeScoped ? undefined : t('kp.openEntity', { term: tag.label })"
              @click="tag.episodeScoped ? undefined : openCard(tag)"
            >
              <!-- Every pill in this MIXED group names its kind, the way the storyline pill above
                   does (operator 2026-10-04): naming one kind and leaving the rest to colour made
                   the one named pill look like the odd one out. Groups that sit under a kind
                   heading (show page, Library, Profile › Interests) carry no label — the heading
                   already says it. -->
              <span class="mr-1.5 font-mono text-[10px] uppercase tracking-wide opacity-80" data-testid="kp-chip-kind">{{
                t(tag.kind === "topic" ? "notes.kind_topic" : "notes.kind_person")
              }}</span>{{ tag.label
              }}<span
                v-if="roleLabel(tag.role)"
                data-testid="kp-person-role"
                :data-role="tag.role?.toLowerCase()"
                class="ml-1 rounded-full bg-canvas/50 px-1.5 py-0.5 text-[0.6rem] font-bold uppercase tracking-wide"
                >{{ roleLabel(tag.role) }}</span
              >
            </component>
          </div>
        </CollapsibleSection>

        <!-- Insights — the longest section by far (up to 36 rows), so the clearest thing to fold. -->
        <CollapsibleSection
          v-if="surfaceInsights.length"
          :title="t('kp.insights')"
          :count="surfaceInsights.length"
          section-key="insights"
          data-testid="kp-insights"
        >
          <!-- Where the substance sits (early/mid/late), tap to jump. Hides if absent. -->
          <EpisodeDensity :slug="slug" @seek="emit('seek', $event)" />
          <!-- Per-type filter (IN.3) — only shown when the episode has more than one insight type. -->
          <!-- ONE row that scrolls, not a wrapping block (operator 2026-09-19). The per-type counts
               widened every chip, so a fourth type pushed "Claim 9" alone onto a second line and
               the insight list below jumped down. `shrink-0` on the chips is load-bearing: without
               it flex squashes them to fit instead of overflowing, so the labels truncate rather
               than scroll. Same overflow pattern the rails use. -->
          <div
            v-if="insightTypeOptions.length > 1"
            class="mb-3 flex gap-1.5 overflow-x-auto pb-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            role="group"
            :aria-label="t('kp.filterByType')"
            data-testid="insight-type-filter"
          >
            <button
              type="button"
              class="shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold transition"
              :class="
                insightTypeFilter === null
                  ? 'bg-accent text-accent-foreground'
                  : 'bg-overlay text-muted hover:text-canvas-foreground'
              "
              @click="insightTypeFilter = null"
            >
              {{ t("kp.filterAll") }}
              <span class="ml-1 font-mono opacity-70">{{ surfaceInsights.length }}</span>
            </button>
            <button
              v-for="opt in insightTypeOptions"
              :key="opt.type"
              type="button"
              class="shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold capitalize transition"
              :class="
                insightTypeFilter === opt.type
                  ? 'bg-accent text-accent-foreground'
                  : 'bg-overlay text-muted hover:text-canvas-foreground'
              "
              @click="insightTypeFilter = opt.type"
            >
              {{ opt.type }}
              <!-- `font-mono` + not capitalized: the count is data, not part of the type's name. -->
              <span class="ml-1 font-mono normal-case opacity-70">{{ opt.count }}</span>
            </button>
          </div>
          <ul class="flex flex-col gap-3">
            <li
              v-for="ins in visibleInsights"
              :key="ins.id"
              :ref="(el) => { if (el) insightEls[ins.id] = el as HTMLElement }"
              class="rounded-xl border p-3 transition-colors"
              :class="
                ins.id === activeInsightId || ins.id === focusInsightId
                  ? 'border-border bg-overlay'
                  : 'border-border'
              "
            >
              <div class="flex items-center justify-between gap-2">
                <span class="flex items-center gap-1.5">
                  <!--
                  The type gets a SHAPE, not a colour (#2004 item 8).

                  Colouring the label would reverse the accent-discipline work that made
                  `.lp-kicker` mono + muted (#2013), and would put several hues back on a panel that
                  can hold 36 of these. A shape survives greyscale and colour-blindness, adds no
                  hue, and sits inside the existing kicker. It is `aria-hidden` because the type
                  word beside it already says the same thing — a screen reader should not hear
                  "diamond claim".

                  ## The green "grounded" dot that used to sit here is GONE

                  It rendered on `insightStartSeconds(ins) != null` — the exact condition that
                  renders the `▶ 3:32` button at the other end of this same row. So it never
                  distinguished one insight from another; it was on for every row that showed a
                  timestamp and absent from every row that did not, which the timestamp already
                  says, more precisely, in a form you can act on.

                  That made it worse than merely redundant. The complaint was that every insight
                  looked identical, and the answer was a type glyph — placed immediately after a
                  constant green dot, so the row still opened with the same mark every time and the
                  one differentiating character had to compete with it. Removing the dot is what
                  makes the glyph readable, which was the point of adding it.
                -->
                  <!--
                  The mark is a symbol, and a symbol nobody can decode is decoration. On a pointer
                  device the meaning is one hover away; the visible word already carries it for
                  everyone else, which is why the mark itself stays `aria-hidden` — a screen reader
                  should hear "claim", not "diamond claim".
                -->
                  <span
                    v-if="insightTypeLabel(ins)"
                    class="lp-kicker inline-flex items-center gap-1.5"
                    :title="insightTypeHint(ins)"
                    data-testid="insight-type"
                  >
                    <InsightTypeMark :type="insightTypeLabel(ins)" />
                    {{ insightTypeLabel(ins) }}
                  </span>
                </span>
                <span class="flex items-center gap-2">
                  <!-- The mm:ss is WHERE in the episode this insight was said — tapping jumps there.
                     Labelled so it isn't read as a bare, unexplained number (IN.4). -->
                  <PlayFrom
                    v-if="insightStartSeconds(ins) != null"
                    :seconds="insightStartSeconds(ins)"
                    :aria-label="t('kp.jumpToMoment', { time: formatTime(insightStartSeconds(ins) as number) })"
                    :title="t('kp.jumpToMoment', { time: formatTime(insightStartSeconds(ins) as number) })"
                    @click="emit('seek', insightStartSeconds(ins) as number)"
                  />
                  <!-- Save this insight — a BOOKMARK, like every other highlight (operator
                       2026-09-27).

                       It was a heart, the shared `FavoriteButton` in its `controlled` variant. The
                       destination was always right — it writes an insight highlight via the capture
                       store, never the favorites store, because favourite(insight) is banned
                       (#1593) — but the GLYPH said favourite, and the accessible name literally
                       said "Save to favorites". So the heart meant a favourite on an episode header
                       and a highlight here, while the identical action on a transcript line two
                       panels away drew a bookmark.

                       One glyph per concept now: bookmark = highlight, heart = favourite, and the
                       transcript line and this share `HighlightToggle` rather than agreeing by
                       coincidence. `label` is truncated insight text, so several on one panel do
                       not all announce identically (2026-09-25, Android tier). -->
                  <HighlightToggle
                    context="insight"
                    :label="ins.text.slice(0, 60)"
                    :saved="savedInsightIds.has(ins.id)"
                    :gated="isGated"
                    @toggle="captureInsight(ins)"
                  />
                </span>
              </div>
              <p class="mt-1 text-sm font-semibold text-surface-foreground">{{ ins.text }}</p>
              <blockquote
                v-if="ins.quotes[0]"
                class="mt-2 border-l-2 border-border pl-3 text-sm text-muted"
              >
                “{{ ins.quotes[0].text }}”
                <span v-if="speakerLabel(ins.quotes[0].speaker)" class="lp-speaker block">
                  — {{ speakerLabel(ins.quotes[0].speaker) }}
                </span>
              </blockquote>
              <!-- #2198: said, but by a voice we could not name. Say so, rather than implying the
                   episode's host or guest said it. -->
              <p
                v-if="ins.attributed === false"
                class="lp-kicker mt-2"
                data-testid="insight-unattributed"
              >
                {{ t("kp.speakerNotIdentified") }}
              </p>
            </li>
          </ul>
          <button
            v-if="!showAll && typeFilteredInsights.length > INSIGHT_COLLAPSED"
            type="button"
            data-testid="kp-insights-show-all"
            class="mt-3 text-sm font-bold text-accent"
            @click="showAll = true"
          >
            {{ t("kp.showAll") }}
          </button>
        </CollapsibleSection>

        <!-- More like this (semantic peers; hidden when the index has no neighbours). -->
        <CollapsibleSection
          v-if="related.length"
          :title="t('kp.related')"
          :count="related.length"
          section-key="related"
          class="mt-5"
        >
          <!-- The same rail and tile as the player page's "More like this" (operator 2026-10-05: same
               section name, same shape everywhere). It was a text list here, with a Play-next
               button the tile's shared action row does not carry. -->
          <CardRail>
            <li v-for="r in related" :key="r.slug" class="lp-rail-item">
              <EpisodeTile :episode="r" />
            </li>
          </CardRail>
        </CollapsibleSection>

        <!-- Your notes on this episode (NT.1) — episode-target notes, timestamped, dictation where
           the platform supports it. -->
        <NoteComposer target="episode" :target-id="slug" />
      </div>
    </template>

    <!-- The storyline named by the lead-in, opened ON TOP of this panel rather than replacing it —
         the same stacking the panel already documents for a card's storyline ("topic underneath,
         storyline on it"). Outside the `v-else` so it survives a chip swapping the body. -->
    <ThemeCard
      v-if="themeOpen && dominantClusterId"
      :id="dominantClusterId"
      :depth="1"
      @close="themeOpen = false"
    />
    <StorylineCard
      v-if="storylineOpen && storylineDominantTopicId"
      :id="storylineDominantTopicId"
      :depth="1"
      @close="storylineOpen = false"
    />
  </aside>
</template>
