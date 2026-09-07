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
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getRelated, searchEpisode } from '../services/api'
import type {
  EpisodeDetail,
  EpisodeSummary,
  Entity,
  Insight,
  SearchHit,
  Topic,
} from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { hitStartSeconds, insightStartSeconds } from '../player/insights'
import { speakerLabel } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import { useAuthStore } from '../stores/auth'
import { useSignInGate } from '../composables/useSignInGate'
import { scrollBehavior } from '../utils/motion'
import { useQueueStore } from '../stores/queue'
import { useCaptureStore } from '../stores/capture'
import InsightTypeMark from './InsightTypeMark.vue'
import EntityCardBody from './EntityCardBody.vue'
import EpisodeDensity from './EpisodeDensity.vue'


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
  }>(),
  { focusInsightId: null },
)
const emit = defineEmits<{
  (e: 'seek', seconds: number): void
  (e: 'close'): void
  /**
   * Announce a capture outcome through the parent's live region (S8).
   *
   * Saving an insight announced NOTHING — the bookmark filling was the only feedback, and on
   * failure it does not fill, so a screen-reader user got silence either way. Emitting rather than
   * adding a second `aria-live` region: two live regions on one page compete, and PlayerView
   * already owns one.
   */
  (e: 'announce', message: string): void
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
    props.insights.length > 0,
)

// --- Topics + People as one compact, expandable row; topics cluster-first (RFC-102) ---
type Tag = {
  key: string
  label: string
  kind: 'topic' | 'person'
  dominant: boolean
  themeMember: boolean
}

// Tapping a chip opens its entity card (PRD-043; library search now lives inside the card).
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)
function openCard(tag: Tag): void {
  cardTarget.value = { kind: tag.kind, id: tag.key }
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
const dominantClusterLabel = computed(
  () => props.topics.find((t) => t.cluster_id === dominantClusterId.value)?.cluster_label ?? null,
)

// Theme clusters (co-occurrence "discussed together") — parallel to the semantic dominant above.
// Marked on the pills (theme ring) + a "Theme ·" lead-in. No-op when topics carry no theme_cluster_id.
const themeClusterCounts = computed<Record<string, number>>(() => {
  const c: Record<string, number> = {}
  for (const t of props.topics)
    if (t.theme_cluster_id) c[t.theme_cluster_id] = (c[t.theme_cluster_id] ?? 0) + 1
  return c
})
const themeDominantId = computed<string | null>(() => {
  const counts = themeClusterCounts.value
  let best: string | null = null
  let bestCount = 1
  let bestSize = -1
  for (const t of props.topics) {
    if (!t.theme_cluster_id) continue
    const n = counts[t.theme_cluster_id] ?? 0
    if (n > bestCount || (n === bestCount && (t.theme_cluster_size ?? 0) > bestSize)) {
      best = t.theme_cluster_id
      bestCount = n
      bestSize = t.theme_cluster_size ?? 0
    }
  }
  return best
})
const themeDominantLabel = computed(
  () =>
    props.topics.find((t) => t.theme_cluster_id === themeDominantId.value)?.theme_cluster_label ??
    null,
)
const allTags = computed<Tag[]>(() => {
  const counts = topicClusterCounts.value
  const dom = dominantClusterId.value
  // Rank: dominant cluster first, then other clustered (larger intra-episode groups earlier),
  // then singletons. Stable sort keeps original order within a rank.
  const rank = (t: { cluster_id: string | null }): number =>
    t.cluster_id === dom && dom ? 0 : t.cluster_id ? 100 - (counts[t.cluster_id] ?? 0) : 1000
  const topics = [...props.topics].sort((a, b) => rank(a) - rank(b))
  return [
    ...topics.map((tp) => ({
      key: tp.id,
      label: tp.label,
      kind: 'topic' as const,
      dominant: Boolean(dom) && tp.cluster_id === dom,
      themeMember: Boolean(tp.theme_cluster_id),
    })),
    ...props.persons.map((p) => ({
      key: p.id,
      label: p.name,
      kind: 'person' as const,
      dominant: false,
      themeMember: false,
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
  return hint === key ? t('kp.insightType.other') : hint
}

function insightTypeLabel(ins: { insight_type?: string | null }): string {
  const t = (ins.insight_type ?? '').toLowerCase()
  return t && t !== 'unknown' ? t : ''
}

// ADR-135/#1191: the player shows `surface`-tagged insights — attributed to a named speaker. The
// server's `surfaceable` gate already excludes `connect` (UNATTRIBUTED insights, no named speaker),
// so this client filter is defensive, NOT a quality call: the 2026-07-29 eval found `connect`
// insights score HIGHER than `surface` (3.12 vs 2.99), so routing_tag is not a quality signal — it's
// about attribution. `drop` is excluded server-side; a null tag = pre-3.1 corpus, kept for back-
// compat. Server returns them salience-sorted; we preserve order and cap at gi_surface_default_limit
// (8) — the eval showed ranks 6-8 are as good as the top-5, so 8 (not 6) is the fold.
const INSIGHT_COLLAPSED = 8
const surfaceInsights = computed(() =>
  props.insights.filter((i) => i.routing_tag == null || i.routing_tag === 'surface'),
)
const showAll = ref(false)
const visibleInsights = computed(() =>
  showAll.value ? surfaceInsights.value : surfaceInsights.value.slice(0, INSIGHT_COLLAPSED),
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
      insightEls.value[id]?.scrollIntoView({ behavior: scrollBehavior(), block: 'center' })
    })
  },
)

// --- ask (extractive grounded search) ---
const q = ref('')
const results = ref<SearchHit[]>([])
const searching = ref(false)
const askError = ref(false)
async function runSearch(): Promise<void> {
  const query = q.value.trim()
  if (!query) {
    results.value = []
    return
  }
  searching.value = true
  askError.value = false
  try {
    const resp = await searchEpisode(props.slug, query)
    results.value = resp.results
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
      'announce',
      ok ? t(saved ? 'capture.removed' : 'capture.savedInsight') : t('capture.saveFailed'),
    )
  })()

// --- related ("more like this") ---
const auth = useAuthStore()
const { isGated, gated } = useSignInGate()
const queue = useQueueStore()
const epArt = episodeArtwork

// Queue a peer episode to play right after the current one (RFC-099 §4 "Play next").
/** Auth-gated: a signed-out tap routes to sign-in rather than POSTing a 401 (#1590). */
const playNext = (slug: string) =>
  gated(async () => {
    // The action reports whether the write survived (#1906); the gate's handler type is void.
    await queue.playNext(slug, props.slug)
  })()
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
watch(() => props.slug, (s) => loadRelated(s))
watch(() => auth.isAuthenticated, loadCaptures)
</script>

<template>
  <aside class="flex h-full flex-col bg-surface" :aria-label="t('kp.title')">
    <!-- Mobile bottom-sheet grab handle (signals the player sits behind; hidden on desktop rail). -->
    <div class="flex shrink-0 justify-center pt-2 lg:hidden" aria-hidden="true">
      <span class="h-1.5 w-10 rounded-full bg-border"></span>
    </div>
    <!-- Replace-in-panel (UXS-014): a tapped chip swaps the panel content to the entity card with a
         ‹ Back — no overlay, no second backdrop. -->
    <EntityCardBody
      v-if="cardTarget"
      variant="inline"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
    <template v-else>
    <header class="flex items-center justify-between border-b border-border px-4 py-3">
      <span class="font-display text-lg font-bold">{{ t('kp.title') }}</span>
      <button type="button" class="text-muted" :aria-label="t('kp.close')" @click="emit('close')">✕</button>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
      <!-- Ask -->
      <form class="mb-5" @submit.prevent="runSearch">
        <label class="sr-only" for="kp-ask">{{ t('kp.ask') }}</label>
        <div class="flex gap-2">
          <input
            id="kp-ask"
            v-model="q"
            type="search"
            :placeholder="t('kp.askPlaceholder')"
            class="min-w-0 flex-1 rounded-full border border-border bg-canvas px-4 py-2 text-sm"
          />
          <button type="submit" class="rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground">
            {{ t('kp.ask') }}
          </button>
        </div>
        <p v-if="searching" class="mt-2 text-sm text-muted">{{ t('kp.searching') }}</p>
        <p v-else-if="askError" class="mt-2 text-sm text-danger">{{ t('kp.searchError') }}</p>
        <ul v-else-if="results.length" class="mt-3 flex flex-col gap-2">
          <li v-for="hit in results" :key="hit.doc_id" class="rounded-xl border border-border p-3">
            <p class="text-sm text-surface-foreground">{{ hit.text }}</p>
            <button
              v-if="hitStartSeconds(hit) != null"
              type="button"
              class="mt-1 font-mono text-xs text-accent"
              @click="emit('seek', hitStartSeconds(hit) as number)"
            >
              ▶ {{ formatTime(hitStartSeconds(hit) as number) }}
            </button>
          </li>
        </ul>
        <p v-else-if="q.trim() && !searching" class="mt-2 text-sm text-muted">{{ t('kp.noResults') }}</p>
      </form>

      <p v-if="!hasAnything" class="text-sm text-muted">{{ t('kp.empty') }}</p>

      <!-- Summary -->
      <section v-if="summary || summaryBullets.length" class="mb-5">
        <h3 v-if="summary" class="lp-section mb-1">{{ t('kp.summary') }}</h3>
        <p v-if="summary" class="text-sm leading-relaxed text-surface-foreground">{{ summary }}</p>

        <!--
          The digest, under the summary and above the insights. Presented as its own labelled block
          rather than loose text: these are ~8 sentences of ~200 characters on a real episode, so
          without a heading they read as a second summary that disagrees with the first.
        -->
        <template v-if="summaryBullets.length">
          <h3 class="lp-section mb-1" :class="summary ? 'mt-4' : ''">{{ t('kp.keyPoints') }}</h3>
          <ul data-testid="summary-bullets" class="space-y-2">
            <li
              v-for="(b, i) in summaryBullets"
              :key="i"
              class="flex gap-2 text-sm leading-relaxed text-surface-foreground"
            >
              <span class="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-muted" aria-hidden="true" />
              <span>{{ b }}</span>
            </li>
          </ul>
        </template>
      </section>

      <!-- Topics & People — one compact, expandable row; topics cluster-first (RFC-102) -->
      <section v-if="allTags.length" class="mb-5">
        <div class="mb-2 flex items-baseline justify-between gap-2">
          <h3 class="lp-section">{{ t('kp.tags') }}</h3>
          <span
            v-if="themeDominantLabel || dominantClusterLabel"
            class="flex min-w-0 flex-col items-end text-xs leading-tight"
          >
            <span v-if="themeDominantLabel" class="truncate text-theme">
              {{ t('kp.theme', { cluster: themeDominantLabel }) }}
            </span>
            <span v-if="dominantClusterLabel" class="truncate text-topic">
              {{ t('kp.similar', { cluster: dominantClusterLabel }) }}
            </span>
          </span>
        </div>
        <div class="flex flex-wrap gap-1.5">
          <!-- data-testid, not the colour class: specs used to select these with
               `button.text-topic`, which couples the test suite to styling — a restyle would break
               them for reasons unrelated to behaviour, and it was the cause of two flaky specs
               (consolidation, perspectives). Flagged in #1612. -->
          <button
            v-for="tag in visibleTags"
            :key="tag.key"
            type="button"
            :data-testid="tag.kind === 'topic' ? 'kp-topic-chip' : 'kp-person-chip'"
            class="rounded-full px-2.5 py-1 text-xs transition"
            :class="[
              tag.kind === 'topic' ? 'text-topic' : 'text-person',
              tag.themeMember
                ? 'lp-theme-chip'
                : tag.dominant
                  ? 'bg-overlay ring-1 ring-topic hover:bg-elevated'
                  : 'bg-overlay hover:bg-elevated',
            ]"
            :aria-label="t('kp.openEntity', { term: tag.label })"
            @click="openCard(tag)"
          >
            {{ tag.label }}
          </button>
        </div>
      </section>

      <!-- Insights -->
      <section v-if="surfaceInsights.length" data-testid="kp-insights">
        <div class="mb-2 flex items-center justify-between">
          <h3 class="lp-section">{{ t('kp.insights') }} · {{ surfaceInsights.length }}</h3>
        </div>
        <!-- Where the substance sits (early/mid/late), tap to jump. Hides if absent. -->
        <EpisodeDensity :slug="slug" @seek="emit('seek', $event)" />
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
                <button
                  v-if="insightStartSeconds(ins) != null"
                  type="button"
                  class="font-mono text-xs text-accent"
                  @click="emit('seek', insightStartSeconds(ins) as number)"
                >
                  ▶ {{ formatTime(insightStartSeconds(ins) as number) }}
                </button>
                <!-- Save this insight to the personal highlights corpus (P2). Auth-gated means
                     deferred, not hidden (#1590): it renders signed-out and routes to sign-in. -->
                <button
                  type="button"
                  class="rounded-full p-0.5 transition"
                  :class="savedInsightIds.has(ins.id) ? 'text-accent' : 'text-muted hover:text-accent'"
                  :aria-pressed="isGated ? undefined : savedInsightIds.has(ins.id)"
                  :aria-label="isGated ? t('auth.signInToCapture') : savedInsightIds.has(ins.id) ? t('capture.savedInsight') : t('capture.saveInsight')"
                  :title="isGated ? t('auth.signInToCapture') : savedInsightIds.has(ins.id) ? t('capture.savedInsight') : t('capture.saveInsight')"
                  @click="captureInsight(ins)"
                >
                  <svg viewBox="0 0 24 24" :fill="savedInsightIds.has(ins.id) ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="2" class="h-4 w-4" aria-hidden="true">
                    <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
                  </svg>
                </button>
                <!-- #1593: the heart used to sit here too, saving the SAME insight to a SECOND
                     list (Library › Saved › Insights) while the bookmark above saved it to
                     Highlights. Same text, two icons, two destinations, two places to look for it
                     later. One save, one destination — and Highlights is the richer one: it carries
                     colours, notes and export. Existing insight-favourites stay readable in Library;
                     this only stops NEW ones being written. -->
              </span>
            </div>
            <p class="mt-1 text-sm font-semibold text-surface-foreground">{{ ins.text }}</p>
            <blockquote v-if="ins.quotes[0]" class="mt-2 border-l-2 border-border pl-3 text-sm text-muted">
              “{{ ins.quotes[0].text }}”
              <span v-if="speakerLabel(ins.quotes[0].speaker)" class="lp-speaker block">
                — {{ speakerLabel(ins.quotes[0].speaker) }}
              </span>
            </blockquote>
          </li>
        </ul>
        <button
          v-if="!showAll && surfaceInsights.length > INSIGHT_COLLAPSED"
          type="button"
          data-testid="kp-insights-show-all"
          class="mt-3 text-sm font-bold text-accent"
          @click="showAll = true"
        >
          {{ t('kp.showAll') }}
        </button>
      </section>

      <!-- More like this (semantic peers; hidden when the index has no neighbours). -->
      <section v-if="related.length" class="mt-5">
        <h3 class="lp-section mb-2">{{ t('kp.related') }}</h3>
        <ul class="flex flex-col">
          <li v-for="r in related" :key="r.slug" class="flex items-center gap-1 border-b border-border">
            <RouterLink
              :to="{ name: 'player', params: { slug: r.slug } }"
              class="flex min-w-0 flex-1 items-center gap-3 py-2 no-underline text-canvas-foreground hover:bg-overlay"
            >
              <img
                v-if="epArt(r)"
                :src="epArt(r)!"
                alt=""
                loading="lazy"
                class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
              />
              <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" />
              <span class="min-w-0 flex-1">
                <span class="block text-sm font-semibold">{{ r.title }}</span>
                <span v-if="r.podcast_title" class="lp-kicker block">{{ r.podcast_title }}</span>
              </span>
            </RouterLink>
            <!-- Play next: queue this peer right after the current episode (RFC-099 §4). Renders
                 signed-out and routes to sign-in (#1590). -->
            <button
              type="button"
              class="shrink-0 rounded-full p-1.5 transition hover:bg-overlay hover:text-accent"
              :class="queue.has(r.slug) ? 'text-canvas-foreground' : 'text-muted'"
              :aria-label="isGated ? t('auth.signInToQueue') : t('queue.playNext')"
              :title="isGated ? t('auth.signInToQueue') : t('queue.playNext')"
              @click="playNext(r.slug)"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" class="h-4 w-4" aria-hidden="true">
                <path d="M5 5l9 7-9 7V5z" /><rect x="16" y="5" width="2.4" height="14" rx="1" />
              </svg>
            </button>
          </li>
        </ul>
      </section>
    </div>
    </template>
  </aside>
</template>
