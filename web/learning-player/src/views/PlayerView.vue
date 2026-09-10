<script setup lang="ts">
/**
 * Player (PRD-039 / RFC-099 §2) — the hero surface. Plays origin audio directly (bridge,
 * never rehost) with a synced transcript: highlight + tap-to-seek + autoscroll, standard
 * transport controls, and cross-session resume (auth-gated, no-ops signed out).
 *
 * Balanced split (UXS-011): single column on mobile (masthead + intelligent artwork zone +
 * controls, then transcript); two columns on desktop (left rail + transcript). The artwork
 * zone doubles as a live intelligence surface (speaking-now + grounded signal); per-show
 * adaptive accent + insight-surfacing are wired progressively (Knowledge Panel = C5/#1084).
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { usePlayerStore } from '../stores/player'
import { useQueueStore } from '../stores/queue'
import { useAuthStore } from '../stores/auth'
import { useSignInGate } from '../composables/useSignInGate'
import { scrollBehavior } from '../utils/motion'
import { useCaptureStore } from '../stores/capture'
import { useCompletedStore } from '../stores/completed'
import { useUserPreferencesStore } from '../stores/userPreferences'
import CardRail from '../components/CardRail.vue'
import EpisodeTile from '../components/EpisodeTile.vue'
import KnowledgePanel from '../components/KnowledgePanel.vue'
import PlayerControls from '../components/PlayerControls.vue'
import CaptureMoment from '../components/CaptureMoment.vue'
import AddToCollectionButton from '../components/AddToCollectionButton.vue'
import OverflowMenu from '../components/OverflowMenu.vue'
import PlayerSkeleton from '../components/PlayerSkeleton.vue'
import { useResurfacingStore } from '../stores/resurfacing'
import TranscriptList from '../components/TranscriptList.vue'
import FavoriteButton from '../components/FavoriteButton.vue'
import DownloadButton from '../components/DownloadButton.vue'
import {
  activeInsightIndex,
  groundedSpansBySegment,
  groundedMomentCount,
  insightStartSeconds,
  INSIGHT_LINGER_MS,
  nextInsightIndex,
} from '../player/insights'
import { insightScrubberMarkers } from '../player/insightMarkers'
import { activeSegmentIndex, formatTime } from '../player/transcriptSync'
import type { ParagraphSpan } from '../player/transcriptCapture'
import {
  ApiError,
  getAudioSource,
  getEntities,
  getEpisode,
  getEpisodeStats,
  getInsights,
  getPlayback,
  getRelated,
  getSegments,
  markSurfaced,
} from '../services/api'
import { localPosition, shouldPush } from '../services/playbackPositions'
import {
  localArtworkFor,
  localKnowledgeFor,
  localSourceFor,
  localTranscriptFor,
} from '../services/downloads'
import { useDownloadsStore } from '../stores/downloads'
import type {
  EpisodeDetail,
  EpisodeStats,
  EpisodeSummary,
  Entity,
  FavoriteAdd,
  Insight,
  Segment,
  Topic,
} from '../services/types'
import Sparkline from '../components/Sparkline.vue'
import { formatDuration, formatPublishDate, speakerLabel } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import { getPlayerViewSnapshot, setPlayerViewSnapshot } from './player-view-cache'
import QueuePanel from '../components/QueuePanel.vue'

const props = defineProps<{ slug: string }>()
const { t, locale } = useI18n()
const router = useRouter()
const route = useRoute()

// Back = return to wherever you came from (Library, Home, Search, a show…), not a hardcoded
// catalog. Falls back to the catalog on a cold deep-link with no in-app history.
function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: 'catalog' })
}
const queue = useQueueStore()
const auth = useAuthStore()
const { isGated, gated } = useSignInGate()
const capture = useCaptureStore()
const completed = useCompletedStore()
/** Mark-as-played toggle (PL.6) — auth-gated like the other per-user actions. */
const toggleCompleted = gated(async () => {
  await completed.toggle(props.slug)
})
const userPrefs = useUserPreferencesStore()

// USERPREFS-1 (#1213) — audio-sync offsets across devices.
// Single nested key holds a map keyed by episode slug. Unbounded in
// principle but realistic bound is 10-50 tuned episodes per user; if
// that ever becomes a problem, migrate to a dedicated
// per-episode-preferences endpoint. localStorage remains the fast-path
// mirror so tuning applies instantly.
const AUDIO_SYNC_OFFSETS_PREF_KEY = 'lp.audioSyncOffsets'

type AudioSyncOffsets = Record<string, number>

function readRemoteOffset(slug: string): number | null {
  const map = userPrefs.get<AudioSyncOffsets>(AUDIO_SYNC_OFFSETS_PREF_KEY)
  if (!map || typeof map !== 'object') return null
  const v = map[slug]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function writeRemoteOffset(slug: string, value: number | null): void {
  const current = userPrefs.get<AudioSyncOffsets>(AUDIO_SYNC_OFFSETS_PREF_KEY) ?? {}
  const next: AudioSyncOffsets = { ...current }
  if (value === null || value === 0) delete next[slug]
  else next[slug] = value
  void userPrefs.set(AUDIO_SYNC_OFFSETS_PREF_KEY, next)
}


const episode = ref<EpisodeDetail | null>(null)
const segments = ref<Segment[]>([])
const audioUrl = ref<string | null>(null)
const insights = ref<Insight[]>([])
const topics = ref<Topic[]>([])
const persons = ref<Entity[]>([])
// #1261-4: "More like this" — semantic peer episodes for a natural continuation
// when this one ends. Silent no-op on error/empty; endpoint already existed.
const relatedEpisodes = ref<EpisodeSummary[]>([])
const panelOpen = ref(false)
const panelDialog = ref<HTMLDialogElement | null>(null)
const insightsOpener = ref<HTMLButtonElement | null>(null)

/**
 * Drive the dialog's MODE from the viewport (S9).
 *
 * `showModal()` gives the mobile sheet what it needs — top layer, focus trap, Escape, inert
 * background. At lg the same panel is a docked rail beside the player, where trapping focus would
 * be wrong: nothing is covered, and the user must be able to Tab back to the transcript.
 */
const DESKTOP_QUERY = '(min-width: 1024px)'
const isDesktop = ref(
  typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia(DESKTOP_QUERY).matches
    : false,
)

function syncPanelDialog(): void {
  const d = panelDialog.value
  if (!d) return
  if (!panelOpen.value) {
    if (d.open) d.close()
    return
  }
  // Re-opening in the other mode requires a close first: showModal() on an already-open dialog
  // throws InvalidStateError.
  if (d.open) d.close()
  if (isDesktop.value) d.show()
  else d.showModal()
}

watch([panelOpen, isDesktop], () => void nextTick(syncPanelDialog))

/**
 * Close came from anywhere — the ✕, Escape, or a backdrop tap. Restore focus to the control that
 * opened it: the opener is `v-if="!panelOpen"`, so it does not exist at the moment the browser
 * would restore focus itself, and focus would otherwise land on <body> and lose the user's place.
 */
function onPanelClose(): void {
  panelOpen.value = false
  void nextTick(() => insightsOpener.value?.focus())
}

/**
 * A modal dialog fills the screen, so a tap on the ::backdrop lands on the dialog element itself —
 * the inner container stops anything inside it from reaching here.
 */
function onPanelBackdropClick(e: MouseEvent): void {
  if (e.target === panelDialog.value) panelOpen.value = false
}
const focusInsightId = ref<string | null>(null)
const queueOpen = ref(false) // issue 1838 — queue & recently-played, from the player
const loading = ref(true)
const notFound = ref(false)
/** The episode exists (or we cannot tell) but loading it failed — offer a retry, not a denial. */
const loadFailed = ref(false)
/**
 * The episode could not be fetched AND is not on this device.
 *
 * A different fact from "the request failed", and the only one the user can act on: an episode
 * they downloaded plays here regardless, so being told "couldn't load this episode" for one they
 * simply never downloaded describes our request instead of their situation.
 */
const notDownloaded = ref(false)
/** The transcript artifact is unreadable, as opposed to not written yet. */
const transcriptBroken = ref(false)

/**
/**
 * The episode summary: `summary_text`, the full prose. That is the whole definition.
 *
 * NO FALLBACK to `summary_title`, and no bullets. A summary control that opens something other
 * than the summary is the complaint this panel has now generated three times: it showed a thematic
 * headline plus a bullet list, with the prose pushed below the fold, so tapping "Summary" produced
 * a screen that was not one. `summary_title` is a headline and `summary_bullets` are a structured
 * digest — both are different things, and substituting either is what made this hard to pin down.
 *
 * The control is gated on this being non-empty, so an episode with no prose summary offers no
 * summary rather than offering a stand-in.
 */
const summaryText = computed(() => episode.value?.summary_text ?? '')
const summaryOpen = ref(false)
const summaryDialog = ref<HTMLDialogElement | null>(null)

// Modal on every viewport: a summary is short-form reading the reader opted into, so trapping focus
// and dimming the page is right here — unlike the Knowledge Panel, which is a docked rail on
// desktop precisely because it is meant to sit alongside the transcript.
watch(summaryOpen, (open) => {
  void nextTick(() => {
    const d = summaryDialog.value
    if (!d) return
    if (open && !d.open) d.showModal()
    else if (!open && d.open) d.close()
  })
})

/** A tap on the backdrop lands on the dialog element itself; the inner container stops the rest. */
function onSummaryBackdropClick(e: MouseEvent): void {
  if (e.target === summaryDialog.value) summaryOpen.value = false
}

// Per-episode reach (UXS-014): anonymous cross-user counts + a daily-opens sparkline.
const stats = ref<EpisodeStats | null>(null)
const statsSeries = computed(() => stats.value?.daily.map((d) => d.count) ?? [])
/**
 * Does the reach chip have anything to SAY? (#1957)
 *
 * Its three children are each gated on data, but the wrapper used to render on `!panelOpen`
 * alone — so when all three were falsy it still painted a rounded background, padding and a
 * backdrop blur with nothing inside. A blind design critic reviewing the player called that out
 * as "a truncated grey pill … reads as a rendering fault".
 *
 * That is not an edge case: `/episodes/{slug}/stats` withholds `listeners` and `opens` below the
 * k-anonymity floor (#1923) and returns `daily: []` alongside them, so all three go false
 * together BY DESIGN. Measured against production, 12 of 12 sampled episodes were withheld — the
 * empty pill was on every episode page, and would have grown rarer only as the audience grew,
 * i.e. the app looked most broken when it was smallest.
 *
 * Withheld reach means "not enough people yet". The honest rendering of that is nothing.
 */
const hasReach = computed(
  () =>
    !!stats.value?.listeners ||
    !!stats.value?.opens ||
    (!!stats.value && statsSeries.value.some((n) => n > 0)),
)
const compact = (n: number): string =>
  n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)

// Transcript ↔ insight bridge: which segments back a grounded insight (highlight + tap-through).
const groundedSpans = computed(() => groundedSpansBySegment(segments.value, insights.value))
// #1140: insight-density ticks for the scrubber ("skip guide" — where the substance is).
const insightMarkers = computed(() => insightScrubberMarkers(insights.value, duration.value))

function openInsight(insightId: string): void {
  panelOpen.value = true
  // Reset then set so re-tapping the same grounded segment re-triggers the centre-scroll.
  focusInsightId.value = null
  void nextTick(() => {
    focusInsightId.value = insightId
  })
}

// Playback state + transport live in the player store (single source of truth for the UI,
// MediaSession, and native controls — #1307). What is left here is genuinely view-shaped:
// deep-link/resume seeking and the transcript sync-offset. Queue-advance and position PERSISTENCE
// both moved out, for the same reason: they must keep working when no player view is mounted, and
// persistence additionally has to pair the slug and the time from one object (see the store).
const player = usePlayerStore()
const downloads = useDownloadsStore()
const { playing, currentTime, duration, rate, audioError } = storeToRefs(player)
// No local <audio>: the store owns a detached element that outlives this view (#1587). Seeking
// and resume still happen here — they are episode/route concerns — but through the store's element.

// Manual sync-nudge UI is HIDDEN for now (not deleted): a better synchronization fix is coming,
// so the listener-facing nudge control is temporarily off. The offset machinery below still
// applies any previously-stored / server-mirrored offset to the highlight. Flip to re-expose.
const SHOW_SYNC_CONTROL = false

// Manual transcript-sync offset (seconds): the bridged stream (acast) injects ads not in our
// transcribed copy, so the transcript can lead the played audio. This lets the listener nudge
// the highlight to match what they hear. Maps content-time ↔ audio-time; persisted per episode.
const syncOffset = ref(0)
function syncKey(slug: string): string {
  return `lp:sync:${slug}`
}
function adjustSync(delta: number): void {
  syncOffset.value = Math.round((syncOffset.value + delta) * 10) / 10
  try {
    localStorage.setItem(syncKey(props.slug), String(syncOffset.value))
  } catch {
    /* storage unavailable — offset still applies for this session */
  }
  // USERPREFS-1 (#1213) — mirror to server so the tuning follows the
  // user across devices. Silent-degrade when unavailable.
  writeRemoteOffset(props.slug, syncOffset.value)
}
function resetSync(): void {
  syncOffset.value = 0
  try {
    localStorage.removeItem(syncKey(props.slug))
  } catch {
    /* ignore */
  }
  writeRemoteOffset(props.slug, null)
}

let resumeSeconds = 0
/**
 * Whether `resumeSeconds` is FINAL for this load.
 *
 * The start-position watcher fires the moment the element reports a duration. That was safe only
 * because `player.load()` and the line computing `resumeSeconds` sat in the same synchronous block
 * with no await between them — the element cannot report anything until the block yields. Nothing
 * named that invariant, and the disk-first path below puts an `await` inside the window by loading
 * a LOCAL file whose metadata lands almost immediately.
 *
 * NO TEST FAILS WHEN THIS GATE IS REMOVED, and that is worth saying rather than hiding. In every
 * current path `resumeSeconds` is assigned in the statement before `startReady`, so the watcher
 * cannot observe a stale one and the hazard is unreachable today. Three other guards of mine turned
 * out unreachable this session and were deleted; this one is kept, for a reason that does not apply
 * to those: it does not defend against a hazard that cannot occur, it REPLACES an invariant that
 * held only by the absence of an await — in the exact function that just gained one. Deleting it
 * would put the correctness back in the ordering of two lines with nothing naming the dependency.
 */
const startReady = ref(false)
/**
 * Whether the network critical path has answered.
 *
 * `loading` used to mean exactly that, and the snapshot watcher below read it as such. The
 * disk-first path clears `loading` as soon as something is PAINTED, which is the point — but it
 * would then snapshot the registry-derived detail (no publish date, no summary, every `has_*`
 * false), so a later reopen would paint a degraded page from cache until revalidation healed it.
 * The two meanings needed separating.
 */
const criticalDone = ref(false)
/** What the fast path actually seeked to, so a later correction can tell "untouched" from "scrubbed". */
let appliedSeconds = 0

/**
 * The `?t=<seconds>` a link asked to start at, or null.
 *
 * Validated rather than trusted: a query string is attacker-supplied, and `el.currentTime = NaN`
 * throws while a negative or absurd value would seek nowhere useful. Anything unusable is simply
 * ignored, so a malformed link still opens the episode.
 */
function deepLinkSeconds(): number | null {
  const raw = route.query.t
  const value = Number(Array.isArray(raw) ? raw[0] : raw)
  if (!Number.isFinite(value) || value < 0) return null
  return value
}

// Audio-time → content-time: subtract the sync offset so the highlight tracks what's heard.
const contentTime = computed(() => currentTime.value - syncOffset.value)
const activeIndex = computed(() => activeSegmentIndex(segments.value, contentTime.value))
const artwork = computed(() => (episode.value ? episodeArtwork(episode.value) : undefined))

const favItem = computed<FavoriteAdd>(() => ({
  kind: 'episode',
  ref: props.slug,
  label: episode.value?.title ?? '',
  sublabel: episode.value?.podcast_title ?? undefined,
  slug: props.slug,
}))

/**
 * Who said the insight — the quote's own speaker first, the transcript segment only as a fallback.
 *
 * `speakingNow` is the voice at the PLAYHEAD, and during the 4s linger the playhead has already
 * moved past the quote, so whenever the next segment is a different voice the insight gets
 * attributed to the wrong person. On real diarized data an unnamed segment also degrades to a
 * generic label ("Host"), while the quote itself now carries a properly resolved name — the server
 * gained that in a93a0ea1 and nothing here was updated to use it.
 *
 * The quote is the thing being attributed, so the quote's speaker is the correct source.
 */
const insightSpeaker = computed(
  () => activeInsight.value?.quotes?.find((q) => q.speaker)?.speaker || speakingNow.value || '',
)

const activeInsight = computed(() => {
  const i = activeInsightIndex(insights.value, contentTime.value, INSIGHT_LINGER_MS)
  return i >= 0 ? insights.value[i] : null
})
const nextInsight = computed(() => {
  const i = nextInsightIndex(insights.value, contentTime.value)
  return i >= 0 ? insights.value[i] : null
})
const nextInsightStartSeconds = computed(() =>
  nextInsight.value ? insightStartSeconds(nextInsight.value) : null,
)
const nextInsightCountdown = computed(() =>
  nextInsightStartSeconds.value != null
    ? Math.max(0, nextInsightStartSeconds.value - contentTime.value)
    : null,
)
function seekToNextInsight(): void {
  if (nextInsightStartSeconds.value != null) seekContent(nextInsightStartSeconds.value)
}
// Grounding receipt for the live insight panel (Zone D): how many distinct MOMENTS in the audio the
// current insight is sourced to.
//
// This used to count entries in `groundedSpans`, on the reasoning that one signal feeding two
// surfaces cannot drift from what is underlined below. It cannot drift, but it was answering a
// different question: that map is keyed by transcript SEGMENT, so a single quote crossing three
// segments was reported to the reader as three moments. The underline is per-segment because
// highlighting is per-segment; the receipt is per-moment because that is what "sourced to" claims.
// They are allowed to differ, and forcing them to agree is what made the number dishonest.
const insightGroundingCount = computed(() =>
  activeInsight.value ? groundedMomentCount(segments.value, activeInsight.value) : 0,
)
const metaLine = computed(() => {
  const parts: string[] = []
  const d = formatPublishDate(episode.value?.publish_date ?? null, locale.value)
  const dur = formatDuration(episode.value?.duration_seconds ?? null)
  if (d) parts.push(d)
  if (dur) parts.push(dur)
  return parts.join(' · ')
})
const speakingNow = computed(() =>
  speakerLabel(activeIndex.value >= 0 ? (segments.value[activeIndex.value]?.speaker ?? null) : null),
)

/**
 * Did the SERVER answer, or did the request never land? (#1906)
 *
 * A secondary surface that 404s or 500s has told us something, and clearing it is honest. A
 * transport failure has told us nothing — clearing it replaces content the user is looking at
 * with an empty rail, which is exactly the "a failed refresh must not delete the old stuff" rule
 * this app has to hold offline.
 */
/**
 * An `EpisodeDetail` reconstructed from the download registry (#1905/#1906).
 *
 * Offline, `getEpisode` cannot answer — but a downloaded episode already carries everything this
 * view needs to render and play: we captured title, show and duration at download time precisely
 * so this path could exist. Without it the critical path below rejects and the user gets an error
 * screen for a file that is sitting on their disk.
 */
function offlineEpisodeDetail(slug: string): EpisodeDetail | null {
  const e = downloads.entry(slug)
  if (!e || e.state !== 'downloaded') return null
  return {
    slug,
    title: e.title ?? slug,
    feed_id: e.feedId ?? '',
    podcast_title: e.showTitle ?? null,
    publish_date: null,
    duration_seconds: e.durationSeconds ?? null,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: localArtworkFor(slug),
    summary_title: null,
    summary_bullets: [],
    summary_text: null,
    has_transcript: !!e.transcriptPath,
    has_summary: false,
    has_gi: false,
    has_kg: false,
    has_bridge: false,
  }
}

function serverAnswered(err: unknown): boolean {
  return err instanceof ApiError
}

async function load(slug: string): Promise<void> {
  const cached = getPlayerViewSnapshot(slug)
  // Per LOAD, not per view: one mounted PlayerView serves every episode you walk to from it.
  startReady.value = false
  criticalDone.value = false
  notFound.value = false
  loadFailed.value = false
  notDownloaded.value = false
  transcriptBroken.value = false
  // Only for a DIFFERENT episode. Returning to the one already playing (tapping the mini-player)
  // must not touch transport state: the store's load() no-ops for the same slug, so nothing would
  // restore what we wiped — the element keeps playing while the UI shows Play at 0:00, the first
  // tap of Play pauses, and stopBackgroundAudio() kills the Android keep-alive service mid-listen,
  // which is exactly what #1310 exists to prevent. Left over from when the view owned the element.
  if (player.currentSlug !== slug) player.resetForLoad()
  // Returning to the episode already playing must not seek it. That used to fall out of the
  // watcher's `immediate` run: on a mid-listen reopen the duration is already non-zero, so it fired
  // during setup — before `load()` — with `resumeSeconds` still 0, did nothing, and burnt
  // `startApplied`. An accident of ordering, and one the gate above would otherwise remove. Said
  // outright here instead. `applyStartPosition` still honours a `?t=` on such a reopen, as it does
  // today, because it reads the query itself.
  if (player.currentSlug === slug && player.el?.src) {
    resumeSeconds = 0
    startReady.value = true
  }
  if (cached) {
    // #16 — reopening an episode we already loaded (usually the one playing, via the mini-player)
    // paints instantly from the cached snapshot instead of blanking behind the loading spinner. The
    // streamed fetches below still run to revalidate in place (no wipe), so a snapshot captured
    // before a slow rail arrived heals on reopen. transcriptOpen is left as the user last set it.
    episode.value = cached.episode
    segments.value = cached.segments
    audioUrl.value = cached.audioUrl
    insights.value = cached.insights
    topics.value = cached.topics
    persons.value = cached.persons
    relatedEpisodes.value = cached.relatedEpisodes
    stats.value = cached.stats
    loading.value = false
  } else {
    loading.value = true
    transcriptOpen.value = false // new episode → transcript starts closed (opt-in per episode)
    segments.value = []
    audioUrl.value = null
    insights.value = []
    topics.value = []
    persons.value = []
    relatedEpisodes.value = []
    stats.value = null
    resumeSeconds = 0
  }
  // Telemetry is best-effort and must NEVER gate the render — fire the open (then the reach stat that
  // depends on the open being counted) WITHOUT awaiting, so a metrics round-trip can't hold up
  // playback. Order preserved via .finally so the stat still reflects this open.
  // logListen moved to the player store (#1924): the view never saw auto-advance or the
  // mini-player, so most real listening went unrecorded. Stats still hang off this same tick.
  void Promise.resolve()
    .catch(() => {})
    .finally(() => {
      getEpisodeStats(slug)
        .then((s) => {
          stats.value = s
        })
        .catch((err: unknown) => {
          if (serverAnswered(err)) stats.value = null
        })
    })
  try {
    syncOffset.value = Number(localStorage.getItem(syncKey(slug))) || 0
  } catch {
    syncOffset.value = 0
  }
  // USERPREFS-1 (#1213) — upgrade the local sync offset if the server
  // preferences (hydrated once at app init in main.ts) have one for this
  // slug. Reading is synchronous; server value wins.
  const remote = readRemoteOffset(slug)
  if (remote !== null) syncOffset.value = remote
  // A DOWNLOADED episode is already on this device — title, show, duration, artwork and the audio
  // itself. Blocking the render on three round-trips for data sitting on the disk is the "loading,
  // loading" the operator reported, and offline it was never reachable at all: `offlineEpisodeDetail`
  // was consulted only when a fetch FAILED, never when it was merely slow.
  //
  // Placed after the sync-offset read on purpose: `applyStartPosition` adds `syncOffset` to a `?t=`
  // target, and firing with it still 0 would seek a shared link to the wrong place. Both reads are
  // synchronous, so this costs nothing. Skipped when a snapshot already painted — a snapshot holds a
  // richer EpisodeDetail than the registry can rebuild, and overwriting it would be a step backwards.
  if (!cached) {
    const diskSrc = localSourceFor(slug)
    const diskDetail = diskSrc ? offlineEpisodeDetail(slug) : null
    if (diskSrc && diskDetail) {
      episode.value = diskDetail
      audioUrl.value = diskSrc
      loading.value = false
      // BEFORE `player.load()`, deliberately. Loading the element is what can make it report a
      // duration, and the duration is what fires the start-position watcher — so the position has
      // to be settled first or the watcher has nothing correct to apply. `startReady` guards that
      // ordering; doing it in this order means the guard has nothing to catch here. Pinned by
      // `__checks__/start-position-order.test.ts`, because the hazard is invisible in a diff.
      //
      // The device's own position. The server's may be newer, and the reconciliation below stays
      // the authority on that — this is only what to start from while the network is answering.
      resumeSeconds = localPosition(slug)?.seconds ?? 0
      appliedSeconds = resumeSeconds
      startReady.value = true
      // `slug`, not `props.slug`: this may resolve for an episode the user has already left.
      player.load({
        slug,
        url: diskSrc,
        title: diskDetail.title,
        artwork: episodeArtwork(diskDetail) ?? null,
      })
      // The registry can only rebuild title, show and duration. Everything the page is actually
      // FOR — the summary, the insights, the topics and people — was written beside the audio at
      // download time, so read it. Async, so the thin paint above is not delayed by a file read;
      // slug-guarded, because the user may have moved on by the time it lands.
      void localKnowledgeFor(slug).then((k) => {
        if (!k || props.slug !== slug) return
        if (k.detail) episode.value = k.detail
        if (!insights.value.length) insights.value = k.insights
        if (!topics.value.length) topics.value = k.topics
        if (!persons.value.length) persons.value = k.persons
      })
    }
  }
  // "More like this" is a secondary rail at the bottom of the page, and it is by far the slowest
  // call on this route: /related embeds the episode text and searches the vector index, measured at
  // ~12 s warm against the fixture corpus (~46 s cold, while MiniLM loads). Inside the Promise.all
  // below it gated EVERYTHING on that — `loading` stayed true, the episode body did not render, and
  // `audioUrl` was not set, so the player could not even begin buffering until the peer rail was
  // ready. A rail nobody has scrolled to yet was holding up playback.
  //
  // Fire it alongside instead, exactly like the reach stats above: the page renders from the six
  // calls it actually needs, and the rail fills in when it arrives (it is `v-if`'d on being
  // non-empty, so there is nothing to lay out until then). The slug guard drops a late response
  // for an episode the user has already navigated away from — `relatedEpisodes` was reset above,
  // and without the check a slow reply would repopulate the previous episode's peers.
  getRelated(slug, 6)
    .then((r) => {
      if (props.slug === slug) relatedEpisodes.value = r.items
    })
    .catch((err: unknown) => {
      if (props.slug === slug && serverAnswered(err)) relatedEpisodes.value = []
    })
  // Transcript / insights / entities are secondary surfaces (below the fold on open) — fire them in
  // parallel but do NOT gate the render on them, exactly like the related rail above. This is what
  // lets audio start after ONE round-trip (detail + audio) instead of waiting on all six, including
  // the ~76 KB transcript. Slug-guarded so a late reply for a since-navigated episode is dropped.
  getSegments(slug)
    .then((segs) => {
      if (props.slug === slug) segments.value = segs?.segments ?? []
    })
    .catch(async (err: unknown) => {
      if (props.slug !== slug) return
      // A downloaded episode carries its own transcript (#1905) — use it before deciding
      // anything is wrong, so offline the transcript is simply there.
      const cached = await localTranscriptFor(slug)
      if (cached && props.slug === slug) {
        segments.value = cached.segments ?? []
        return
      }
      // The request never landed: say nothing rather than replacing a painted transcript with a
      // "broken" banner the network invented.
      if (!serverAnswered(err)) return
      // A 404 means "no transcript yet"; anything else means the artifact is BROKEN (the route 500s
      // on an unreadable segments file) — surface that rather than a perpetual "pending".
      transcriptBroken.value = !(err instanceof ApiError && err.status === 404)
      segments.value = []
    })
  getInsights(slug)
    .then((ins) => {
      if (props.slug === slug) insights.value = ins?.insights ?? []
    })
    .catch(async (err: unknown) => {
      if (props.slug !== slug) return
      // A downloaded episode carries its own insights — use them before concluding there are none.
      const local = await localKnowledgeFor(slug)
      if (local?.insights.length && props.slug === slug) {
        insights.value = local.insights
        return
      }
      if (serverAnswered(err)) insights.value = []
    })
  getEntities(slug)
    .then((ents) => {
      if (props.slug !== slug) return
      topics.value = ents?.topics ?? []
      persons.value = ents?.persons ?? []
    })
    .catch(async (err: unknown) => {
      if (props.slug !== slug) return
      const local = await localKnowledgeFor(slug)
      if (local && (local.topics.length || local.persons.length) && props.slug === slug) {
        topics.value = local.topics
        persons.value = local.persons
        return
      }
      if (!serverAnswered(err)) return
      topics.value = []
      persons.value = []
    })
  try {
    // CRITICAL PATH — only what's needed to render the player and START playback: the episode, its
    // audio source, and the saved position. Everything else streams in above.
    const [fetched, audio, playback] = await Promise.all([
      // Unlike its siblings this used to have no catch, so ANY transport failure aborted the whole
      // critical path — including for an episode already on disk (#1906).
      getEpisode(slug).catch((err: unknown) => {
        if (serverAnswered(err)) throw err
        return null
      }),
      getAudioSource(slug).catch(() => null),
      getPlayback(slug).catch(() => null),
    ])
    // The stored detail is the SERVER's, captured at download time — summary text, bullets and the
    // `has_*` flags the page gates its sections on. The registry stub below has none of that and
    // hardcodes them false, which is why a downloaded episode reported "no summary" for a summary
    // that was already on the device.
    const detail = fetched ?? (await localKnowledgeFor(slug))?.detail ?? offlineEpisodeDetail(slug)
    // A transport failure with nothing on disk is still a failure.
    if (!detail) throw new Error('episode unavailable offline')
    episode.value = detail
    const localSrc = localSourceFor(slug)
    audioUrl.value = audio?.url ?? localSrc
    // The local file wins inside player.load() via the injected resolver; passing it here too is
    // what lets playback start at all when the network gave us no origin URL.
    if (audio?.url || localSrc) {
      player.load({
        slug: props.slug,
        url: audio?.url ?? localSrc ?? '',
        title: episode.value?.title ?? null,
        artwork: artwork.value ?? null,
      })
    }
    // Offline, GET /playback fails and `playback` is null — fall back to the position this device
    // recorded, or a downloaded episode always restarts from the beginning (#1906).
    //
    // When BOTH exist, the same rule that decides whether to push decides what to resume from.
    // This used to prefer any pending local position outright, on the reasoning that it was
    // written after our last successful push — which says nothing about a write made on ANOTHER
    // device in between, and that is exactly when the two disagree (#1925 review). A phone that
    // listened offline while the laptop moved ahead would resume from the older of the two.
    const local = localPosition(slug)
    const server = playback
      ? {
          seconds: playback.position_seconds,
          finished: !!playback.finished,
          updatedAt: playback.updated_at,
        }
      : null
    resumeSeconds = (local && shouldPush(local, server) ? local.seconds : server?.seconds) ?? local?.seconds ?? 0
    // ...unless the LINK named a moment (#1914). A recap's saved line, a shared quote, an MCP
    // citation: they point at a place in the episode, and honouring the resume position instead
    // would silently drop the only reason the link existed. An explicit ask beats a remembered
    // one — and only for THIS load, so the next visit resumes normally.
    const asked = deepLinkSeconds()
    if (asked !== null) resumeSeconds = asked
    // If the fast path already started this episode from the device's position and the server's
    // turns out to be the newer one, correct it — but only while the correction is still invisible:
    // nothing playing, nothing scrubbed, no `?t=` to stomp. The trade this accepts is narrow and
    // deliberate: a downloaded episode, listened to more recently on ANOTHER device, whose owner
    // presses play inside the network window, resumes from the older position. `shouldPush` then
    // declines to overwrite the newer server value, so nothing is lost — only the start point.
    // `player.seek`, not `applyStartPosition`, which would re-run the `?t=` branch and re-stamp rate.
    if (startApplied === slug && asked === null) {
      const el = player.el
      const untouched = !!el && Math.abs(el.currentTime - appliedSeconds) < 0.5
      if (
        player.currentSlug === slug &&
        !player.playing &&
        untouched &&
        Math.abs(resumeSeconds - appliedSeconds) > 1
      ) {
        player.seek(resumeSeconds)
        appliedSeconds = resumeSeconds
      }
    }
    startReady.value = true
    // Lock-screen / headphone / BT metadata for the current episode (#1308).
    player.setMetadata({
      title: detail.title,
      artist: detail.podcast_title ?? undefined,
      artworkUrl: episodeArtwork(detail) ?? undefined,
    })
  } catch (err: unknown) {
    // A revalidation failure on a cache-hit reopen (#16) must NOT tear down the already-painted page
    // — the user is looking at valid cached content; a dropped network round-trip is not a reason to
    // replace it with an error.
    if (cached) return
    // "Not found" has to MEAN not found. Any failure used to land here, so a dropped connection
    // told the user an episode that exists does not — and no reload prompt with it.
    if (err instanceof ApiError && err.status === 404) notFound.value = true
    else {
      loadFailed.value = true
      // The server never answered and there is nothing on disk: say which, because "download it
      // while you have signal" is a move and "couldn't load" is not.
      notDownloaded.value = !serverAnswered(err) && !downloads.isDownloaded(slug)
    }
  } finally {
    loading.value = false
    criticalDone.value = true
  }
}

function applyStartPosition(): void {
  const el = player.el
  if (!el) return
  // A ?t= deep-link (jump-to-moment from search) wins over the saved resume position.
  const deepLink = Number(route.query.t)
  if (Number.isFinite(deepLink) && deepLink > 0) {
    // ?t= is a content-time (from a search jump-to-moment) → map to audio-time.
    el.currentTime = deepLink + syncOffset.value
  } else if (resumeSeconds > 1 && resumeSeconds < duration.value - 1) {
    el.currentTime = resumeSeconds
  }
  el.playbackRate = rate.value
}

// Duration lands asynchronously after `load()`. Apply the deep-link / resume position exactly once
// per episode, the first time a real duration appears — the view no longer receives the element's
// loadedmetadata event, since the element is not in this template any more.
let startApplied: string | null = null
watch(
  // `startReady` is a dependency, not just a condition: the duration and the settled position arrive
  // in either order, and the watcher must run when the SECOND of them does — still exactly once per
  // slug, which `startApplied` continues to guarantee.
  () => [player.currentSlug, duration.value, startReady.value] as const,
  ([slug, d, ready]) => {
    if (!ready || !slug || !d || startApplied === slug) return
    // The loaded episode must be THIS view's episode. Deep-linking to Y with ?t= while X is still
    // playing fires this immediately against X — seeking the wrong episode to Y's timestamp, an
    // audible jump in something the user is still listening to. The element outlives the view now,
    // so "whatever is loaded" and "what this page is about" are no longer the same thing.
    if (slug !== props.slug) return
    startApplied = slug
    applyStartPosition()
  },
  { immediate: true },
)

// Transcript is OPTIONAL and closed by default (mobile): pressing play should NOT jump the
// listener into the transcript. A Show/Hide toggle reveals it; opening scrolls it into view.
// (Desktop keeps the transcript visible as the side column — see the template's lg: classes.)
const transcriptEl = ref<HTMLElement | null>(null)
const transcriptOpen = ref(false)

function toggleTranscript(): void {
  transcriptOpen.value = !transcriptOpen.value
  if (transcriptOpen.value) {
    void nextTick(() =>
      transcriptEl.value?.scrollIntoView({ behavior: scrollBehavior(), block: 'start' }),
    )
  }
}

// Jump to a transcript/insight position (content-time) → audio-time via the sync offset.
// (toggle / seek / skip / cycleRate now live in the player store.)
function seekContent(contentSeconds: number): void {
  player.seek(contentSeconds + syncOffset.value)
}

// --- capture (P2, PRD-040): mark a moment, save a transcript paragraph/phrase ---
// A paragraph's save control reads as "saved" when any of its segments is covered by a saved span.
const savedSegmentIds = computed(() => capture.savedSegmentIds)
/**
 * The outcome of the last capture, shown by BOTH placements of the control (#1592).
 *
 * Was a boolean `momentFlash`, which could only express "it worked". Failure therefore had no
 * visual state at all: the handler announced into the `sr-only` region and returned, so a sighted
 * user could not distinguish a failed save from a missed tap. Three states, because there are
 * three outcomes.
 */
const captureState = ref<'idle' | 'saved' | 'failed'>('idle')
// Screen-reader confirmation for captures (the visual state alone isn't announced). Polite so it
// never interrupts the now-playing live region.
const captureAnnounce = ref('')
let flashTimer: ReturnType<typeof setTimeout> | undefined

/**
 * How long the capture receipt stays followable (ms).
 *
 * Deliberately the same 4s as Zone D's insight linger rather than the old 1.5s flash: this is now
 * a LINK to where the capture went, and 1.5s is not long enough to notice a new control, read it
 * and reach for it. Both numbers answer the same question — how long does a thing stay on screen
 * after the moment that produced it — so they should not disagree.
 */
const CAPTURE_RECEIPT_MS = 4000

/**
 * Is the transport currently PINNED to the top of the viewport? (#2004 item 6)
 *
 * The transport is `sticky top-0`, so it needs a notch inset — but only once it is actually stuck.
 * The inset used to be unconditional CSS, which meant that at the top of the page, where the control
 * sits directly under the artwork and is NOT pinned, it still carried
 * `mt-4` + `env(safe-area-inset-top)` ≈ 75px of padding that earns nothing there. That was the dead
 * band between the artwork and the player.
 *
 * The intent was conditional and the implementation was not, so shrinking the margin until the
 * screenshot looked right would have broken the pinned case the padding exists for.
 *
 * A zero-height sentinel sits immediately above the sticky element: while the sentinel is visible the
 * transport cannot be pinned; once it scrolls out of view, it is. Guarded on
 * `typeof IntersectionObserver` like `TrendingTopics.vue`, so environments without it (jsdom) simply
 * keep the unpinned padding rather than throwing.
 */
const transportStuck = ref(false)
const stickySentinelEl = ref<HTMLElement | null>(null)
let stuckObserver: IntersectionObserver | null = null

onMounted(() => {
  if (typeof IntersectionObserver === 'undefined') return
  stuckObserver = new IntersectionObserver(
    (entries) => {
      const e = entries[0]
      if (e) transportStuck.value = !e.isIntersecting
    },
    { threshold: 0 },
  )
  watch(
    stickySentinelEl,
    (el, _prev, onCleanup) => {
      stuckObserver?.disconnect()
      if (el) stuckObserver?.observe(el)
      onCleanup(() => stuckObserver?.disconnect())
    },
    { immediate: true },
  )
})
onBeforeUnmount(() => {
  stuckObserver?.disconnect()
  stuckObserver = null
})

function announceCapture(message: string): void {
  // Re-set so an identical consecutive message still re-announces.
  captureAnnounce.value = ''
  void nextTick(() => {
    captureAnnounce.value = message
  })
}

/** One-tap "mark this moment" at the current content-time, tagged with who's speaking. */
/** Auth-gated: a signed-out tap routes to sign-in rather than POSTing a 401 (#1590). */
const markMoment = () =>
  gated(async () => {
    const speaker =
      activeIndex.value >= 0 ? (segments.value[activeIndex.value]?.speaker ?? null) : null
    // Announce what ACTUALLY happened. The store swallows write failures, so this used to tell a
    // screen-reader user "Marked" — and flash the confirmation — when the POST had failed and
    // nothing was stored (S8). A confirmation of something that did not happen is worse than
    // silence: it stops the user retrying.
    const ok = await capture.captureMoment(props.slug, Math.max(0, contentTime.value), speaker)
    if (flashTimer) clearTimeout(flashTimer)
    captureState.value = ok ? 'saved' : 'failed'
    announceCapture(ok ? t('capture.marked') : t('capture.saveFailed'))
    // A capture is the only in-app action that adds to the resurfacing ladder, so it is the only
    // one that can move the Library badge without a navigation (#1592).
    if (ok) void useResurfacingStore().load()
    // Both outcomes clear on the same timer. A failure that stuck would be a permanently red
    // control; a failure that vanished faster than the success would be the old bug in miniature.
    flashTimer = setTimeout(() => {
      captureState.value = 'idle'
    }, CAPTURE_RECEIPT_MS)
  })()

/**
 * Capture is auth-gated, so a signed-out tap routes to sign-in (#1590).
 *
 * #1592 made this control permanently visible on touch — before that it was hover-only, so a
 * signed-out visitor could barely reach it. Now they can, and an ungated tap would POST a 401 and
 * announce nothing: the highlight appears to vanish.
 */
const onCaptureParagraph = (span: ParagraphSpan) =>
  gated(async () => {
    const ok = await capture.captureSpan(props.slug, span)
    announceCapture(ok ? t('capture.savedHighlight') : t('capture.saveFailed'))
  })()

function ensureCaptureLoaded(): void {
  if (!auth.isAuthenticated) return
  // `void capture.ensureLoaded()` floated the promise: a failing GET /highlights (503, offline,
  // expired session) became an UNHANDLED rejection in the browser, not only in tests. Nothing
  // depends on this resolving — the capture controls render from an empty store and the page is
  // fully usable — so a failure is caught and left un-loaded, which lets the next call retry.
  void capture.ensureLoaded().catch(() => {})
  void completed.ensureLoaded().catch(() => {})
}

/**
 * Arriving with `?revisit=<highlight_id>` IS the review — advance that highlight's spaced ladder
 * (#35).
 *
 * The inbox's "▶ jump", the Your Week card and the digest email all carry the marker, so all three
 * advance through this one path. Previously the ONLY advance path was the inbox's dismiss button,
 * so a user who actually went back and listened never progressed, and the digest re-sent the same
 * five items indefinitely — "spaced" repetition that never spaced.
 *
 * Fire-and-forget by design: this is bookkeeping, not something the page waits on or reports.
 * Signed-out arrivals are skipped rather than 401-ing, and a rejected call is swallowed — a
 * revisit that fails to record shows up again later, which is the safe direction to fail.
 */
const markedRevisits = new Set<string>()

function markRevisitFromQuery(): void {
  const id = route.query.revisit
  if (typeof id !== 'string' || !id) return
  // Auth hydrates asynchronously, so at mount a genuinely signed-in user can still read as signed
  // out. Bailing permanently here would drop their revisit; the watch below re-runs this the
  // moment auth resolves. The Set makes that idempotent — one repetition consumed per arrival,
  // however many times the trigger fires.
  if (!auth.isAuthenticated || markedRevisits.has(id)) return
  markedRevisits.add(id)
  void markSurfaced(id).catch(() => {
    /* the item stays due and resurfaces later — never block the player on bookkeeping */
  })
}

onMounted(() => {
  load(props.slug)
  ensureCaptureLoaded()
  markRevisitFromQuery()
  // MediaSession prev/next → the queue (#1308). Handlers read props.slug at call time.
  player.setSkipHandlers({
    next: () => {
      const n = queue.nextAfter(props.slug)
      if (n) void router.push({ name: 'player', params: { slug: n } })
    },
    prev: () => {
      const p = queue.prevBefore(props.slug)
      if (p) void router.push({ name: 'player', params: { slug: p } })
    },
  })
})
watch(() => props.slug, (s) => load(s))
// Snapshot the loaded surface per slug so reopening this episode paints instantly (#16). Records
// only once the critical path has painted (loading === false) and there is an episode to show; the
// streamed rails each reassign their ref as they arrive, keeping the snapshot current.
watch(
  [episode, segments, insights, topics, persons, stats, relatedEpisodes, criticalDone],
  () => {
    if (!criticalDone.value || !episode.value) return
    setPlayerViewSnapshot(props.slug, {
      episode: episode.value,
      segments: segments.value,
      audioUrl: audioUrl.value,
      insights: insights.value,
      topics: topics.value,
      persons: persons.value,
      relatedEpisodes: relatedEpisodes.value,
      stats: stats.value,
    })
  },
)
watch(() => auth.isAuthenticated, () => {
  ensureCaptureLoaded()
  markRevisitFromQuery() // auth resolved after mount — the arrival still counts
})
// Navigating between revisit items without unmounting the player (Your Week → card → card) changes
// only the query, so the mount hook never re-runs. Each new id is its own arrival.
watch(() => route.query.revisit, markRevisitFromQuery)
/**
 * Track the breakpoint live: a phone rotating into landscape, or a desktop window dragged across
 * 1024px, must switch the panel between modal sheet and docked rail. Reading matchMedia once at
 * setup would strand it in whichever mode the page happened to load in.
 */
let desktopMql: MediaQueryList | null = null
const onDesktopChange = (e: MediaQueryListEvent): void => {
  isDesktop.value = e.matches
}
onMounted(() => {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
  desktopMql = window.matchMedia(DESKTOP_QUERY)
  isDesktop.value = desktopMql.matches
  desktopMql.addEventListener('change', onDesktopChange)
})

onBeforeUnmount(() => {
  // No persist() here any more: the store saves on pause, on episode switch, on pagehide and on a
  // 10s cadence, keyed on what is actually PLAYING. Saving from here paired this page's slug with
  // the store's time, which is exactly the pair that could disagree.
  if (flashTimer) clearTimeout(flashTimer)
  desktopMql?.removeEventListener('change', onDesktopChange)
  // A dialog left open while its view unmounts keeps the top layer and the inert background.
  if (panelDialog.value?.open) panelDialog.value.close()
})
</script>

<template>
  <section>
    <button type="button" class="lp-nav" @click="goBack">‹ {{ t('player.back') }}</button>
    <!-- Polite SR confirmation for captures (mark-moment / save line or phrase). -->
    <p aria-live="polite" class="sr-only">{{ captureAnnounce }}</p>
    <QueuePanel v-if="queueOpen" @close="queueOpen = false" />

    <!-- F1.3: reserve the player's shape on a cold uncached load (a cached episode paints instantly
         from its snapshot and never reaches here), so the surface fills in place with no jump. -->
    <PlayerSkeleton v-if="loading" />
    <p v-else-if="notFound" class="mt-4 text-danger">{{ t('player.notFound') }}</p>
    <p v-else-if="notDownloaded" class="mt-4 text-muted" data-testid="player-not-downloaded">
      {{ t('player.notDownloaded') }}
    </p>
    <p v-else-if="loadFailed" class="mt-4 text-danger">
      {{ t('player.loadFailed') }}
      <button type="button" class="ml-2 underline" data-testid="player-retry" @click="load(props.slug)">
        {{ t('player.retry') }}
      </button>
    </p>

    <div
      v-else-if="episode"
      class="mt-3 lg:grid lg:gap-8"
      :class="
        panelOpen
          ? 'lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1fr)_minmax(0,0.85fr)]'
          : 'lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]'
      "
    >
      <!-- Left rail: masthead + intelligent artwork zone + controls.
           On mobile the rail is `display:contents` so its children (incl. the sticky controls)
           flow directly into the page scroll — that makes the controls' containing block span
           the transcript, so `sticky top-0` keeps them pinned while the transcript scrolls under.
           On desktop (lg) it's a normal grid column again. -->
      <div class="contents lg:block">
        <div class="flex items-start justify-between gap-3">
          <RouterLink
            v-if="episode.podcast_title && episode.feed_id"
            :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
            class="lp-kicker min-w-0 no-underline"
          >
            {{ episode.podcast_title }}
          </RouterLink>
          <!-- Offline (or for an entry downloaded before feed_id was captured) there is no show
               page to link to — show the name unlinked rather than hiding it. -->
          <span v-else-if="episode.podcast_title" class="lp-kicker min-w-0">{{
            episode.podcast_title
          }}</span>
          <span v-else />
          <div class="flex shrink-0 items-center gap-2">
            <!-- Mark this moment (P2 capture). Auth-gated means deferred, not hidden (#1590):
                 this is the cheapest entry to the learning loop, so hiding it hid the loop.

                 DESKTOP ONLY. On mobile this control lives in the sticky transport instead (#1592)
                 — the masthead scrolls away, and the moment you want to mark is mid-listen with a
                 thumb already on the controls. Rendering it in both places at once would be two
                 controls for one action; `lg:` and the transport's `lg:hidden` are complements, so
                 exactly one is on screen at any width. -->
            <CaptureMoment
              class="hidden lg:inline-flex"
              :state="captureState"
              :gated="isGated"
              variant="icon"
              @capture="markMoment"
            />
            <FavoriteButton :item="favItem" class="text-xl" />

            <DownloadButton :slug="props.slug" />

            <!--
              Pin THIS episode into a collection (#2013 follow-up).

              The control existed on browse rows (`EpisodeCard`, `kind: 'episode'`) and on the show
              page (`PodcastView`, `kind: 'show'`) but not here — so you could pin an episode from a
              list, or pin a whole show, but not the episode you were actually listening to, which
              is the moment you are most likely to want it.
            -->
            <AddToCollectionButton :item="{ kind: 'episode', ref: props.slug }" />
            <!-- Secondary actions overflow (UXS-014). Mark-as-played lives here — it's a rare,
                 deliberate action, not a primary transport control (PL.6). -->
            <OverflowMenu :label="t('player.moreActions')">
              <template #default="{ close }">
                <button
                  type="button"
                  data-menuitem
                  role="menuitem"
                  data-testid="mark-played"
                  class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-canvas-foreground transition hover:bg-overlay"
                  @click="toggleCompleted(); close()"
                >
                  {{ completed.has(props.slug) ? t('player.markUnplayed') : t('player.markPlayed') }}
                </button>
              </template>
            </OverflowMenu>
          </div>
        </div>
        <h1 class="mt-1 font-display text-3xl font-extrabold leading-tight tracking-tight">
          {{ episode.title }}
        </h1>
        <div v-if="metaLine || episode.has_gi" class="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted">
          <span v-if="metaLine">{{ metaLine }}</span>
          <span
            v-if="episode.has_gi"
            class="inline-flex items-center gap-1 rounded-full bg-overlay px-2 py-0.5 text-xs font-bold text-grounded"
          >● {{ t('player.grounded') }}</span>
        </div>
        <!-- Hero artwork (UXS-014/UXS-011 §43): the Ask/Insights actions + the summary sit over
             the top of the image; a live-intelligence band ("Zone D") owns the bottom — see below
             for why that moved out of the top row entirely. -->
        <div
          class="group relative mt-3 aspect-square w-full overflow-hidden rounded-2xl border border-border bg-elevated"
        >
          <img
            v-if="artwork"
            :src="artwork"
            :alt="episode.podcast_title ?? episode.title"
            class="h-full w-full object-cover"
          />
          <div class="absolute inset-0">
            <!--
              Top toolbar: Ask/Insights actions + reach, pinned top-right. `relative z-10` because
              the live-intelligence band below is bottom-anchored and grows UPWARD with its text
              (#Zone-D rewrite) — for a long insight it can reach as far up as this row, and
              without a higher stacking order its scrim would paint over these controls and make
              them unclickable.

              The insight itself used to live HERE, squeezed into a `basis-full` line under these
              controls (236px of 348px went to Summary/Insights/reach, leaving the insight card
              **48px of text width** — three characters a line). It now owns the full lower band
              instead of competing with a toolbar for a corner of the picture.
            -->
            <div class="relative z-10 flex flex-wrap items-start justify-end gap-2 p-3">
              <!--
                ONE control cluster, pinned right, in one row: Summary, Insights, reach.

                These were scattered — Insights mid-top, reach top-right, Summary alone at the
                bottom of the artwork — so three unrelated-looking chrome elements sat on three
                edges of the picture. Together they read as one toolbar and leave the rest of the
                artwork alone.

                Compactness comes from the padding, the type scale and a much smaller sparkline,
                NOT from shortening the Insights label. That control reads "N insights" on purpose:
                #1595 replaced "💡 3", which was "the least legible control on the page, styled
                like a statistic, for the product's central feature". Returning it to a bare count
                to save 40px would undo that for the sake of tidiness.
              -->
              <div class="flex shrink-0 items-center gap-1.5">
              <!-- Per-episode reach (UXS-014): listeners · opens · insights, with a tiny opens-over-time
                   sparkline. The insights score opens the Knowledge panel. -->
              <!-- Insights: the reason to choose this over a normal podcast app, so it is a
                   LABELLED control, not a 💡 emoji tucked into the stats cluster next to
                   listener/open counts (#1595). It used to read "💡 3" — the least legible control
                   on the page, styled like a statistic, for the product's central feature. -->
              <!-- Summary — sits WITH the other controls now, not alone at the foot of the art. -->
              <button
                v-if="summaryText && !panelOpen"
                type="button"
                data-testid="player-open-summary"
                :title="t('player.summaryOpenHint')"
                :aria-label="t('player.summaryOpenHint')"
                class="inline-flex shrink-0 items-center gap-1 rounded-full bg-canvas/95 px-2.5 py-1 text-[11px] font-bold text-canvas-foreground backdrop-blur transition hover:bg-canvas"
                @click="summaryOpen = true"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-3 w-3" aria-hidden="true">
                  <path d="M4 6h16M4 12h16M4 18h10" stroke-linecap="round" />
                </svg>
                {{ t('player.summaryOpen') }}
              </button>
              <button
                v-if="!panelOpen && insights.length"
                ref="insightsOpener"
                type="button"
                data-testid="player-open-insights"
                class="shrink-0 rounded-full bg-accent px-2.5 py-1 text-[11px] font-bold text-accent-foreground shadow-lg transition hover:opacity-90"
                @click="panelOpen = true"
              >
                <!-- No count on the opener pill (operator PL.5): the number reads as noise here; the
                     Insights section header inside the panel still carries the count (UXS-014). -->
                ✦ {{ t('kp.title') }}
              </button>
              <!-- Reach: a quieter scrim than the actions beside it — it is context, not a control
                   you act on, so it should not compete with them for the eye. The sparkline drops
                   from 116px to 44px; at this size it reads as a shape, which is all it ever
                   communicated. -->
              <div
                v-if="!panelOpen && hasReach"
                data-testid="player-reach"
                class="flex shrink-0 items-center gap-1.5 rounded-full bg-canvas/95 px-2.5 py-1 backdrop-blur"
              >
                <div class="flex items-center gap-2 text-[11px] font-bold leading-none">
                  <span
                    v-if="stats?.listeners"
                    class="flex items-center gap-1 text-canvas-foreground"
                    :aria-label="t('stats.listeners', stats.listeners, { named: { count: stats.listeners } })"
                    :title="t('stats.listeners', stats.listeners, { named: { count: stats.listeners } })"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-3 w-3" aria-hidden="true"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
                    {{ compact(stats.listeners) }}
                  </span>
                  <span
                    v-if="stats?.opens"
                    class="flex items-center gap-1 text-canvas-foreground"
                    :aria-label="t('stats.opens', stats.opens, { named: { count: stats.opens } })"
                    :title="t('stats.opens', stats.opens, { named: { count: stats.opens } })"
                  >▶ {{ compact(stats.opens) }}</span>
                </div>
                <Sparkline
                  v-if="stats && statsSeries.some((n) => n > 0)"
                  :values="statsSeries"
                  :width="44"
                  :height="14"
                  class="block text-canvas-foreground"
                />
              </div>
              </div>
            </div>

            <!--
              Zone D — the live-intelligence band (UXS-011 §43: "speaking-now, a grounding badge,
              and the insight surfacing at this moment"). Bottom-anchored (`absolute … bottom-0`)
              and sized by its own content, so it grows UPWARD as the insight text wraps rather
              than claiming a fixed fraction of the square — measured across the real corpus,
              insight length runs median ~105 chars (~3 lines) to a max of 266 (~9 lines), and a
              box sized for the max would make the median look empty.

              Two states, switched on whether an insight's quote window (plus a short linger) has
              the playhead:
                live — attribution, the insight itself, its grounding receipt, and what's next;
                rest — nothing is being said right now, so the artwork returns and only a quiet
                       one-line NEXT stays, which is exactly when "what's coming" earns its keep.
              There is deliberately no third "nothing to show yet" state that falls back to the
              FIRST insight — that shipped once, put an insight on screen before a word had been
              spoken, and was rejected. `activeInsight` is -1/null until something is actually
              playing; this only ever renders `rest` before playback starts.
            -->
            <Transition name="zone-d-fade">
              <div
                v-if="activeInsight"
                key="live"
                data-testid="player-zone-d-live"
                class="absolute inset-x-0 bottom-0"
              >
                <!--
                  Scrim: two ~80px bands stacked directly above the panel (normal document flow,
                  not a percentage of the square), so the fade always starts at the panel's own
                  top edge — wherever that lands once the text above has wrapped. A flat
                  `bg-canvas/40`–`/80` tint over the WHOLE lower artwork was tried and failed
                  legibility over real cover art (recorded regression): text only reads reliably
                  once it sits in the near-opaque zone right at the panel, so a ramp — not a
                  uniform wash — is load-bearing here, not decorative.
                -->
                <div class="relative h-20">
                  <div class="zone-d-scrim absolute inset-0 backdrop-blur-md" />
                  <div class="zone-d-scrim-tint absolute inset-0" />
                </div>
                <!--
                  ZONE D IS DELIBERATELY NOT A LIVE REGION (#1978 follow-up).

                  This panel replaces its contents every time the audio reaches a new insight —
                  every few tens of seconds, unprompted. Announcing that would be actively hostile:
                  a screen-reader user is LISTENING to the episode, and the panel restates a claim
                  drawn from the words they are hearing right now. `aria-live` would talk over the
                  podcast to paraphrase the podcast, repeatedly, with no way to decline.

                  It also matches the policy the app already holds: PlayerView owns exactly ONE live
                  region (the capture announcer below), and KnowledgePanel emits through it rather
                  than declaring a second, because two live regions on one page compete.

                  The panel is still reachable — it is ordinary content in the document, announced
                  when navigated to, with the speaker carried in the sr-only span below. The
                  decision is "available on demand", not "hidden". `__checks__/live-regions.test.ts`
                  fails if a live-region attribute is added here, so reversing this is a deliberate act.
                -->
                <!--
                  NO BOX (#2004 follow-up). The panel used to be a flat `bg-canvas/95` fill, which
                  reads as a framed rectangle sitting on the artwork.

                  It cannot simply become transparent: a flat wash across the lower artwork was
                  tried and failed legibility over real cover art (recorded regression above), and
                  the text needs a near-opaque ground. So the ramp continues INTO the panel instead
                  of stopping at a hard edge — same 95% canvas at the bottom, blended upward, so the
                  legibility is unchanged and the rectangle is gone.
                -->
                <div class="zone-d-body px-4 pb-4 pt-1 backdrop-blur-md">
                  <!-- Attribution: ONE glyph for the whole panel. The sr-only span keeps the
                       "speaking now" context for screen readers even though it's folded visually
                       into this one line rather than a separate pill. -->
                  <p class="lp-kicker">
                    ✦ {{ t('player.insightLabel') }}<template v-if="insightSpeaker"> · {{ t('player.insightBy', { speaker: insightSpeaker }) }}</template>
                    <span v-if="insightSpeaker" class="sr-only">{{ t('player.speakingNow') }}: {{ insightSpeaker }}</span>
                  </p>
                  <!-- Hero content: the insight is what this whole surface exists to show, so it
                       reads at display size. `line-clamp-[12]` is a ceiling well above the real
                       9-line max (a future outlier guard), not a target. -->
                  <p class="mt-1.5 font-display text-base leading-snug text-canvas-foreground line-clamp-[12]">
                    {{ activeInsight.text }}
                  </p>
                  <p v-if="insightGroundingCount > 0" class="mt-2 text-xs font-semibold text-muted">
                    {{ t('player.sourcedMoments', { count: insightGroundingCount }, insightGroundingCount) }}
                  </p>
                  <button
                    v-if="nextInsight"
                    type="button"
                    class="mt-3 block w-full rounded-lg text-left transition hover:bg-overlay/40"
                    @click="seekToNextInsight"
                  >
                    <span class="lp-kicker block leading-none">{{ t('player.next') }} · {{ t('player.nextIn', { time: formatTime(nextInsightCountdown ?? 0) }) }}</span>
                    <span class="mt-1 block text-xs text-muted line-clamp-2">{{ nextInsight.text }}</span>
                  </button>
                </div>
              </div>
              <div
                v-else
                key="rest"
                data-testid="player-zone-d-rest"
                class="absolute inset-x-0 bottom-0 p-3"
              >
                <button
                  v-if="nextInsight"
                  type="button"
                  class="inline-flex items-center gap-1.5 rounded-full bg-canvas/95 px-3 py-1 backdrop-blur transition hover:bg-canvas/90"
                  @click="seekToNextInsight"
                >
                  <span class="lp-kicker leading-none">{{ t('player.next') }} · {{ t('player.nextIn', { time: formatTime(nextInsightCountdown ?? 0) }) }}</span>
                </button>
              </div>
            </Transition>
          </div>
        </div>


        <!-- Sticky sentinel: a zero-height marker whose visibility tells us whether the transport
             below is currently pinned. See `transportStuck`. -->
        <div ref="stickySentinelEl" aria-hidden="true" class="h-px lg:hidden" />
        <!-- Mobile: the controls float (sticky) at the top so they stay reachable while the
             transcript scrolls underneath. The wrapper carries an opaque page background + a
             little top padding (safe-area aware) so the transcript is masked as it scrolls under,
             the rounded panel keeps breathing room, and it clears a device status bar instead of
             being clipped at y=0. Desktop: static in the left column (wrapper is inert). -->
        <div
          ref="stickyEl"
          data-testid="player-controls-sticky"
          :data-stuck="transportStuck ? 'true' : 'false'"
          class="sticky top-0 z-20 mt-4 bg-canvas pb-2 lg:static lg:z-auto lg:mt-4 lg:bg-transparent lg:p-0"
          :class="transportStuck ? 'pt-[max(0.5rem,env(safe-area-inset-top))] lg:pt-0' : 'pt-2 lg:pt-0'"
        >
          <p v-if="audioError" class="rounded-2xl border border-border bg-surface p-4 text-danger">
            {{ t('player.audioError') }}
          </p>
          <PlayerControls
            v-else-if="audioUrl"
            :playing="playing"
            :current-time="currentTime"
            :duration="duration"
            :rate="rate"
            :markers="insightMarkers"
            @toggle="player.toggle"
            @seek="player.seek"
            @skip="player.skip"
            @cycle-rate="player.cycleRate"
          >
            <!-- Transcript toggle: a compact icon pill pinned to the LEFT of the controls row,
                 mirroring the speed pill on the right. Adds zero height (absolute in the existing
                 row). A CC-style transport affordance — accent when the transcript is open, plus a
                 tooltip. Mobile only (desktop shows the transcript as the side column). -->
            <template #corner>
              <div class="flex items-center gap-1.5">
              <button
                v-if="segments.length"
                type="button"
                class="flex h-11 w-11 items-center justify-center rounded-full border border-border transition"
                :class="transcriptOpen ? 'bg-accent text-accent-foreground' : 'text-muted hover:bg-overlay'"
                :aria-expanded="transcriptOpen"
                :aria-label="transcriptOpen ? t('player.hideTranscript') : t('player.showTranscript')"
                :title="transcriptOpen ? t('player.hideTranscript') : t('player.showTranscript')"
                data-testid="transcript-toggle"
                @click="toggleTranscript"
              >
                <!-- Transcript glyph: a captions/subtitles mark — a framed screen with text lines,
                     the universal "read the text" transport affordance. -->
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true">
                  <rect x="3" y="5" width="18" height="14" rx="2.5" />
                  <path d="M7 10.5h7M7 14h10" />
                </svg>
              </button>
              <!-- Capture, beside the transcript toggle. The grouping is the point: this corner is
                   CONTENT actions (read it, keep it) and the right corner is PLAYBACK actions
                   (speed, queue). Putting capture on the right would have been one free slot and no
                   rule. -->
              <CaptureMoment
                :state="captureState"
                :gated="isGated"
                variant="pill"
                @capture="markMoment"
              />
              </div>
            </template>
            <!-- Queue & recently-played — a transport affordance next to the speed pill, where it's
                 reachable while playing (was misplaced at the top of the page). -->
            <template #corner-right>
              <button
                type="button"
                class="inline-flex h-11 w-11 items-center justify-center rounded-full border border-border text-canvas-foreground transition"
                :aria-label="t('queue.open')"
                :title="t('queue.open')"
                data-testid="player-queue"
                @click="queueOpen = true"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true">
                  <path d="M3 6h13" /><path d="M3 12h13" /><path d="M3 18h9" /><path d="m17 15 4 3-4 3" />
                </svg>
              </button>
            </template>
          </PlayerControls>
          <p v-else class="rounded-2xl border border-border bg-surface p-4 text-muted">
            {{ t('player.audioUnavailable') }}
          </p>
        </div>
      </div>

      <!-- Middle: synced transcript. The mobile show/hide toggle lives in the floating controls
           panel (PlayerControls #corner slot) so it's reachable at any scroll position. -->
      <div ref="transcriptEl" class="mt-6 scroll-mt-20 lg:mt-0 lg:flex lg:max-h-[70dvh] lg:flex-col">
        <!-- Transcript body — opt-in on mobile (toggled), always shown on desktop. `lg:contents`
             dissolves this wrapper at lg so the transcript keeps flowing inside the flex column
             (lg:flex-1 scroll) exactly as before. -->
        <div :class="transcriptOpen ? 'block' : 'hidden'" class="lg:contents">
          <!-- Manual sync nudge — HIDDEN for now (SHOW_SYNC_CONTROL); a better sync fix is coming. -->
          <div
            v-if="segments.length && SHOW_SYNC_CONTROL"
            class="mb-2 flex items-center justify-end gap-2 text-xs text-muted"
          >
            <span :title="t('player.syncHelp')">{{ t('player.sync') }}</span>
            <div class="flex items-center gap-1">
              <button
                type="button"
                class="rounded-full border border-border px-2 py-0.5 font-mono leading-none"
                :aria-label="t('player.syncEarlier')"
                @click="adjustSync(-1)"
              >
                −
              </button>
              <button
                type="button"
                class="min-w-[3.5rem] rounded-full px-1 py-0.5 text-center font-mono tabular-nums"
                :class="syncOffset !== 0 ? 'text-canvas-foreground' : 'text-muted'"
                :aria-label="t('player.syncReset')"
                @click="resetSync"
              >
                {{ syncOffset > 0 ? '+' : '' }}{{ syncOffset }}s
              </button>
              <button
                type="button"
                class="rounded-full border border-border px-2 py-0.5 font-mono leading-none"
                :aria-label="t('player.syncLater')"
                @click="adjustSync(1)"
              >
                +
              </button>
            </div>
          </div>
          <TranscriptList
            v-if="segments.length"
            :segments="segments"
            :active-index="activeIndex"
            :grounded="groundedSpans"
            :can-capture="true"
            :gated="!auth.isAuthenticated"
            :saved-segment-ids="savedSegmentIds"
            class="min-h-0 lg:flex-1"
            @seek="seekContent"
            @insight="openInsight"
            @capture="onCaptureParagraph"
          />
          <p
            v-else
            data-testid="player-transcript-empty"
            class="rounded-2xl border border-border bg-surface p-4"
            :class="transcriptBroken ? 'text-danger' : 'text-muted'"
          >
            {{ transcriptBroken ? t('player.transcriptBroken') : t('player.transcriptPending') }}
          </p>
        </div>

        <!-- #1261-4: "More like this" — related episodes rail below the
             transcript. Silent when the endpoint returns empty. -->
        <section
          v-if="relatedEpisodes.length"
          class="mt-6"
          data-testid="related-episodes-rail"
          :aria-label="t('player.relatedEpisodes')"
        >
          <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
            {{ t('player.relatedEpisodes') }}
          </h2>
          <!--
            A TILE, not the horizontal card. `EpisodeCard` puts its text in a column beside the
            artwork, which in a rail slot left the title ~100px: one real title wrapped to eight
            lines, the slot grew to roughly 800px tall, and the action row floated over the artwork.
            Slots are narrower too — the old 224px made a 224px square of artwork dominate the rail.
          -->
          <CardRail>
            <li
              v-for="ep in relatedEpisodes"
              :key="ep.slug"
              class="w-44 shrink-0 sm:w-48"
            >
              <EpisodeTile :episode="ep" />
            </li>
          </CardRail>
        </section>
      </div>

      <!--
        Knowledge Panel: persistent rail on desktop, MODAL sheet on mobile.

        A native <dialog>, so the browser supplies focus trapping, Escape-to-close, an inert
        background and the correct accessibility tree. Hand-rolling those is ~60 lines of code whose
        bugs are silent and land on keyboard and screen-reader users; this shipped as a plain <div>
        that covered the screen without declaring itself a dialog, so Tab walked into the hidden page
        behind it and Escape did nothing.

        ONE element, two modes — `showModal()` on mobile, `show()` (non-modal, in normal flow) at lg.
        Rendering the panel twice would be simpler markup and would double every request it makes on
        mount, so the mode switch lives in script instead.

        The dialog element itself carries only the UA-style reset and positioning; the visual
        container stays an inner div, because `border-0` and `border-t` on one element resolve by
        stylesheet order rather than attribute order.

        The explicit height is not decoration: the UA stylesheet sets `height: fit-content` on
        <dialog>, which beats a top/bottom stretch, so the sheet sized itself to its content and ran
        off the bottom of the screen — entity-card controls ended up unclickable at "outside of the
        viewport". A plain <div> had no such rule, which is why this only appeared on conversion.
      -->
      <dialog
        ref="panelDialog"
        data-testid="knowledge-panel"
        :aria-label="t('kp.title')"
        class="m-0 h-[calc(100dvh-2rem)] max-h-none max-w-none border-0 bg-transparent p-0 text-canvas-foreground backdrop:bg-black/50 fixed inset-x-0 bottom-0 top-8 z-40 w-full lg:static lg:top-auto lg:z-auto lg:h-auto lg:w-auto lg:backdrop:bg-transparent"
        @close="onPanelClose"
        @click="onPanelBackdropClick"
      >
      <div
        class="h-full overflow-hidden rounded-t-2xl border-t border-border bg-canvas pb-[env(safe-area-inset-bottom)] lg:max-h-[70dvh] lg:rounded-2xl lg:border lg:pb-0"
      >
        <KnowledgePanel
          :episode="episode"
          :insights="insights"
          :topics="topics"
          :persons="persons"
          :slug="slug"
          :active-insight-id="activeInsight?.id ?? null"
          :focus-insight-id="focusInsightId"
          @seek="seekContent"
          @announce="announceCapture"
          @close="panelOpen = false"
        />
      </div>
      </dialog>

      <!--
        Episode summary — opened from the control on the artwork, never laid over it.

        A native <dialog> for the same reason the panel above is one: the browser supplies focus
        trapping, Escape, an inert background and the right accessibility tree. Capped at 80dvh with
        the prose scrolling INSIDE, so a long summary is reachable to its end — the specific failure
        of the old overlay, which clipped to the hero's fixed square and trailed off in an ellipsis.
      -->
      <dialog
        ref="summaryDialog"
        data-testid="episode-summary-dialog"
        :aria-label="t('player.summaryRegion')"
        class="m-auto max-h-[80dvh] w-[min(34rem,calc(100vw-2rem))] rounded-2xl border border-border bg-canvas p-0 text-canvas-foreground backdrop:bg-black/60"
        @close="summaryOpen = false"
        @click="onSummaryBackdropClick"
      >
        <!-- Content only while open. A <dialog> renders its children regardless, so without this
             the whole summary sits in the DOM (and in the accessibility tree) behind a closed
             dialog — the thing this change exists to stop. -->
        <!--
          No title bar. "Episode summary" over a rule, above the episode's own headline, was two
          headings and a border spent saying what the content already says — real estate on a
          phone, where the dialog is capped at 80dvh and every row costs prose.

          The close sits on the headline's line instead, and that row is `sticky` inside the
          scroller: dropping the bar must not put the only way out at the top of text the reader
          has scrolled past. `aria-label` on the <dialog> still names the region for assistive
          tech, so removing the visible <h2> costs nothing there.
        -->
        <div v-if="summaryOpen" class="max-h-[80dvh] overflow-y-auto px-5 pb-5">
          <div class="sticky top-0 flex items-start justify-between gap-3 bg-canvas pb-2 pt-4">
            <!--
              A visible "Summary" label (#2004 item 16) — it NAMES the panel, it is not content.
              Under it used to sit the thematic `summary_title` ("AI Psychosis: Agents, Claws, and
              the Skill-Issue Frontier"), which is neither the episode title nor the summary, and
              made the panel read as though the wrong thing had opened.
            -->
            <div class="min-w-0">
              <p class="lp-kicker" data-testid="summary-label">{{ t('player.summaryOpen') }}</p>
            </div>
            <button
              type="button"
              data-testid="episode-summary-close"
              :aria-label="t('player.summaryClose')"
              class="-mr-1 -mt-1 shrink-0 rounded-full px-2 py-1 text-sm text-muted transition hover:text-canvas-foreground"
              @click="summaryOpen = false"
            >
              ✕
            </button>
          </div>
          <!-- The full prose, and nothing beside it. It was a quote-styled block indented behind a
               rule because it sat BELOW the bullets; it is the panel's content now, so it reads as
               body text. -->
          <p
            class="mt-2 whitespace-pre-line text-sm leading-relaxed text-canvas-foreground"
            data-testid="episode-summary-text"
          >
            {{ summaryText }}
          </p>
        </div>
      </dialog>
    </div>
  </section>
</template>

<style scoped>
/*
 * Zone D scrim ramp (UXS-011 §43 rewrite): the blur and the tint fade together, over the SAME
 * ~80px band, from fully applied at the panel's own top edge to nothing above it. A flat
 * `bg-canvas/40`–`/80` wash across the whole lower artwork was tried and failed legibility over
 * real cover art (recorded regression) — text only reads reliably where it sits in the near-opaque
 * zone right at the panel, so the ramp does the legibility work, not a uniform tint.
 */
.zone-d-scrim {
  -webkit-mask-image: linear-gradient(to top, black, transparent);
  mask-image: linear-gradient(to top, black, transparent);
}
/* Continues the scrim through the panel body, so the two read as one gradient rather than a
   gradient stopping at the top edge of a filled box. */
.zone-d-body {
  background: linear-gradient(
    to top,
    color-mix(in srgb, var(--lp-canvas) 95%, transparent) 60%,
    color-mix(in srgb, var(--lp-canvas) 88%, transparent) 100%
  );
}
.zone-d-scrim-tint {
  background: linear-gradient(
    to top,
    color-mix(in srgb, var(--lp-canvas) 95%, transparent) 0%,
    transparent 100%
  );
}

/* Short fade/slide between the live and rest states — `--lp-motion` scales it like every other
   transition in the app, and `prefers-reduced-motion` (style.css) already pins it to ~0 without
   this fighting that rule. */
.zone-d-fade-enter-active,
.zone-d-fade-leave-active {
  transition:
    opacity calc(0.2s * var(--lp-motion)) ease,
    transform calc(0.2s * var(--lp-motion)) ease;
}
.zone-d-fade-enter-from,
.zone-d-fade-leave-to {
  opacity: 0;
  transform: translateY(6px);
}
</style>
