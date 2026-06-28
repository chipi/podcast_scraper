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
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useQueueStore } from '../stores/queue'
import { useAuthStore } from '../stores/auth'
import { useCaptureStore } from '../stores/capture'
import KnowledgePanel from '../components/KnowledgePanel.vue'
import PlayerControls from '../components/PlayerControls.vue'
import TranscriptList from '../components/TranscriptList.vue'
import FavoriteButton from '../components/FavoriteButton.vue'
import { activeInsightIndex, groundedSpansBySegment } from '../player/insights'
import { activeSegmentIndex, PLAYBACK_RATES } from '../player/transcriptSync'
import type { ParagraphSpan } from '../player/transcriptCapture'
import {
  getAudioSource,
  getEntities,
  getEpisode,
  getEpisodeStats,
  getInsights,
  getPlayback,
  getSegments,
  logListen,
  putPlayback,
} from '../services/api'
import type {
  EpisodeDetail,
  EpisodeStats,
  Entity,
  FavoriteAdd,
  Insight,
  Segment,
  Topic,
} from '../services/types'
import Sparkline from '../components/Sparkline.vue'
import { formatDuration, formatPublishDate, speakerLabel } from '../utils/format'
import { episodeArtwork } from '../utils/episode'

const props = defineProps<{ slug: string }>()
const { t, locale } = useI18n()
const router = useRouter()
const route = useRoute()
const queue = useQueueStore()
const auth = useAuthStore()
const capture = useCaptureStore()

async function onEnded(): Promise<void> {
  playing.value = false
  const next = queue.nextAfter(props.slug)
  if (next) await router.push({ name: 'player', params: { slug: next } })
}

const episode = ref<EpisodeDetail | null>(null)
const segments = ref<Segment[]>([])
const audioUrl = ref<string | null>(null)
const insights = ref<Insight[]>([])
const topics = ref<Topic[]>([])
const persons = ref<Entity[]>([])
const panelOpen = ref(false)
const focusInsightId = ref<string | null>(null)
const loading = ref(true)
const notFound = ref(false)

// Per-episode reach (UXS-014): anonymous cross-user counts + a daily-opens sparkline.
const stats = ref<EpisodeStats | null>(null)
const statsSeries = computed(() => stats.value?.daily.map((d) => d.count) ?? [])
const compact = (n: number): string =>
  n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)

// Transcript ↔ insight bridge: which segments back a grounded insight (highlight + tap-through).
const groundedSpans = computed(() => groundedSpansBySegment(segments.value, insights.value))

function openInsight(insightId: string): void {
  panelOpen.value = true
  // Reset then set so re-tapping the same grounded segment re-triggers the centre-scroll.
  focusInsightId.value = null
  void nextTick(() => {
    focusInsightId.value = insightId
  })
}

const audioEl = ref<HTMLAudioElement | null>(null)
const playing = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const rate = ref(1)
const audioError = ref(false)

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
}
function resetSync(): void {
  syncOffset.value = 0
  try {
    localStorage.removeItem(syncKey(props.slug))
  } catch {
    /* ignore */
  }
}

let resumeSeconds = 0
let lastSaved = 0

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

const activeInsight = computed(() => {
  const i = activeInsightIndex(insights.value, contentTime.value)
  return i >= 0 ? insights.value[i] : null
})
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

async function load(slug: string): Promise<void> {
  loading.value = true
  notFound.value = false
  audioError.value = false
  scrolledToTranscript = false // re-arm scroll-to-transcript for the new episode
  segments.value = []
  audioUrl.value = null
  insights.value = []
  topics.value = []
  persons.value = []
  stats.value = null
  resumeSeconds = 0
  // Record the open (best-effort) then fetch fresh reach — order so this open is counted.
  await logListen(slug)
  getEpisodeStats(slug)
    .then((s) => {
      stats.value = s
    })
    .catch(() => {
      stats.value = null
    })
  try {
    syncOffset.value = Number(localStorage.getItem(syncKey(slug))) || 0
  } catch {
    syncOffset.value = 0
  }
  try {
    const [detail, segs, audio, playback, ins, ents] = await Promise.all([
      getEpisode(slug),
      getSegments(slug).catch(() => null),
      getAudioSource(slug).catch(() => null),
      getPlayback(slug).catch(() => null),
      getInsights(slug).catch(() => null),
      getEntities(slug).catch(() => null),
    ])
    episode.value = detail
    segments.value = segs?.segments ?? []
    audioUrl.value = audio?.url ?? null
    insights.value = ins?.insights ?? []
    topics.value = ents?.topics ?? []
    persons.value = ents?.persons ?? []
    resumeSeconds = playback?.position_seconds ?? 0
  } catch {
    notFound.value = true
  } finally {
    loading.value = false
  }
}

function onLoadedMetadata(): void {
  const el = audioEl.value
  if (!el) return
  duration.value = el.duration || 0
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

function persist(): void {
  if (props.slug) void putPlayback(props.slug, currentTime.value)
}

function onTimeUpdate(): void {
  currentTime.value = audioEl.value?.currentTime ?? 0
  const now = Date.now()
  if (now - lastSaved > 10_000) {
    lastSaved = now
    persist()
  }
}

// Transcript section — on the first play of an episode, bring it into view (the masthead/artwork
// is tall; pressing play means "I'm listening", so surface the synced transcript).
const transcriptEl = ref<HTMLElement | null>(null)
let scrolledToTranscript = false

function toggle(): void {
  const el = audioEl.value
  if (!el) return
  if (el.paused) {
    void el.play()
    if (!scrolledToTranscript) {
      scrolledToTranscript = true
      void nextTick(() =>
        transcriptEl.value?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      )
    }
  } else {
    el.pause()
  }
}

function seek(to: number): void {
  const el = audioEl.value
  if (!el) return
  el.currentTime = Math.max(0, Math.min(to, duration.value || to))
}

// Jump to a transcript/insight position (content-time) → audio-time via the sync offset.
function seekContent(contentSeconds: number): void {
  seek(contentSeconds + syncOffset.value)
}

function skip(delta: number): void {
  seek(currentTime.value + delta)
}

function cycleRate(): void {
  const i = PLAYBACK_RATES.indexOf(rate.value as (typeof PLAYBACK_RATES)[number])
  rate.value = PLAYBACK_RATES[(i + 1) % PLAYBACK_RATES.length]
  if (audioEl.value) audioEl.value.playbackRate = rate.value
}

// --- capture (P2, PRD-040): mark a moment, save a transcript paragraph/phrase ---
// A paragraph's save control reads as "saved" when any of its segments is covered by a saved span.
const savedSegmentIds = computed(() => capture.savedSegmentIds)
const momentFlash = ref(false)
// Screen-reader confirmation for captures (the visual flash alone isn't announced). Polite so it
// never interrupts the now-playing live region.
const captureAnnounce = ref('')
let flashTimer: ReturnType<typeof setTimeout> | undefined

function announceCapture(message: string): void {
  // Re-set so an identical consecutive message still re-announces.
  captureAnnounce.value = ''
  void nextTick(() => {
    captureAnnounce.value = message
  })
}

/** One-tap "mark this moment" at the current content-time, tagged with who's speaking. */
async function markMoment(): Promise<void> {
  const speaker = activeIndex.value >= 0 ? (segments.value[activeIndex.value]?.speaker ?? null) : null
  await capture.captureMoment(props.slug, Math.max(0, contentTime.value), speaker)
  momentFlash.value = true
  announceCapture(t('capture.marked'))
  if (flashTimer) clearTimeout(flashTimer)
  flashTimer = setTimeout(() => {
    momentFlash.value = false
  }, 1500)
}

async function onCaptureParagraph(span: ParagraphSpan): Promise<void> {
  await capture.captureSpan(props.slug, span)
  announceCapture(t('capture.savedHighlight'))
}

function ensureCaptureLoaded(): void {
  if (auth.isAuthenticated) void capture.ensureLoaded()
}

onMounted(() => {
  load(props.slug)
  ensureCaptureLoaded()
})
watch(() => props.slug, (s) => load(s))
watch(() => auth.isAuthenticated, ensureCaptureLoaded)
onBeforeUnmount(() => {
  persist()
  if (flashTimer) clearTimeout(flashTimer)
})
</script>

<template>
  <section>
    <RouterLink :to="{ name: 'catalog' }" class="lp-nav">‹ {{ t('player.back') }}</RouterLink>
    <!-- Polite SR confirmation for captures (mark-moment / save line or phrase). -->
    <p aria-live="polite" class="sr-only">{{ captureAnnounce }}</p>

    <p v-if="loading" class="mt-4 text-muted">{{ t('player.loading') }}</p>
    <p v-else-if="notFound" class="mt-4 text-danger">{{ t('player.notFound') }}</p>

    <div
      v-else-if="episode"
      class="mt-3 lg:grid lg:gap-8"
      :class="
        panelOpen
          ? 'lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1fr)_minmax(0,0.85fr)]'
          : 'lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]'
      "
    >
      <!-- Left rail: masthead + intelligent artwork zone + controls -->
      <div>
        <div class="flex items-start justify-between gap-3">
          <RouterLink
            v-if="episode.podcast_title"
            :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
            class="lp-kicker min-w-0 no-underline"
          >
            {{ episode.podcast_title }}
          </RouterLink>
          <span v-else />
          <div class="flex shrink-0 items-center gap-2">
            <!-- Mark this moment (P2 capture; auth-gated). Brief "saved" flash on tap. -->
            <button
              v-if="auth.isAuthenticated"
              type="button"
              class="rounded-full p-1 text-xl transition"
              :class="momentFlash ? 'text-accent' : 'text-muted hover:text-accent'"
              :aria-label="momentFlash ? t('capture.marked') : t('capture.markMoment')"
              :title="momentFlash ? t('capture.marked') : t('capture.markMoment')"
              @click="markMoment"
            >
              <svg viewBox="0 0 24 24" :fill="momentFlash ? 'currentColor' : 'none'" stroke="currentColor" stroke-width="2" class="h-5 w-5" aria-hidden="true">
                <path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z" />
              </svg>
            </button>
            <FavoriteButton :item="favItem" class="text-xl" />
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
        <!-- Hero artwork (UXS-014): live intelligence + Ask/Insights actions + the summary all sit
             OVER the image, reclaiming the vertical space of separate stacked blocks. -->
        <div
          class="group relative mt-3 aspect-square w-full overflow-hidden rounded-2xl border border-border bg-elevated"
        >
          <img
            v-if="artwork"
            :src="artwork"
            :alt="episode.podcast_title ?? episode.title"
            class="h-full w-full object-cover"
          />
          <div class="absolute inset-0 flex flex-col justify-between">
            <!-- Top: live intelligence (left) + Ask/Insights pull-out actions (right) -->
            <div class="flex items-start justify-between gap-2 p-3">
              <div class="min-w-0">
                <div
                  v-if="activeInsight"
                  class="rounded-xl bg-canvas/80 px-3 py-2 backdrop-blur"
                >
                  <span class="lp-kicker block leading-none">{{ t('player.insightNow') }}</span>
                  <span class="mt-1 block text-sm font-semibold line-clamp-3">{{ activeInsight.text }}</span>
                </div>
                <div
                  v-else-if="speakingNow"
                  class="inline-flex items-baseline gap-1 rounded-full bg-canvas/70 px-3 py-1.5 backdrop-blur"
                >
                  <span class="lp-kicker leading-none">{{ t('player.speakingNow') }}</span>
                  <span class="text-sm font-semibold">{{ speakingNow }}</span>
                </div>
              </div>
              <!-- Per-episode reach (UXS-014): listeners · opens · insights, with a tiny opens-over-time
                   sparkline. The insights score opens the Knowledge panel. -->
              <div
                v-if="!panelOpen"
                class="shrink-0 rounded-xl bg-canvas/80 px-3 py-2 text-right backdrop-blur"
              >
                <div class="flex items-center justify-end gap-3 text-xs font-bold leading-none">
                  <span
                    v-if="stats && stats.listeners > 0"
                    class="flex items-center gap-1 text-canvas-foreground"
                    :aria-label="t('stats.listeners', stats.listeners, { named: { count: stats.listeners } })"
                    :title="t('stats.listeners', stats.listeners, { named: { count: stats.listeners } })"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-3.5 w-3.5" aria-hidden="true"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
                    {{ compact(stats.listeners) }}
                  </span>
                  <span
                    v-if="stats && stats.opens > 0"
                    class="flex items-center gap-1 text-canvas-foreground"
                    :aria-label="t('stats.opens', stats.opens, { named: { count: stats.opens } })"
                    :title="t('stats.opens', stats.opens, { named: { count: stats.opens } })"
                  >▶ {{ compact(stats.opens) }}</span>
                  <button
                    type="button"
                    class="flex items-center gap-1 text-accent transition hover:opacity-80"
                    :aria-label="t('kp.insights')"
                    @click="panelOpen = true"
                  >💡 {{ insights.length }}</button>
                </div>
                <Sparkline
                  v-if="stats && statsSeries.some((n) => n > 0)"
                  :values="statsSeries"
                  :width="116"
                  :height="22"
                  class="mt-1.5 block text-accent"
                />
              </div>
            </div>

            <!-- Bottom: the FULL summary over a legibility gradient. Hidden by default so the artwork
                 reads clean; slides up + fades in on hover/focus (and is always shown on touch, where
                 there's no hover). The fixed-square hero means length never shifts the layout. -->
            <div
              v-if="episode.summary_text || episode.summary_title"
              tabindex="0"
              role="region"
              :aria-label="t('player.summaryRegion')"
              class="max-h-[66%] translate-y-full overflow-y-auto bg-gradient-to-t from-black/95 via-black/85 to-black/40 px-5 pb-5 pt-6 opacity-0 backdrop-blur-[2px] transition-all duration-300 ease-out group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:translate-y-0 group-focus-within:opacity-100 focus-visible:translate-y-0 focus-visible:opacity-100 [@media(hover:none)]:translate-y-0 [@media(hover:none)]:opacity-100"
            >
              <p class="whitespace-pre-line border-l-2 border-accent pl-4 font-display text-base font-semibold leading-snug tracking-tight text-white drop-shadow sm:text-lg">{{ episode.summary_text || episode.summary_title }}</p>
            </div>
          </div>
        </div>

        <audio
          ref="audioEl"
          :src="audioUrl ?? undefined"
          preload="metadata"
          class="hidden"
          @loadedmetadata="onLoadedMetadata"
          @timeupdate="onTimeUpdate"
          @play="playing = true"
          @pause="playing = false"
          @ended="onEnded"
          @error="audioError = true"
        />

        <p v-if="audioError" class="mt-4 rounded-2xl border border-border bg-surface p-4 text-danger">
          {{ t('player.audioError') }}
        </p>
        <PlayerControls
          v-else-if="audioUrl"
          class="mt-4"
          :playing="playing"
          :current-time="currentTime"
          :duration="duration"
          :rate="rate"
          @toggle="toggle"
          @seek="seek"
          @skip="skip"
          @cycle-rate="cycleRate"
        />
        <p v-else class="mt-4 rounded-2xl border border-border bg-surface p-4 text-muted">
          {{ t('player.audioUnavailable') }}
        </p>
      </div>

      <!-- Middle: synced transcript -->
      <div ref="transcriptEl" class="mt-6 scroll-mt-20 lg:mt-0 lg:flex lg:max-h-[70vh] lg:flex-col">
        <!-- Manual sync nudge: align the highlight with the played audio (ad-insertion drift). -->
        <div
          v-if="segments.length"
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
              :class="syncOffset !== 0 ? 'text-accent' : 'text-muted'"
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
          :can-capture="auth.isAuthenticated"
          :saved-segment-ids="savedSegmentIds"
          class="min-h-0 lg:flex-1"
          @seek="seekContent"
          @insight="openInsight"
          @capture="onCaptureParagraph"
        />
        <p v-else class="rounded-2xl border border-border bg-surface p-4 text-muted">
          {{ t('player.transcriptPending') }}
        </p>
      </div>

      <!-- Knowledge Panel: persistent rail on desktop, overlay sheet on mobile. -->
      <div
        v-if="panelOpen"
        class="fixed inset-x-0 bottom-0 top-8 z-40 overflow-hidden rounded-t-2xl border-t border-border lg:static lg:top-auto lg:z-auto lg:mt-0 lg:max-h-[70vh] lg:rounded-2xl lg:border"
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
          @close="panelOpen = false"
        />
      </div>
      <!-- Mobile backdrop -->
      <div
        v-if="panelOpen"
        class="fixed inset-0 z-30 bg-black/50 lg:hidden"
        @click="panelOpen = false"
      />
    </div>
  </section>
</template>
