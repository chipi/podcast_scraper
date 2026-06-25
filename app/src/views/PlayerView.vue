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
import KnowledgePanel from '../components/KnowledgePanel.vue'
import PlayerControls from '../components/PlayerControls.vue'
import TranscriptList from '../components/TranscriptList.vue'
import { activeInsightIndex, groundedSpansBySegment } from '../player/insights'
import { activeSegmentIndex, PLAYBACK_RATES } from '../player/transcriptSync'
import {
  getAudioSource,
  getEntities,
  getEpisode,
  getInsights,
  getPlayback,
  getSegments,
  putPlayback,
} from '../services/api'
import type { EpisodeDetail, Entity, Insight, Segment, Topic } from '../services/types'
import { formatDuration, formatPublishDate } from '../utils/format'

const props = defineProps<{ slug: string }>()
const { t, locale } = useI18n()
const router = useRouter()
const route = useRoute()
const queue = useQueueStore()

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
const artwork = computed(
  () =>
    episode.value?.artwork_url ||
    episode.value?.episode_image_url ||
    episode.value?.feed_image_url,
)
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
const speakingNow = computed(() => {
  const s = activeIndex.value >= 0 ? segments.value[activeIndex.value]?.speaker : null
  if (!s) return null
  return s.startsWith('person:') ? s.slice('person:'.length).replace(/-/g, ' ') : s
})

async function load(slug: string): Promise<void> {
  loading.value = true
  notFound.value = false
  audioError.value = false
  segments.value = []
  audioUrl.value = null
  insights.value = []
  topics.value = []
  persons.value = []
  resumeSeconds = 0
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

function toggle(): void {
  const el = audioEl.value
  if (!el) return
  if (el.paused) void el.play()
  else el.pause()
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

onMounted(() => load(props.slug))
watch(() => props.slug, (s) => load(s))
onBeforeUnmount(() => persist())
</script>

<template>
  <section>
    <RouterLink :to="{ name: 'catalog' }" class="lp-kicker no-underline">‹ {{ t('player.back') }}</RouterLink>

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
        <RouterLink
          v-if="episode.podcast_title"
          :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
          class="lp-kicker inline-block no-underline"
        >
          {{ episode.podcast_title }}
        </RouterLink>
        <h1 class="mt-1 font-display text-3xl font-extrabold leading-tight tracking-tight">
          {{ episode.title }}
        </h1>
        <p v-if="metaLine" class="mt-1 text-sm text-muted">{{ metaLine }}</p>
        <p v-if="episode.summary_text || episode.summary_title" class="mt-2 text-sm text-muted line-clamp-3">
          {{ episode.summary_text || episode.summary_title }}
        </p>

        <div
          class="relative mt-4 aspect-square w-full overflow-hidden rounded-2xl border border-border bg-elevated"
        >
          <img
            v-if="artwork"
            :src="artwork"
            :alt="episode.podcast_title ?? episode.title"
            class="h-full w-full object-cover"
          />
          <div class="pointer-events-none absolute inset-0 flex flex-col justify-between p-4">
            <div class="flex justify-end">
              <span v-if="episode.has_gi" class="rounded-full bg-canvas/70 px-3 py-1 text-xs font-bold text-grounded backdrop-blur">
                ● {{ t('player.grounded') }}
              </span>
            </div>
            <!-- Live intelligence: surface the insight being spoken; else who's speaking. -->
            <div
              v-if="activeInsight"
              class="self-start max-w-[90%] rounded-xl bg-canvas/80 px-3 py-2 backdrop-blur"
            >
              <span class="lp-kicker block leading-none">{{ t('player.insightNow') }}</span>
              <span class="mt-1 block text-sm font-semibold line-clamp-3">{{ activeInsight.text }}</span>
            </div>
            <div
              v-else-if="speakingNow"
              class="self-start rounded-full bg-canvas/70 px-3 py-1.5 backdrop-blur"
            >
              <span class="lp-kicker block leading-none">{{ t('player.speakingNow') }}</span>
              <span class="text-sm font-semibold">{{ speakingNow }}</span>
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

        <!-- Knowledge dock: one tap to reveal the panel (collapsed by default). -->
        <div v-if="!panelOpen" class="mt-3 flex gap-2">
          <button
            type="button"
            class="flex-1 rounded-full border border-border px-4 py-3 text-sm font-bold"
            @click="panelOpen = true"
          >
            💡 {{ insights.length }} {{ t('kp.insights') }}
          </button>
          <button
            type="button"
            class="flex-1 rounded-full border border-border px-4 py-3 text-sm font-bold"
            @click="panelOpen = true"
          >
            🔍 {{ t('kp.ask') }}
          </button>
        </div>
      </div>

      <!-- Middle: synced transcript -->
      <div class="mt-6 lg:mt-0 lg:flex lg:max-h-[70vh] lg:flex-col">
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
          class="min-h-0 lg:flex-1"
          @seek="seekContent"
          @insight="openInsight"
        />
        <p v-else class="rounded-2xl border border-border bg-surface p-4 text-muted">
          {{ t('player.transcriptPending') }}
        </p>
      </div>

      <!-- Knowledge Panel: persistent rail on desktop, overlay sheet on mobile. -->
      <div
        v-if="panelOpen"
        class="fixed inset-x-0 bottom-0 top-14 z-40 overflow-hidden rounded-t-2xl border-t border-border lg:static lg:z-auto lg:mt-0 lg:max-h-[70vh] lg:rounded-2xl lg:border"
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
