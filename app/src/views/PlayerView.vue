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
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import PlayerControls from '../components/PlayerControls.vue'
import TranscriptList from '../components/TranscriptList.vue'
import { activeSegmentIndex, PLAYBACK_RATES } from '../player/transcriptSync'
import { getAudioSource, getEpisode, getPlayback, getSegments, putPlayback } from '../services/api'
import type { EpisodeDetail, Segment } from '../services/types'

const props = defineProps<{ slug: string }>()
const { t } = useI18n()

const episode = ref<EpisodeDetail | null>(null)
const segments = ref<Segment[]>([])
const audioUrl = ref<string | null>(null)
const loading = ref(true)
const notFound = ref(false)

const audioEl = ref<HTMLAudioElement | null>(null)
const playing = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const rate = ref(1)

let resumeSeconds = 0
let lastSaved = 0

const activeIndex = computed(() => activeSegmentIndex(segments.value, currentTime.value))
const artwork = computed(() => episode.value?.episode_image_url || episode.value?.feed_image_url)
const speakingNow = computed(() => {
  const s = activeIndex.value >= 0 ? segments.value[activeIndex.value]?.speaker : null
  if (!s) return null
  return s.startsWith('person:') ? s.slice('person:'.length).replace(/-/g, ' ') : s
})

async function load(slug: string): Promise<void> {
  loading.value = true
  notFound.value = false
  segments.value = []
  audioUrl.value = null
  resumeSeconds = 0
  try {
    const [detail, segs, audio, playback] = await Promise.all([
      getEpisode(slug),
      getSegments(slug).catch(() => null),
      getAudioSource(slug).catch(() => null),
      getPlayback(slug).catch(() => null),
    ])
    episode.value = detail
    segments.value = segs?.segments ?? []
    audioUrl.value = audio?.url ?? null
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
  if (resumeSeconds > 1 && resumeSeconds < duration.value - 1) {
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

    <div v-else-if="episode" class="mt-3 lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:gap-8">
      <!-- Left rail: masthead + intelligent artwork zone + controls -->
      <div>
        <span class="lp-kicker">{{ episode.podcast_title }}</span>
        <h1 class="mt-1 font-display text-3xl font-extrabold leading-tight tracking-tight">
          {{ episode.title }}
        </h1>

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
            <div v-if="speakingNow" class="self-start rounded-full bg-canvas/70 px-3 py-1.5 backdrop-blur">
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
          @ended="playing = false"
        />

        <PlayerControls
          v-if="audioUrl"
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

      <!-- Right: synced transcript -->
      <div class="mt-6 lg:mt-0 lg:max-h-[70vh]">
        <TranscriptList
          v-if="segments.length"
          :segments="segments"
          :active-index="activeIndex"
          class="lg:max-h-[70vh]"
          @seek="seek"
        />
        <p v-else class="rounded-2xl border border-border bg-surface p-4 text-muted">
          {{ t('player.transcriptPending') }}
        </p>
      </div>
    </div>
  </section>
</template>
