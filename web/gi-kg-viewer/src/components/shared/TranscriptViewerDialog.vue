<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { fetchWithTimeout } from '../../api/httpClient'
import {
  audioRelpathFromTranscriptRelpath,
  corpusMediaFileViewUrl,
  corpusTextFileViewUrl,
} from '../../utils/transcriptSourceDisplay'
import {
  buildTranscriptHighlightSegments,
  DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES,
  formatSegmentTimeRange,
  parseTranscriptSegmentsJson,
  segmentsSidecarRelpathFromTranscriptRelpath,
  splitTranscriptAroundHighlight,
  transcriptExceedsMaxBytes,
  type TranscriptHighlightParts,
  type TranscriptHighlightSegment,
  type TranscriptSegment,
} from '../../utils/transcriptViewerModel'
import { StaleGeneration } from '../../utils/staleGeneration'
import AppDialog from './AppDialog.vue'

export type TranscriptViewerOpenPayload = {
  corpusRoot: string
  transcriptRelpath: string
  /** Pre-built tab URL (same as “Open transcript in new tab”). */
  rawTabUrl: string
  charStart?: unknown
  charEnd?: unknown
  /**
   * When non-empty, highlights merged spans in one body (e.g. all insight supporting quotes).
   * Ignores ``charStart`` / ``charEnd``.
   */
  charRanges?: ReadonlyArray<{ charStart?: unknown; charEnd?: unknown }>
  /** Pre-formatted audio line (GI quote), shown in header. */
  audioTimingLabel?: string | null
  /** Pre-formatted GI char range (e.g. `Characters 0–60`), shown in header. */
  charPositionLabel?: string | null
  /** Dialog subtitle (e.g. episode or file name). */
  subtitle?: string | null
  /** Optional GI quote start time (ms) to seek local audio on open. */
  audioSeekStartMs?: unknown
  maxBytes?: number
}

const dialogOpen = ref(false)
const loading = ref(false)
const errorText = ref<string | null>(null)
const oversized = ref(false)
const rawTabUrl = ref('')
const highlightParts = ref<TranscriptHighlightParts | null>(null)
const highlightSegments = ref<TranscriptHighlightSegment[] | null>(null)
const plainFullText = ref('')
const segments = ref<TranscriptSegment[] | null>(null)
const audioTimingLabel = ref<string | null>(null)
const charPositionLabel = ref<string | null>(null)
const subtitle = ref<string | null>(null)
const audioUrl = ref<string | null>(null)
const pendingSeekMs = ref<number | null>(null)
const audioEl = ref<HTMLAudioElement | null>(null)
const openMaxBytes = ref(DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES)

const transcriptOpenGate = new StaleGeneration()

function resetState(): void {
  loading.value = false
  errorText.value = null
  oversized.value = false
  rawTabUrl.value = ''
  highlightParts.value = null
  highlightSegments.value = null
  plainFullText.value = ''
  segments.value = null
  audioTimingLabel.value = null
  charPositionLabel.value = null
  subtitle.value = null
  audioUrl.value = null
  pendingSeekMs.value = null
  openMaxBytes.value = DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES
}

function close(): void {
  dialogOpen.value = false
}

function onDialogOpenChange(next: boolean): void {
  dialogOpen.value = next
}

function releaseAudioOnClose(): void {
  // Stop playback and release the media resource when the dialog closes; otherwise
  // audio keeps playing in the background and the <audio> element lingers.
  const el = audioEl.value
  if (el) {
    try {
      el.pause()
    } catch {
      // ignore — element may already be detached
    }
  }
  audioUrl.value = null
}

watch(dialogOpen, (open) => {
  if (!open) releaseAudioOnClose()
})

async function scrollFirstHighlightIntoView(): Promise<void> {
  await nextTick()
  requestAnimationFrame(() => {
    const body = document.querySelector('[data-testid="transcript-viewer-body"]')
    body
      ?.querySelector('[data-testid="transcript-viewer-highlight"]')
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  })
}

async function fetchSegmentsJson(
  corpusRoot: string,
  transcriptRelpath: string,
  maxBytes: number,
  seq: number,
): Promise<TranscriptSegment[] | null> {
  const segRel = segmentsSidecarRelpathFromTranscriptRelpath(transcriptRelpath)
  const url = corpusTextFileViewUrl(corpusRoot, segRel)
  let res: Response
  try {
    res = await fetchWithTimeout(url, { credentials: 'same-origin' })
  } catch {
    return null
  }
  if (transcriptOpenGate.isStale(seq)) {
    return null
  }
  if (!res.ok) {
    return null
  }
  const len = res.headers.get('Content-Length')
  if (transcriptExceedsMaxBytes(len, 0, maxBytes)) {
    return null
  }
  let rawText: string
  try {
    rawText = await res.text()
  } catch {
    return null
  }
  if (transcriptOpenGate.isStale(seq)) {
    return null
  }
  if (transcriptExceedsMaxBytes(len, new TextEncoder().encode(rawText).length, maxBytes)) {
    return null
  }
  try {
    const data = JSON.parse(rawText) as unknown
    return parseTranscriptSegmentsJson(data)
  } catch {
    return null
  }
}

function applyPendingSeek(): void {
  const el = audioEl.value
  const ms = pendingSeekMs.value
  if (!el || ms == null || !Number.isFinite(ms) || ms < 0) {
    return
  }
  el.currentTime = ms / 1000
  pendingSeekMs.value = null
}

function seekToMs(ms: number): void {
  if (!Number.isFinite(ms) || ms < 0) {
    return
  }
  const el = audioEl.value
  if (!el) {
    pendingSeekMs.value = ms
    return
  }
  el.currentTime = ms / 1000
}

async function probeAndSetAudio(url: string, seq: number): Promise<void> {
  // Most pre-Wave-3 episodes have no persisted media. Probe existence first so we
  // only render the <audio> player when there is something to play, instead of a
  // broken control on every legacy episode.
  try {
    const res = await fetchWithTimeout(url, { method: 'HEAD', credentials: 'same-origin' })
    if (transcriptOpenGate.isStale(seq)) {
      return
    }
    if (res.ok) {
      audioUrl.value = url
    }
  } catch {
    // Network/timeout — leave the player hidden.
  }
}

async function open(payload: TranscriptViewerOpenPayload): Promise<void> {
  resetState()
  const seq = transcriptOpenGate.bump()
  const maxBytes = payload.maxBytes ?? DEFAULT_TRANSCRIPT_VIEWER_MAX_BYTES
  openMaxBytes.value = maxBytes
  rawTabUrl.value = payload.rawTabUrl
  audioTimingLabel.value = payload.audioTimingLabel ?? null
  charPositionLabel.value = payload.charPositionLabel ?? null
  subtitle.value = payload.subtitle ?? null
  const seekRaw = payload.audioSeekStartMs
  const seekNum = seekRaw == null ? NaN : Number(seekRaw)
  pendingSeekMs.value = Number.isFinite(seekNum) && seekNum >= 0 ? seekNum : null
  const mediaRel = audioRelpathFromTranscriptRelpath(payload.transcriptRelpath)
  void probeAndSetAudio(corpusMediaFileViewUrl(payload.corpusRoot, mediaRel), seq)
  loading.value = true
  errorText.value = null
  dialogOpen.value = true

  const url = corpusTextFileViewUrl(payload.corpusRoot, payload.transcriptRelpath)
  let res: Response
  try {
    res = await fetchWithTimeout(url, { credentials: 'same-origin' })
  } catch (e) {
    if (transcriptOpenGate.isCurrent(seq)) {
      loading.value = false
      errorText.value = e instanceof Error ? e.message : String(e)
    }
    return
  }

  if (transcriptOpenGate.isStale(seq)) {
    return
  }

  if (!res.ok) {
    if (transcriptOpenGate.isCurrent(seq)) {
      loading.value = false
      errorText.value = `HTTP ${res.status} loading transcript`
    }
    return
  }

  const contentLength = res.headers.get('Content-Length')
  if (transcriptExceedsMaxBytes(contentLength, 0, maxBytes)) {
    if (transcriptOpenGate.isCurrent(seq)) {
      loading.value = false
      oversized.value = true
    }
    return
  }

  let text: string
  try {
    text = await res.text()
  } catch (e) {
    if (transcriptOpenGate.isCurrent(seq)) {
      loading.value = false
      errorText.value = e instanceof Error ? e.message : String(e)
    }
    return
  }

  if (transcriptOpenGate.isStale(seq)) {
    return
  }

  const byteLen = new TextEncoder().encode(text).length
  if (transcriptExceedsMaxBytes(contentLength, byteLen, maxBytes)) {
    if (transcriptOpenGate.isCurrent(seq)) {
      loading.value = false
      oversized.value = true
    }
    return
  }

  const multi =
    Array.isArray(payload.charRanges) && payload.charRanges.length > 0
  if (multi) {
    const ranges = payload.charRanges as ReadonlyArray<{
      charStart: unknown
      charEnd: unknown
    }>
    highlightSegments.value = buildTranscriptHighlightSegments(text, ranges)
    highlightParts.value = null
  } else {
    highlightParts.value = splitTranscriptAroundHighlight(text, payload.charStart, payload.charEnd)
    highlightSegments.value = null
  }
  plainFullText.value = text
  if (transcriptOpenGate.isCurrent(seq)) {
    loading.value = false
  }

  void fetchSegmentsJson(payload.corpusRoot, payload.transcriptRelpath, maxBytes, seq).then((s) => {
    if (transcriptOpenGate.isCurrent(seq)) {
      segments.value = s
    }
  })

  const hasMark =
    Boolean(highlightParts.value?.highlight) ||
    Boolean(highlightSegments.value?.some((s) => s.type === 'mark' && s.text))
  if (hasMark) {
    await scrollFirstHighlightIntoView()
  }
}

defineExpose({ open, close, seekToMs })
</script>

<template>
  <AppDialog
    :open="dialogOpen"
    title="Transcript"
    testid="transcript-viewer-dialog"
    close-testid="transcript-viewer-close"
    width-class="w-[min(100%,42rem)]"
    max-height-class="max-h-[min(92vh,48rem)]"
    @update:open="onDialogOpenChange"
  >
    <template #header>
      <p
        v-if="subtitle"
        class="mt-0.5 truncate text-[11px] text-muted"
      >
        {{ subtitle }}
      </p>
      <p
        v-if="audioTimingLabel"
        class="mt-1 text-[11px] text-surface-foreground"
      >
        <span class="font-medium text-muted">Audio: </span>{{ audioTimingLabel }}
      </p>
      <p
        v-if="charPositionLabel"
        class="mt-1 text-[11px] text-surface-foreground"
        data-testid="transcript-viewer-char-range"
      >
        <span class="font-medium text-muted">Passage: </span>{{ charPositionLabel }}
      </p>
      <p
        v-if="rawTabUrl"
        class="mt-1"
      >
        <a
          class="text-[11px] font-medium text-primary underline decoration-primary/40 underline-offset-2"
          :href="rawTabUrl"
          target="_blank"
          rel="noopener noreferrer"
          data-testid="transcript-viewer-open-raw"
        >Open raw transcript in new tab</a>
      </p>
      <p class="mt-1 text-[10px] leading-snug text-muted">
        Highlight position is approximate if the server serves a different transcript variant than GI indexed (e.g. cleaned vs raw).
      </p>
      <audio
        v-if="audioUrl"
        ref="audioEl"
        data-testid="transcript-viewer-audio"
        class="mt-2 w-full"
        controls
        preload="metadata"
        aria-label="Episode audio"
        :src="audioUrl"
        @loadedmetadata="applyPendingSeek"
      />
    </template>

    <div class="px-4 py-3 text-xs leading-relaxed">
      <p
        v-if="loading"
        class="text-muted"
      >
        Loading…
      </p>
      <p
        v-else-if="errorText"
        class="text-destructive"
      >
        {{ errorText }}
      </p>
      <div
        v-else-if="oversized"
        class="space-y-2 text-muted"
      >
        <p>
          This transcript is too large to load in the viewer (over
          {{ Math.round(openMaxBytes / (1024 * 1024)) }} MiB). Use the link in the header to open the raw file.
        </p>
      </div>
      <template v-else>
        <div
          v-if="highlightSegments"
          data-testid="transcript-viewer-body"
          class="select-text whitespace-pre-wrap break-words rounded border border-border bg-canvas/80 p-2 font-mono text-[11px] text-surface-foreground"
        >
          <template v-for="(seg, si) in highlightSegments" :key="si">
            <mark
              v-if="seg.type === 'mark' && seg.text"
              data-testid="transcript-viewer-highlight"
              class="rounded-sm bg-primary/25 px-0.5 text-surface-foreground"
            >{{ seg.text }}</mark>
            <span v-else-if="seg.type === 'text' && seg.text">{{ seg.text }}</span>
          </template>
        </div>
        <div
          v-else-if="highlightParts"
          data-testid="transcript-viewer-body"
          class="select-text whitespace-pre-wrap break-words rounded border border-border bg-canvas/80 p-2 font-mono text-[11px] text-surface-foreground"
        >
          <span>{{ highlightParts.before }}</span>
          <mark
            v-if="highlightParts.highlight"
            data-testid="transcript-viewer-highlight"
            class="rounded-sm bg-primary/25 px-0.5 text-surface-foreground"
          >{{ highlightParts.highlight }}</mark>
          <span>{{ highlightParts.after }}</span>
        </div>
        <div
          v-else
          data-testid="transcript-viewer-body"
          class="select-text whitespace-pre-wrap break-words rounded border border-border bg-canvas/80 p-2 font-mono text-[11px] text-surface-foreground"
        >
          {{ plainFullText }}
        </div>

        <details
          v-if="segments && segments.length > 0"
          class="mt-3 rounded border border-border bg-elevated/40 p-2"
        >
          <summary class="cursor-pointer text-[11px] font-medium text-surface-foreground">
            Timeline ({{ segments.length }} segments)
          </summary>
          <ol
            class="mt-2 max-h-48 list-decimal space-y-1.5 overflow-y-auto pl-4 text-[11px] text-muted"
            data-testid="transcript-viewer-timeline"
          >
            <li
              v-for="(seg, i) in segments"
              :key="i"
              class="pl-0.5"
            >
              <!-- I5: when audio is loaded, each row seeks the player to its start. -->
              <button
                v-if="audioUrl"
                type="button"
                class="group w-full cursor-pointer rounded px-0.5 text-left hover:bg-primary/10 focus-visible:outline focus-visible:outline-1 focus-visible:outline-primary"
                :data-testid="`transcript-viewer-timeline-seek-${i}`"
                :aria-label="`Play from ${formatSegmentTimeRange(seg)}`"
                @click="seekToMs(Math.round(seg.startSec * 1000))"
              >
                <span class="font-mono text-[10px] text-primary group-hover:underline">{{ formatSegmentTimeRange(seg) }}</span>
                <span class="text-surface-foreground"> — {{ seg.text.trim() || '—' }}</span>
              </button>
              <template v-else>
                <span class="font-mono text-[10px] text-primary">{{ formatSegmentTimeRange(seg) }}</span>
                <span class="text-surface-foreground"> — {{ seg.text.trim() || '—' }}</span>
              </template>
            </li>
          </ol>
        </details>
      </template>
    </div>
  </AppDialog>
</template>
