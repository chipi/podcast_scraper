<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    episodeImageUrl?: string | null
    feedImageUrl?: string | null
    episodeImageLocalRelpath?: string | null
    feedImageLocalRelpath?: string | null
    /** Corpus root as sent to ``path=`` on API calls; required for local artwork URLs. */
    corpusPath?: string | null
    alt: string
    sizeClass?: string
  }>(),
  {
    episodeImageUrl: null,
    feedImageUrl: null,
    episodeImageLocalRelpath: null,
    feedImageLocalRelpath: null,
    corpusPath: null,
    sizeClass: 'h-10 w-10',
  },
)

/** After ``@error``, try the next candidate (e.g. local binary 404 → remote URL). */
const fallbackStep = ref(0)

watch(
  () =>
    [
      props.episodeImageUrl,
      props.feedImageUrl,
      props.episodeImageLocalRelpath,
      props.feedImageLocalRelpath,
      props.corpusPath,
    ] as const,
  () => {
    fallbackStep.value = 0
  },
)

function binaryUrl(relpath: string): string {
  const q = new URLSearchParams({
    path: (props.corpusPath ?? '').trim(),
    relpath: relpath.trim(),
  })
  return `/api/corpus/binary?${q.toString()}`
}

const candidateSrcs = computed((): string[] => {
  const cp = (props.corpusPath ?? '').trim()
  const epLoc = props.episodeImageLocalRelpath?.trim()
  const e = props.episodeImageUrl?.trim()
  const fdLoc = props.feedImageLocalRelpath?.trim()
  const f = props.feedImageUrl?.trim()
  const raw: string[] = []
  if (cp && epLoc) {
    raw.push(binaryUrl(epLoc))
  }
  if (e) {
    raw.push(e)
  }
  if (cp && fdLoc) {
    raw.push(binaryUrl(fdLoc))
  }
  if (f) {
    raw.push(f)
  }
  const seen = new Set<string>()
  const out: string[] = []
  for (const u of raw) {
    if (u && !seen.has(u)) {
      seen.add(u)
      out.push(u)
    }
  }
  return out
})

const src = computed(() => candidateSrcs.value[fallbackStep.value] ?? '')

function onError(): void {
  const list = candidateSrcs.value
  if (fallbackStep.value < list.length - 1) {
    fallbackStep.value += 1
  } else {
    fallbackStep.value = list.length
  }
}
</script>

<template>
  <div
    :class="[
      'shrink-0 overflow-hidden rounded border border-border bg-elevated',
      sizeClass,
    ]"
    data-testid="podcast-cover"
  >
    <img
      v-if="src"
      :src="src"
      :alt="alt"
      loading="lazy"
      referrerpolicy="no-referrer"
      class="h-full w-full object-cover"
      @error="onError"
    >
    <div
      v-else
      class="flex h-full w-full items-center justify-center text-[10px] font-medium text-muted"
      aria-hidden="true"
    >
      ♪
    </div>
  </div>
</template>
