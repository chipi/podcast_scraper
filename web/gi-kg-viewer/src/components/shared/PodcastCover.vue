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

const broken = ref(false)

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
    broken.value = false
  },
)

function binaryUrl(relpath: string): string {
  const q = new URLSearchParams({
    path: (props.corpusPath ?? '').trim(),
    relpath: relpath.trim(),
  })
  return `/api/corpus/binary?${q.toString()}`
}

const src = computed(() => {
  if (broken.value) {
    return ''
  }
  const cp = (props.corpusPath ?? '').trim()
  const epLoc = props.episodeImageLocalRelpath?.trim()
  if (cp && epLoc) {
    return binaryUrl(epLoc)
  }
  const e = props.episodeImageUrl?.trim()
  if (e) {
    return e
  }
  const fdLoc = props.feedImageLocalRelpath?.trim()
  if (cp && fdLoc) {
    return binaryUrl(fdLoc)
  }
  const f = props.feedImageUrl?.trim()
  return f || ''
})

function onError(): void {
  broken.value = true
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
