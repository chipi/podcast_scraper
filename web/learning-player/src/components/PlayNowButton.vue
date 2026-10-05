<script setup lang="ts">
/**
 * ▶ — start this episode NOW (operator 2026-10-05: play beside the heart and queue on What's new).
 *
 * Opens the player with `?play=1`, the intent PlayerView already honours for Home's Resume: start
 * after the resume seek, once, and stay paused if the browser's autoplay policy refuses. A row's
 * title opens the episode to LOOK at it; this is for when you already know you want to listen.
 *
 * A `<button>`, not a link, so the overlay plate `EpisodeActions` puts on its direct-child buttons
 * reaches it like its siblings.
 */
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'

const props = defineProps<{ slug: string }>()
const emit = defineEmits<{ play: [] }>()
const { t } = useI18n()
const router = useRouter()

function playNow(): void {
  emit('play')
  void router.push({ name: 'player', params: { slug: props.slug }, query: { play: '1' } })
}
</script>

<template>
  <button
    type="button"
    class="lp-tap z-30 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
    data-testid="play-now"
    @click.prevent.stop="playNow"
  >
    <span aria-hidden="true" class="text-xs">▶</span>
    <span class="sr-only">{{ t('player.play') }}</span>
  </button>
</template>
