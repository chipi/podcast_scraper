<script setup lang="ts">
/**
 * "▶ Play from 1:05" — the ONE jump-to-a-moment control (operator 2026-10-05).
 *
 * It PLAYS. An explicit ▶ starts playback (operator 2026-10-05: "tap = open and play just for
 * Resume and where we have an explicit play icon"; every other episode tap only opens). So a `to`
 * gains `?play=1`, the intent PlayerView honours after it seeks; a host handling `click` itself
 * must seek AND play.
 *
 * Search said "▶ Play from 0:20"; Saved, Revisit, the episode-notes panel and topic perspectives said
 * a bare "▶ 1:05", and the listening recap said "Open line". Same action, three spellings, seven
 * hand-rolled copies. This is the one.
 *
 * A link when it is given `to` (navigating to the player at the moment), a button otherwise (the
 * host seeks or routes itself on `click`). A moment with no timestamp shows `fallback` instead of a
 * time, so the control never invents a 0:00.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, type RouteLocationRaw } from "vue-router"
import { formatTime } from "../player/transcriptSync"

const props = defineProps<{
  seconds: number | null | undefined
  to?: RouteLocationRaw
  fallback?: string
}>()
defineEmits<{ (e: "click", ev: MouseEvent): void }>()

const { t } = useI18n()
const label = computed(() =>
  props.seconds != null
    ? t("search.playHere", { time: formatTime(props.seconds) })
    : (props.fallback ?? ""),
)
/** The destination with `play=1` added — the ▶ means play, not just open. */
const target = computed<RouteLocationRaw | undefined>(() => {
  const to = props.to
  if (!to || typeof to === "string") return to
  return { ...to, query: { ...("query" in to ? to.query : {}), play: "1" } } as RouteLocationRaw
})
const cls = "shrink-0 whitespace-nowrap font-mono text-xs font-bold text-accent no-underline"
</script>

<template>
  <RouterLink v-if="target" :to="target" :class="cls" data-testid="play-from">▶ {{ label }}</RouterLink>
  <button v-else type="button" :class="cls" data-testid="play-from" @click="$emit('click', $event)">
    ▶ {{ label }}
  </button>
</template>
