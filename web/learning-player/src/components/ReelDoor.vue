<script setup lang="ts">
/**
 * "▶ 3 min on this" — the way into a topic's, theme's or storyline's reel from its own card or
 * page (operator 2026-10-11): up to fifteen minutes of clips about it across episodes
 * (`TopicReelView`), named by what it actually runs ("real length everywhere").
 * It sits on the subject page's own "listen" row, beside "Learn this in 7 days": the Follow pill's
 * size, in its own colour (`listen`), so the two read as different options and neither as Follow,
 * which owns the accent (operator 2026-10-11). ▶ says it plays now.
 *
 * It appears only once the reel is known to have at least two clips, so it never opens an empty
 * reel; a subject nobody speaks to has no door.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { BOARD_REEL_PREFIX, getTopicReel, type ShortListenKind } from '../services/api'
import { track } from '../services/analytics'
import { reelMinutes } from '../utils/reelLength'

const props = withDefaults(
  defineProps<{ subjectId: string; label: string; source?: 'subject_page' | 'board' }>(),
  { source: 'subject_page' },
)
// A board's own short listen (`board:<id>`, #2340) opens through the same door, from the board.
const kind = computed<ShortListenKind>(() =>
  props.subjectId.startsWith(BOARD_REEL_PREFIX)
    ? 'board'
    : props.subjectId.startsWith('thc:')
      ? 'storyline'
      : props.subjectId.startsWith('tc:')
        ? 'theme'
        : 'topic',
)
const { t } = useI18n()
const MINUTES = 15
const MIN_CLIPS = 2
const ready = ref(false)
const minutes = ref(MINUTES)

let latest = 0
async function check(): Promise<void> {
  const run = ++latest
  ready.value = false
  const reel = await getTopicReel(props.subjectId, MINUTES).catch(() => null)
  if (run !== latest) return
  ready.value = (reel?.clips.length ?? 0) >= MIN_CLIPS
  minutes.value = reelMinutes(reel?.total_seconds ?? 0, MINUTES)
}
watch(() => props.subjectId, check, { immediate: true })
</script>

<template>
  <RouterLink
    v-if="ready"
    :to="{ name: 'topic-reel', params: { id: subjectId }, query: { minutes: String(MINUTES) } }"
    class="lp-tap inline-flex shrink-0 items-center gap-1 rounded-full px-3 py-1 text-xs font-bold no-underline transition bg-listen/15 text-listen ring-1 ring-inset ring-listen/40 hover:bg-listen/25"
    :aria-label="t('reel.doorLabel', { minutes, topic: label }, minutes)"
    data-testid="reel-door"
    @click="track('reel_open', { source, kind })"
  >
    <span aria-hidden="true">▶ {{ t('reel.door', { minutes }) }}</span>
  </RouterLink>
</template>
