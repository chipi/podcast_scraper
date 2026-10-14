<script setup lang="ts">
/**
 * "Learn this in a week" on a topic card and a theme or storyline page (operator 2026-10-11: another
 * door into the seven-day path than Discover). Beside "▶ 15 min on this" on the page's "listen"
 * row: the Follow pill's size, in its own colour (`learn`), unlike the reel door and unlike Follow
 * (operator 2026-10-11: two separate options). The calendar mark says it starts a week.
 *
 * - Shown only when the subject has a path (`GET /discover/learn/subjects/{id}`).
 * - Already learning this: "Learning · day 3 of 7 ›", to its week.
 * - Learned it (a finished path, on for its week or in the history): "✓ Learned · learn again",
 *   which starts it afresh (operator 2026-10-14: say it was done, do not hide it).
 * - Up to three paths at once (operator 2026-10-11); at three, starting replaces the oldest, so it
 *   asks in place first.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import {
  getLearn,
  getLearningHistory,
  getLearnSubject,
  startLearn,
  type LearningPath,
  type LearnSubject,
} from '../services/api'
import { track } from '../services/analytics'
import CalendarMark from './CalendarMark.vue'

const props = defineProps<{ subjectId: string }>()
const { t } = useI18n()
const router = useRouter()
const subject = ref<LearnSubject | null>(null)
/** The listener's paths, oldest first. */
const paths = ref<LearningPath[]>([])
const maxPaths = ref(3)
/** Subjects whose path the listener finished, from the learning history. */
const finishedSubjects = ref(new Set<string>())
const confirming = ref(false)
const busy = ref(false)

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  confirming.value = false
  const [s, mine, history] = await Promise.all([
    getLearnSubject(props.subjectId).catch(() => null),
    getLearn('corpus').catch(() => null),
    getLearningHistory().catch(() => []),
  ])
  if (run !== latest) return
  subject.value = s
  paths.value = mine?.paths ?? []
  maxPaths.value = mine?.max_paths ?? 3
  finishedSubjects.value = new Set(history.filter((h) => h.outcome === 'finished').map((h) => h.subject_id))
}
watch(() => props.subjectId, load, { immediate: true })

const current = computed(() => paths.value.find((p) => p.subject_id === props.subjectId) ?? null)
/** Being learned now: a path on this subject not yet finished. */
const learningThis = computed(() => current.value !== null && !current.value.finished_at)
/** Learned before: a finished path, still on for its week or only in the history. */
const learned = computed(
  () => !learningThis.value && (!!current.value?.finished_at || finishedSubjects.value.has(props.subjectId)),
)
/** Paths being learned: a finished one does not count toward the three at once. */
const active = computed(() => paths.value.filter((p) => !p.finished_at))
/** At the cap, starting this replaces the oldest path being learned. */
const replaces = computed(() =>
  !learningThis.value && active.value.length >= maxPaths.value ? active.value[0] : null,
)

async function start(): Promise<void> {
  if (!subject.value || busy.value) return
  if (replaces.value && !confirming.value) {
    confirming.value = true
    return
  }
  busy.value = true
  try {
    await startLearn(props.subjectId, replaces.value?.subject_id)
    track('learn_path', { action: 'start', kind: subject.value.subject_kind, source: 'subject_page' })
    void router.push({ name: 'learn', query: { subject: props.subjectId } })
  } catch {
    // Nothing started: the door stays.
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <template v-if="subject">
    <RouterLink
      v-if="learningThis && current"
      :to="{ name: 'learn', query: { subject: props.subjectId } }"
      class="lp-tap inline-flex shrink-0 items-center gap-1 rounded-full px-3 py-1 text-xs font-bold no-underline transition bg-learn/15 text-learn ring-1 ring-inset ring-learn/40 hover:bg-learn/25"
      data-testid="learn-door"
    ><CalendarMark />{{ t('learn.doorDay', { n: current.day, total: current.steps.length }) }} ›</RouterLink>
    <button
      v-else-if="!confirming"
      type="button"
      class="lp-tap inline-flex shrink-0 items-center gap-1 rounded-full px-3 py-1 text-xs font-bold no-underline transition bg-learn/15 text-learn ring-1 ring-inset ring-learn/40 hover:bg-learn/25"
      :disabled="busy"
      data-testid="learn-door"
      :data-learned="learned ? 'true' : undefined"
      @click="start"
    ><CalendarMark />{{ learned ? t('learn.doorAgain') : t('learn.door', { days: subject.steps }) }}</button>
    <span v-else class="inline-flex flex-wrap items-center gap-x-3 text-xs" role="group" data-testid="learn-door-confirm">
      <span class="text-muted">{{ t('learn.atCap', { subject: replaces?.subject_label ?? '' }) }}</span>
      <button type="button" class="lp-tap min-h-[44px] font-bold text-accent" :disabled="busy" data-testid="learn-door-yes" @click="start">{{ t('learn.doorStart') }}</button>
      <button type="button" class="lp-tap min-h-[44px] font-bold" data-testid="learn-door-no" @click="confirming = false">{{ t('learn.doorCancel') }}</button>
    </span>
  </template>
</template>
