<script setup lang="ts">
/**
 * "Learn this in a week" on a topic card and a theme or storyline page (operator 2026-10-11: another
 * door into the seven-day path than Discover). Beside "▶ 15 min on this" on the page's "listen"
 * row, and styled UNLIKE it (operator 2026-10-11: two separate options): an outlined pill with a
 * calendar mark, because it starts a week rather than playing now.
 *
 * - Shown only when the subject has a path (`GET /discover/learn/subjects/{id}`).
 * - Already learning this: "Day 3 of 7 ›", to the whole week.
 * - Learning something else: starting replaces it, so it asks in place first.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import { getLearn, getLearnSubject, startLearn, type LearningPath, type LearnSubject } from '../services/api'
import { track } from '../services/analytics'
import CalendarMark from './CalendarMark.vue'

const props = defineProps<{ subjectId: string }>()
const { t } = useI18n()
const router = useRouter()
const subject = ref<LearnSubject | null>(null)
const current = ref<LearningPath | null>(null)
const confirming = ref(false)
const busy = ref(false)

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  confirming.value = false
  const [s, mine] = await Promise.all([
    getLearnSubject(props.subjectId).catch(() => null),
    getLearn('corpus').catch(() => null),
  ])
  if (run !== latest) return
  subject.value = s
  current.value = mine?.path ?? null
}
watch(() => props.subjectId, load, { immediate: true })

const learningThis = computed(() => current.value?.subject_id === props.subjectId)

async function start(): Promise<void> {
  if (!subject.value || busy.value) return
  if (current.value && !confirming.value) {
    confirming.value = true
    return
  }
  busy.value = true
  try {
    await startLearn(props.subjectId)
    track('learn_path', { action: 'start', kind: subject.value.subject_kind })
    void router.push({ name: 'learn' })
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
      :to="{ name: 'learn' }"
      class="lp-tap inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border border-border px-4 text-sm font-semibold text-canvas-foreground no-underline transition hover:bg-overlay"
      data-testid="learn-door"
    ><CalendarMark />{{ t('learn.doorDay', { n: current.day, total: current.steps.length }) }} ›</RouterLink>
    <button
      v-else-if="!confirming"
      type="button"
      class="lp-tap inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full border border-border px-4 text-sm font-semibold text-canvas-foreground transition hover:bg-overlay"
      :disabled="busy"
      data-testid="learn-door"
      @click="start"
    ><CalendarMark />{{ t('learn.door', { days: subject.steps }) }}</button>
    <span v-else class="inline-flex flex-wrap items-center gap-x-3 text-xs" role="group" data-testid="learn-door-confirm">
      <span class="text-muted">{{ t('learn.doorReplaces', { subject: current?.subject_label ?? '' }) }}</span>
      <button type="button" class="lp-tap min-h-[44px] font-bold text-accent" :disabled="busy" data-testid="learn-door-yes" @click="start">{{ t('learn.doorStart') }}</button>
      <button type="button" class="lp-tap min-h-[44px] font-bold" data-testid="learn-door-no" @click="confirming = false">{{ t('learn.doorCancel') }}</button>
    </span>
  </template>
</template>
