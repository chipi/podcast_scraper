<script setup lang="ts">
/**
 * "Learn X this week" on Discover (operator 2026-10-11): a seven-day path through a theme or
 * storyline, one step a day. With a path: its name, "Day 3 of 7", a dot per day, and the step to
 * do now (the earliest open day not yet done) with "See the whole week". Without one: a few themes
 * and storylines — under Mine those of the listener's world, under Everyone the rising ones — each
 * in its kind's colours, one tap to start. Signed in only (the path is the listener's own).
 */
import { computed, onActivated, onDeactivated, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import LearnStep from './LearnStep.vue'
import { getLearn, learnOpenNext, learnStepDone, startLearn, type LearningPath, type LearnSubject } from '../services/api'
import { track } from '../services/analytics'
import { kindPill } from '../utils/interests'

const props = defineProps<{ scope: 'mine' | 'corpus' }>()
const { t } = useI18n()

const path = ref<LearningPath | null>(null)
const candidates = ref<LearnSubject[]>([])
const busy = ref(false)

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  const got = await getLearn(props.scope).catch(() => null)
  if (run !== latest || !got) return
  path.value = got.path
  candidates.value = got.candidates
}
watch(() => props.scope, load, { immediate: true })
// Discover is kept alive, and the path changes elsewhere — a day done or the path stopped on the
// whole-week page — so a return re-reads it (only after the tab was left once: see DiscoveryList).
let leftOnce = false
onDeactivated(() => {
  leftOnce = true
})
onActivated(() => {
  if (leftOnce) void load()
})

/** The step to do now: the earliest open day not yet done; with every open day done, today's. */
const now = computed(() => {
  const steps = path.value?.steps ?? []
  return steps.find((s) => s.unlocked && !s.done) ?? steps.find((s) => s.day === path.value?.day) ?? null
})
const doneCount = computed(() => path.value?.steps.filter((s) => s.done).length ?? 0)

async function start(subject: LearnSubject): Promise<void> {
  if (busy.value) return
  busy.value = true
  try {
    path.value = await startLearn(subject.subject_id)
    track('learn_path', { action: 'start', kind: subject.subject_kind, source: 'discover' })
  } catch {
    // Nothing started: the chips stay, rather than pretend.
  } finally {
    busy.value = false
  }
}
async function markDone(): Promise<void> {
  const step = now.value
  if (!step || busy.value || !path.value) return
  busy.value = true
  try {
    const kind = path.value.subject_kind
    path.value = await learnStepDone(step.day - 1)
    track('learn_path', { action: 'step_done', kind, day: step.day })
  } catch {
    // Not recorded: the button stays.
  } finally {
    busy.value = false
  }
}
async function openNext(): Promise<void> {
  if (busy.value || !path.value) return
  busy.value = true
  try {
    const kind = path.value.subject_kind
    const was = path.value.day
    path.value = await learnOpenNext()
    track('learn_path', { action: 'open_next', kind, day: was + 1 })
  } catch {
    // Not opened: the button stays.
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section v-if="path || candidates.length" class="mt-7" data-testid="learn-this-week">
    <template v-if="path">
      <div class="mb-3 flex items-baseline justify-between gap-3">
        <h2 class="lp-section min-w-0" data-testid="learn-title">{{ t('learn.title', { subject: path.subject_label }) }}</h2>
        <RouterLink
          :to="{ name: 'learn' }"
          class="lp-tap shrink-0 whitespace-nowrap text-sm font-bold text-accent no-underline"
          data-testid="learn-see-week"
        >{{ t('learn.seeWeek') }} ›</RouterLink>
      </div>
      <p class="mb-3 flex items-center gap-3 text-sm text-muted">
        <span data-testid="learn-day">{{ t('learn.dayOf', { n: path.day, total: path.steps.length }) }}</span>
        <span class="flex gap-1" :aria-label="t('learn.progress', { done: doneCount, total: path.steps.length })" role="img">
          <span
            v-for="s in path.steps"
            :key="s.day"
            class="h-2 w-2 rounded-full"
            :class="s.done ? 'bg-accent' : s.unlocked ? 'bg-canvas-foreground/40' : 'bg-border'"
            data-testid="learn-dot"
          />
        </span>
      </p>
      <LearnStep
        v-if="now"
        :step="now"
        :busy="busy"
        :kind="path.subject_kind"
        open
        :can-open-next="!!path.can_open_next && now.day === path.day"
        @done="markDone"
        @next="openNext"
      />
    </template>
    <template v-else>
      <h2 class="lp-section mb-1">{{ t('learn.offerTitle') }}</h2>
      <p class="mb-3 text-sm text-muted">{{ t('learn.offerNote') }}</p>
      <ul class="flex flex-wrap gap-2">
        <li v-for="c in candidates" :key="c.subject_id">
          <button
            type="button"
            class="lp-tap inline-flex min-h-[44px] items-center rounded-full px-4 text-sm font-semibold transition hover:opacity-80"
            :class="kindPill(c.subject_kind)"
            :data-kind="c.subject_kind"
            :disabled="busy"
            data-testid="learn-subject"
            @click="start(c)"
          >{{ t('learn.subject', { subject: c.subject_label, days: c.steps }) }}</button>
        </li>
      </ul>
    </template>
  </section>
</template>
