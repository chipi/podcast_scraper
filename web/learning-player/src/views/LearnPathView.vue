<script setup lang="ts">
/**
 * The whole week of a "Learn X this week" path (`/learn`, operator 2026-10-11): every day in order,
 * today's open (playable, markable), earlier ones closed with whether they were done (tap to open
 * and replay), later ones saying when they open; "Stop this path" (confirmed in place). Without a
 * path, it says so and points back to Discover.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import LearnStep from '../components/LearnStep.vue'
import { getLearn, learnOpenNext, learnStepDone, stopLearn, type LearningPath } from '../services/api'
import { track } from '../services/analytics'

const { t } = useI18n()
const router = useRouter()
const path = ref<LearningPath | null>(null)
const loaded = ref(false)
const busy = ref(false)
const confirming = ref(false)
/** Days whose open state the listener flipped: today starts open, every other day closed. */
const flipped = ref(new Set<number>())
const today = computed(() => path.value?.day ?? 0)
function isOpen(day: number): boolean {
  return (day === today.value) !== flipped.value.has(day)
}
function toggle(day: number): void {
  const next = new Set(flipped.value)
  if (next.has(day)) next.delete(day)
  else next.add(day)
  flipped.value = next
}

onMounted(async () => {
  const got = await getLearn('corpus').catch(() => null)
  path.value = got?.path ?? null
  loaded.value = true
})

async function markDone(index: number): Promise<void> {
  if (busy.value || !path.value) return
  busy.value = true
  try {
    const kind = path.value.subject_kind
    path.value = await learnStepDone(index)
    track('learn_path', { action: 'step_done', kind, day: index + 1 })
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
    flipped.value = new Set() // the new day is "today": it opens, the one just done closes
    track('learn_path', { action: 'open_next', kind, day: was + 1 })
  } catch {
    // Not opened: the button stays.
  } finally {
    busy.value = false
  }
}
async function stop(): Promise<void> {
  if (!path.value) return
  const kind = path.value.subject_kind
  await stopLearn().catch(() => undefined)
  track('learn_path', { action: 'stop', kind })
  void router.push({ name: 'browse' })
}
function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: 'browse' })
}
</script>

<template>
  <section class="lp-page pb-8 pt-4" data-testid="learn-view">
    <button type="button" class="lp-nav" :aria-label="t('nav.back')" @click="goBack">
      <BackIcon />
      <span>{{ t('nav.back') }}</span>
    </button>
    <template v-if="path">
      <p class="lp-kicker mt-4">{{ t('learn.kicker') }}</p>
      <h1 class="font-display text-3xl font-extrabold tracking-tight" data-testid="learn-view-title">{{ path.subject_label }}</h1>
      <p class="mt-1 text-sm text-muted">{{ t('learn.dayOf', { n: path.day, total: path.steps.length }) }}</p>
      <ol class="mt-6 flex flex-col gap-3">
        <li v-for="(s, i) in path.steps" :key="s.day"><LearnStep
            :step="s"
            :busy="busy"
            :kind="path.subject_kind"
            :open="isOpen(s.day)"
            :can-open-next="!!path.can_open_next && s.day === path.day"
            @done="markDone(i)"
            @next="openNext"
            @toggle="toggle(s.day)"
          /></li>
      </ol>
      <div class="mt-8">
        <button
          v-if="!confirming"
          type="button"
          class="lp-tap min-h-[44px] text-sm font-bold text-muted"
          data-testid="learn-stop"
          @click="confirming = true"
        >{{ t('learn.stop') }}</button>
        <p v-else class="flex flex-wrap items-center gap-4 text-sm">
          <span>{{ t('learn.stopConfirm') }}</span>
          <button type="button" class="lp-tap min-h-[44px] font-bold text-accent" data-testid="learn-stop-yes" @click="stop">{{ t('learn.stopYes') }}</button>
          <button type="button" class="lp-tap min-h-[44px] font-bold" data-testid="learn-stop-no" @click="confirming = false">{{ t('learn.stopNo') }}</button>
        </p>
      </div>
    </template>
    <p v-else-if="loaded" class="mt-6 text-sm text-muted" data-testid="learn-none">
      {{ t('learn.none') }}
      <RouterLink :to="{ name: 'browse' }" class="font-bold text-accent">{{ t('learn.toDiscover') }}</RouterLink>
    </p>
  </section>
</template>
