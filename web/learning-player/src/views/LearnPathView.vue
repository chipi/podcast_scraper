<script setup lang="ts">
/**
 * The whole week of a "Learn X this week" path (`/learn`, operator 2026-10-11): every day in order,
 * the open ones playable and markable, the rest saying when they open; "Stop this path" (confirmed
 * in place). Without a path, it says so and points back to Discover.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import LearnStep from '../components/LearnStep.vue'
import { getLearn, learnStepDone, stopLearn, type LearningPath } from '../services/api'
import { track } from '../services/analytics'

const { t } = useI18n()
const router = useRouter()
const path = ref<LearningPath | null>(null)
const loaded = ref(false)
const busy = ref(false)
const confirming = ref(false)

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
    <button type="button" class="lp-nav" :aria-label="t('nav.back')" @click="goBack"><BackIcon /></button>
    <template v-if="path">
      <p class="lp-kicker mt-4">{{ t('learn.kicker') }}</p>
      <h1 class="font-display text-3xl font-extrabold tracking-tight" data-testid="learn-view-title">{{ path.subject_label }}</h1>
      <p class="mt-1 text-sm text-muted">{{ t('learn.dayOf', { n: path.day, total: path.steps.length }) }}</p>
      <ol class="mt-6 flex flex-col gap-3">
        <li v-for="(s, i) in path.steps" :key="s.day"><LearnStep :step="s" :busy="busy" @done="markDone(i)" /></li>
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
