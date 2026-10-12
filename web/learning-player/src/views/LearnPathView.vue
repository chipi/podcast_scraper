<script setup lang="ts">
/**
 * The whole week of a "Learn X this week" path (`/learn?subject=`, operator 2026-10-11): every day
 * in order, today's open (playable, markable), earlier ones closed with whether they were done (tap
 * to open and replay), later ones saying when they open; "Leave this path" (confirmed in place).
 * With no subject named: the one path straight away, or with several (up to three) a list of them.
 * Without a path, it says so and points back to Discover.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import LearnStep from '../components/LearnStep.vue'
import { getLearn, getLearningPath, learnOpenNext, learnStepDone, stopLearn, type LearningPath } from '../services/api'
import { track } from '../services/analytics'

const props = defineProps<{ subject?: string }>()
const { t } = useI18n()
const router = useRouter()
const path = ref<LearningPath | null>(null)
/** With no subject named and several paths: the list to choose from. */
const paths = ref<LearningPath[]>([])
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

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  loaded.value = false
  flipped.value = new Set()
  confirming.value = false
  let one: LearningPath | null = null
  let all: LearningPath[] = []
  if (props.subject) {
    one = await getLearningPath(props.subject).catch(() => null)
  } else {
    all = (await getLearn('corpus').catch(() => null))?.paths ?? []
    if (all.length === 1) one = all[0]
  }
  if (run !== latest) return
  path.value = one
  paths.value = one ? [] : all
  loaded.value = true
}
watch(() => props.subject, load, { immediate: true })

async function markDone(index: number): Promise<void> {
  if (busy.value || !path.value) return
  busy.value = true
  try {
    const kind = path.value.subject_kind
    path.value = await learnStepDone(path.value.subject_id, index)
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
    path.value = await learnOpenNext(path.value.subject_id)
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
  try {
    await stopLearn(path.value.subject_id)
  } catch {
    return // not stopped: the page stays, rather than pretend
  }
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
        <li v-for="s in path.steps" :key="s.day"><LearnStep
            :step="s"
            :busy="busy"
            :kind="path.subject_kind"
            :open="isOpen(s.day)"
            :can-open-next="!!path.can_open_next && s.day === path.day"
            @done="markDone(s.day - 1)"
            @next="openNext"
            @toggle="toggle(s.day)"
          /></li>
      </ol>
      <!-- A button on the right (operator 2026-10-12): it was plain grey text on the left and read
           as a caption, not as something to press. -->
      <div class="mt-8 flex justify-end">
        <button
          v-if="!confirming"
          type="button"
          class="min-h-[44px] rounded-full border border-border px-4 text-sm font-bold text-muted transition hover:text-canvas-foreground"
          data-testid="learn-stop"
          @click="confirming = true"
        >{{ t('learn.stop') }}</button>
        <p v-else class="flex flex-wrap items-center justify-end gap-4 text-sm">
          <span>{{ t('learn.stopConfirm') }}</span>
          <button type="button" class="lp-tap min-h-[44px] font-bold text-accent" data-testid="learn-stop-yes" @click="stop">{{ t('learn.stopYes') }}</button>
          <button type="button" class="lp-tap min-h-[44px] font-bold" data-testid="learn-stop-no" @click="confirming = false">{{ t('learn.stopNo') }}</button>
        </p>
      </div>
    </template>
    <template v-else-if="paths.length">
      <p class="lp-kicker mt-4">{{ t('learn.kicker') }}</p>
      <h1 class="font-display text-3xl font-extrabold tracking-tight">{{ t('learn.yourPaths') }}</h1>
      <ul class="mt-6 flex flex-col gap-3">
        <li v-for="p in paths" :key="p.subject_id">
          <RouterLink
            :to="{ name: 'learn', query: { subject: p.subject_id } }"
            class="lp-tap block rounded-xl border border-border p-4 no-underline"
            data-testid="learn-path-link"
          >
            <span class="block font-semibold text-canvas-foreground">{{ p.subject_label }}</span>
            <span class="text-sm text-muted">{{ t('learn.dayOf', { n: p.day, total: p.steps.length }) }}</span>
          </RouterLink>
        </li>
      </ul>
    </template>
    <p v-else-if="loaded" class="mt-6 text-sm text-muted" data-testid="learn-none">
      {{ t('learn.none') }}
      <RouterLink :to="{ name: 'browse' }" class="font-bold text-accent">{{ t('learn.toDiscover') }}</RouterLink>
    </p>
  </section>
</template>
