<script setup lang="ts">
/**
 * "Learn something this week" on Discover (operator 2026-10-11): where a path starts. A few themes
 * and storylines — under Mine those of the listener's world, under Everyone the rising ones — each
 * in its kind's colours; one tap starts a seven-day path and opens its week. The paths already on
 * (up to three) are listed as links to their weeks; the day to do lives on Home's "In progress",
 * the door into what was started. At three paths there is nothing to start here until one stops.
 * Signed in only (the paths are the listener's own).
 */
import { onActivated, onDeactivated, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import { ApiError, getLearn, startLearn, type LearningPath, type LearnSubject } from '../services/api'
import { track } from '../services/analytics'
import { kindPill } from '../utils/interests'

const props = defineProps<{ scope: 'mine' | 'corpus' }>()
const { t } = useI18n()
const router = useRouter()

const paths = ref<LearningPath[]>([])
const candidates = ref<LearnSubject[]>([])
const busy = ref(false)

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  const got = await getLearn(props.scope).catch(() => null)
  if (run !== latest || !got) return
  paths.value = got.paths
  candidates.value = got.candidates
}
watch(() => props.scope, load, { immediate: true })
// Discover is kept alive, and paths change elsewhere — a day done, a path stopped — so a return
// re-reads them (only after the tab was left once: see DiscoveryList).
let leftOnce = false
onDeactivated(() => {
  leftOnce = true
})
onActivated(() => {
  if (leftOnce) void load()
})

async function start(subject: LearnSubject): Promise<void> {
  if (busy.value) return
  busy.value = true
  try {
    const path = await startLearn(subject.subject_id)
    track('learn_path', { action: 'start', kind: subject.subject_kind, source: 'discover' })
    void router.push({ name: 'learn', query: { subject: path.subject_id } })
  } catch (e) {
    // Nothing started: the chips stay, rather than pretend. At the cap (another tab filled it since
    // this loaded), re-read so the offers make way for the paths that are on.
    if (e instanceof ApiError && e.status === 409) void load()
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section v-if="paths.length || candidates.length" class="mt-7" data-testid="learn-this-week">
    <template v-if="candidates.length">
      <h2 class="lp-section mb-1" data-testid="learn-title">{{ t('learn.offerTitle') }}</h2>
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
    <div v-if="paths.length" :class="candidates.length ? 'mt-4' : ''">
      <h2 v-if="!candidates.length" class="lp-section mb-3">{{ t('learn.yourPaths') }}</h2>
      <p v-else class="mb-2 text-xs font-bold uppercase tracking-wide text-muted">{{ t('learn.learningNow') }}</p>
      <ul class="flex flex-wrap gap-2">
        <li v-for="p in paths" :key="p.subject_id">
          <RouterLink
            :to="{ name: 'learn', query: { subject: p.subject_id } }"
            class="lp-tap inline-flex min-h-[44px] items-center rounded-full bg-learn/15 px-4 text-sm font-semibold text-learn no-underline ring-1 ring-inset ring-learn/40"
            data-testid="learn-current"
            @click="track('learn_path', { action: 'open', kind: p.subject_kind, source: 'discover' })"
          >{{ t('learn.current', { subject: p.subject_label, n: p.day, total: p.steps.length }) }} ›</RouterLink>
        </li>
      </ul>
    </div>
  </section>
</template>
