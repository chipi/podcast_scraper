<script setup lang="ts">
/**
 * "Your listening" — the recap panel in Profile (#1914).
 *
 * This REPLACES a number that was not true. The tile here used to read `listening_seconds / 3600`
 * from `/me/stats`, which is `sum(position_seconds)` — a lifetime snapshot of furthest position
 * reached. It goes up when you seek forward without hearing anything, does not move when you
 * re-listen, and cannot be windowed at all. The figure shown now is time actually accrued,
 * recorded per save and clamped so a seek cannot inflate it.
 *
 * Because that recording only started with Phase 0, the panel ALWAYS states its coverage
 * ("recorded 5 of 7 days"). Showing a partial total silently would repeat the original sin in a
 * new form; the caveat costs one line and disappears on its own as the window fills in.
 *
 * Renders nothing at all when there is no recording yet — an empty recap is worse than no recap.
 */
import { computed, onActivated, onMounted, ref, watch } from 'vue'
import { track } from '../services/analytics'
import { identityChangedSince, identityEpoch } from '../services/identity'
import { useAuthStore } from '../stores/auth'
import { useI18n } from 'vue-i18n'
import PlayFrom from './PlayFrom.vue'
import { getRecap } from '../services/api'
import type { RecapResponse, RecapSubject, RecapWindow } from '../services/types'
import { trendLabel } from '../utils/recapTrend'
import TranslatedFrom from './TranslatedFrom.vue'
import { formatListened } from '../utils/format'
import { kindPill } from '../utils/interests'
import KindWord from './KindWord.vue'
import Tabs from './Tabs.vue'
import type { TabSpec } from './tabs'
import type { UserStats } from '../services/types'

const { t } = useI18n()
/** The all-time half the window cannot say: the current streaks (operator 2026-10-12). */
const props = defineProps<{ stats?: UserStats | null }>()

// `ytd` rather than `year`: a rolling 365 days is not a thing anyone thinks about, and "your 2026
// so far" is honest TODAY in a way a complete-year recap cannot be until the year has been lived.
const WINDOWS: RecapWindow[] = ['week', 'month', 'ytd']
const windowTabs = computed<TabSpec<RecapWindow>[]>(() => WINDOWS.map((w) => ({ key: w, label: t(`recap.${w}`) })))
const window_ = ref<RecapWindow>('week')
const recap = ref<RecapResponse | null>(null)
const loading = ref(true)

/**
 * Fetch the recap. Never rejects.
 *
 * `onMounted(load)` and `watch(window_, load)` both call this fire-and-forget, so a rejection had
 * nowhere to go and became an unhandled rejection — on Profile, with no network, every time. It
 * also left `loading` true forever, so the panel sat on a skeleton rather than saying anything.
 *
 * A failed refresh keeps whatever is already on screen (the arc's rule); only a first load with
 * nothing to show reports that it could not fetch.
 */
const failed = ref(false)
async function load(): Promise<void> {
  loading.value = true
  const generation = identityEpoch()
  try {
    const next = await getRecap(window_.value)
    // A recap that resolves after an account switch belongs to the previous account.
    if (identityChangedSince(generation)) return
    recap.value = next
    failed.value = false
  } catch {
    failed.value = !recap.value
  } finally {
    loading.value = false
  }
}
onMounted(load)
// Profile is kept alive, so mount happens once per app session. Without these two, the recap of
// whoever was signed in first stayed on screen for every account after them (operator 2026-10-07:
// a brand-new Apple account showed the Google account's half hour, six episodes and saved line),
// and a returning listener saw the recap as it was when Profile first opened.
onActivated(load)
const auth = useAuthStore()
watch(
  () => auth.user?.user_id ?? null,
  (uid, previous) => {
    if (uid === previous) return
    recap.value = null
    failed.value = false
    reportedRecap.value = false
    if (uid) void load()
  },
)
watch(window_, load)

/**
 * `recap_view` (#2267) — reported when a recap actually RENDERS, not when this mounts.
 *
 * The component mounts and then fetches, and a listener with nothing to recap gets an empty state.
 * Reporting on mount would count those as recap views and make the learning-actions figure larger
 * than the behaviour behind it. Watching the loaded value counts only the real ones, once each.
 */
const reportedRecap = ref(false)
watch(
  recap,
  (r) => {
    if (!r || reportedRecap.value) return
    reportedRecap.value = true
    track('recap_view', { trigger: 'panel' })
  },
  { immediate: true },
)

/**
 * One day chart for both kinds of listening (operator 2026-10-12): full episodes and short listens
 * stacked, since a short listen is not in `listening_seconds` and a day of moments was a blank bar.
 */
const bars = computed(() =>
  Object.entries(recap.value?.by_day ?? {}).map(([day, full]) => {
    const moments = recap.value?.moments_by_day?.[day] ?? 0
    return { day, full, moments, total: full + moments }
  }),
)
const peak = computed(() => Math.max(1, ...bars.value.map((b) => b.total)))
const listened = computed(() => formatListened(recap.value?.listening_seconds))
const momentsTime = computed(() => formatListened(recap.value?.moments_seconds))
/** Nothing recorded yet means nothing honest to say. */
const hasAnything = computed(
  () =>
    !!recap.value &&
    (recap.value.days_recorded > 0 ||
      recap.value.episodes_started > 0 ||
      (recap.value.moments_heard ?? 0) > 0 ||
      (recap.value.learning_days ?? 0) > 0),
)
/** Only worth saying while it is actually partial. */
const partial = computed(
  () => !!recap.value && recap.value.days_recorded < recap.value.days_in_window,
)
/** "3 shows · 2 finished" — the finishes only when there were any. */
const episodesNote = computed(() => {
  const r = recap.value
  const shows = t('recap.shows', r?.distinct_shows ?? 0, { named: { count: r?.distinct_shows ?? 0 } })
  return r?.episodes_finished
    ? t('recap.episodesNote', { shows, finished: r.episodes_finished })
    : t('recap.episodesShowsOnly', { shows })
})
/** The streaks, said once, only those that are running. */
const streaks = computed(() => {
  const parts: string[] = []
  const day = props.stats?.day_streak ?? 0
  const learn = props.stats?.learning_streak ?? 0
  if (day) parts.push(t('stats.streakLine', day, { named: { count: day } }))
  if (learn) parts.push(t('stats.learningStreakLine', learn, { named: { count: learn } }))
  return parts.join(' · ')
})
/** What the window was about: one mixed row, so each pill names its kind. */
const subjects = computed<RecapSubject[]>(() => recap.value?.subjects ?? [])
function subjectTotal(s: RecapSubject): number {
  return s.episodes + s.moments + s.learning_days
}
/** The pill's spoken name says what the number is made of; the pill shows the total. */
function subjectName(s: RecapSubject): string {
  const parts = [
    s.episodes ? t('recap.subjectEpisodes', s.episodes, { named: { count: s.episodes } }) : '',
    s.moments ? t('recap.subjectMoments', s.moments, { named: { count: s.moments } }) : '',
    s.learning_days
      ? t('recap.subjectLearning', s.learning_days, { named: { count: s.learning_days } })
      : '',
  ].filter(Boolean)
  return `${t(`notes.kind_${s.kind}`)} ${s.label}: ${parts.join(', ')}`
}
/**
 * How a subject moved, or '' when there is nothing worth saying.
 *
 * "New" beats "+2" for something that was absent — it is a different fact, not a bigger number.
 * A delta of zero renders nothing at all: an arrow that means "unchanged" is noise on every chip
 * that did not move, which is most of them.
 */
function trendOf(subject: RecapSubject): string {
  return trendLabel(subject, t('recap.new'))
}
/** A saved line opens its episode AT the moment it came from, not at the beginning. */
const lineTarget = computed(() => {
  const line = recap.value?.best_line
  if (!line?.episode_slug) return null
  const t_ = line.start_ms != null ? Math.floor(line.start_ms / 1000) : null
  return { name: 'player', params: { slug: line.episode_slug }, query: t_ ? { t: String(t_) } : {} }
})
</script>

<template>
  <section
    v-if="!loading && hasAnything"
    class="rounded-2xl border border-border p-5"
    data-testid="stats-recap"
  >
    <!-- The window is the shared segmented control, as the settings' choices are, every label on
         one line (operator 2026-10-14); the title wraps above it rather than beside it. -->
    <div class="mb-4 flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
      <h2 class="lp-section whitespace-nowrap">{{ t('recap.title') }}</h2>
      <Tabs
        v-model="window_"
        :tabs="windowTabs"
        :label="t('recap.window')"
        id-prefix="recap-window"
        variant="segment"
        pattern="radio"
      />
    </div>
    <!-- Four numbers for the window, one per way of listening (operator 2026-10-12): full
         episodes, short listens, learning days — and what was heard, with its shows and finishes
         as the tile's second line rather than three more tiles. -->
    <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <div class="rounded-xl bg-overlay p-4" data-testid="recap-listened">
        <span class="font-display text-3xl font-extrabold leading-none">{{ listened }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.listened') }}</div>
      </div>
      <div class="rounded-xl bg-overlay p-4" data-testid="recap-moments">
        <span class="font-display text-3xl font-extrabold leading-none">{{ recap!.moments_heard ?? 0 }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.moments') }}</div>
        <div v-if="recap!.moments_heard" class="lp-kicker mt-1">{{ momentsTime }}</div>
      </div>
      <div class="rounded-xl bg-overlay p-4" data-testid="recap-learning">
        <span class="font-display text-3xl font-extrabold leading-none">{{ recap!.learning_days ?? 0 }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.learningDays') }}</div>
      </div>
      <div class="rounded-xl bg-overlay p-4" data-testid="recap-episodes">
        <span class="font-display text-3xl font-extrabold leading-none">{{ recap!.distinct_episodes }}</span>
        <div class="mt-2 text-xs font-medium text-muted">{{ t('recap.episodes') }}</div>
        <div class="lp-kicker mt-1">{{ episodesNote }}</div>
      </div>
    </div>
    <!-- The day series: bars, both kinds stacked, a legend only when there were moments. -->
    <div class="mt-3 rounded-xl bg-overlay p-4">
      <div class="flex h-16 items-end gap-1" role="img" :aria-label="t('recap.byDay')">
        <div
          v-for="b in bars"
          :key="b.day"
          class="flex flex-1 flex-col-reverse overflow-hidden rounded-sm"
          :style="{ height: `${Math.max(2, (b.total / peak) * 100)}%`, opacity: b.total > 0 ? 1 : 0.25 }"
          :title="`${b.day}: ${Math.round(b.full / 60)}m + ${Math.round(b.moments / 60)}m`"
          data-testid="recap-day"
        >
          <span class="block bg-muted" :style="{ height: `${b.total ? (b.full / b.total) * 100 : 100}%` }" />
          <span v-if="b.moments" class="block bg-accent" :style="{ height: `${(b.moments / b.total) * 100}%` }" />
        </div>
      </div>
      <p v-if="recap!.moments_heard" class="mt-2 flex gap-3 text-xs text-muted">
        <span><span class="mr-1 inline-block h-2 w-2 rounded-sm bg-muted" />{{ t('recap.legendEpisodes') }}</span>
        <span><span class="mr-1 inline-block h-2 w-2 rounded-sm bg-accent" />{{ t('recap.legendMoments') }}</span>
      </p>
      <!-- Always visible while partial: the number above is real, and this says how much of the
           window produced it. -->
      <p v-if="partial" class="mt-2 text-xs text-muted">
        {{ t('recap.coverage', { recorded: recap!.days_recorded, total: recap!.days_in_window }) }}
      </p>
    </div>
    <p v-if="streaks" class="mt-3 text-sm" data-testid="recap-streaks">{{ streaks }}</p>
    <p v-if="recap!.moment_episodes" class="mt-1 text-sm text-muted" data-testid="recap-moment-to-full">
      {{ t('recap.momentToFull', { n: recap!.moment_to_full ?? 0, total: recap!.moment_episodes }) }}
    </p>
    <div v-if="subjects.length" class="mt-4">
      <h3 class="lp-kicker mb-1">{{ t('recap.about') }}</h3>
      <p class="mb-2 text-xs text-muted">{{ t('recap.vsPrevious') }}</p>
      <!-- One row of everything the window was about — the topics and people of the episodes, the
           subjects of the short listens and of the learning days (operator 2026-10-12). Mixed
           kinds, so each pill names its kind; the number is the window's total for it, and its
           spoken name says what that total is made of. -->
      <ul class="flex flex-wrap gap-2">
        <li
          v-for="subject in subjects"
          :key="`${subject.kind}:${subject.id}`"
          class="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold"
          :class="kindPill(subject.kind)"
          :aria-label="subjectName(subject)"
          data-testid="recap-subject"
        >
          <KindWord :kind="subject.kind" />{{ subject.label
          }}<span class="ml-1.5 font-normal opacity-70">{{ subjectTotal(subject) }}</span
          ><span
            v-if="trendOf(subject)"
            class="ml-1.5 font-medium"
            :class="subject.delta > 0 ? 'text-canvas-foreground' : 'text-muted'"
          >{{ trendOf(subject) }}</span>
        </li>
      </ul>
    </div>
    <!-- The one part of this that is an artifact rather than a statistic. -->
    <figure v-if="recap!.best_line" class="mt-4 rounded-xl bg-overlay p-4">
      <h3 class="lp-kicker mb-2">{{ t('recap.line') }}</h3>
      <blockquote class="text-sm italic">“{{ recap!.best_line.quote_text }}”</blockquote>
      <TranslatedFrom :lang="recap!.best_line.translated_from" />
      <!-- The shared jump control (operator 2026-10-05): "▶ Play from 3:12", or the old "Open
           line" wording when the line has no time to name. -->
      <PlayFrom
        v-if="lineTarget"
        class="mt-2 inline-block"
        :seconds="recap!.best_line.start_ms != null ? recap!.best_line.start_ms / 1000 : null"
        :fallback="t('recap.openLine')"
        :to="lineTarget"
      />
    </figure>
  </section>
</template>
