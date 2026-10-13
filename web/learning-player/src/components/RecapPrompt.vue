<script setup lang="ts">
/**
 * A one-line look back, on Home (#1914 slice 5).
 *
 * Deliberately NOT the recap. The recap lives in Profile and stays there — that is the place you
 * can always go to look. This is the periodic prompt that reminds you it exists, because a recap
 * nobody sees is not a feature, and Profile is a screen people visit on purpose rather than by
 * accident.
 *
 * So it says the least it can while still being worth reading: how long, how many, and the one
 * thing that kept coming up. Everything else is one tap away.
 *
 * Self-hides when signed out, while loading, or when there is nothing recorded — the same
 * contract YourWeek has directly above it. A row that says "0h, 0 episodes" is worse than no row:
 * it takes space to tell you nothing happened, which you already knew.
 */
import { computed, onActivated, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getRecap } from '../services/api'
import { identityChangedSince, identityEpoch } from '../services/identity'
import type { RecapResponse } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { trendLabel } from '../utils/recapTrend'
import { formatListened } from '../utils/format'
import { kindPill } from '../utils/interests'
import KindWord from './KindWord.vue'

const { t } = useI18n()
const auth = useAuthStore()
const recap = ref<RecapResponse | null>(null)

async function load(): Promise<void> {
  if (!auth.isAuthenticated) return
  const generation = identityEpoch()
  const next = await getRecap('week')
  if (!identityChangedSince(generation)) recap.value = next
}
onMounted(load)
// Home is kept alive: without these the first account's week stayed on Home for every account
// signed in after it on the same device (operator 2026-10-07), and never refreshed on return.
onActivated(load)
watch(
  () => auth.user?.user_id ?? null,
  (uid, previous) => {
    if (uid === previous) return
    recap.value = null
    if (uid) void load()
  },
)

const listened = computed(() => formatListened(recap.value?.listening_seconds))
/**
 * The week's other two ways of listening, said only when they happened (operator 2026-10-12):
 * the line is Profile's four numbers in one sentence, so the two places say the same week.
 */
const extras = computed(() => {
  const r = recap.value
  const out: string[] = []
  if (r?.moments_heard) out.push(t('recap.promptMoments', r.moments_heard, { named: { count: r.moments_heard } }))
  if (r?.learning_days) out.push(t('recap.promptLearning', r.learning_days, { named: { count: r.learning_days } }))
  return out
})
/**
 * What the week was about — topics, people, and the subjects of its short listens and learning
 * days (operator 2026-10-12), the same list as Profile's: the top three as pills, each naming its
 * kind since the row mixes them. The numbers line no longer carries a headline subject — with
 * moments and learning days it ran off the card ("2 learnin…").
 *
 * Movement is the reason the row is worth its height: the same labels every week say nothing,
 * "up two" says what changed.
 */
const moreTopics = computed(() =>
  (recap.value?.subjects ?? []).slice(0, 3).map((s) => ({
    key: `${s.kind}:${s.id}`,
    kind: s.kind,
    label: s.label,
    trend: trendLabel(s, t('recap.new')),
    isNew: s.is_new,
  })),
)
/** Nothing heard at all — no listening, no moment, no learning day — means nothing to look back on. */
const worthShowing = computed(
  () =>
    (recap.value?.listening_seconds ?? 0) > 0 ||
    (recap.value?.moments_heard ?? 0) > 0 ||
    (recap.value?.learning_days ?? 0) > 0,
)
/**
 * The card is ONE link, so its accessible name is everything inside it — which now includes three
 * pills and their NEW/↑2 badges, announced as one run-on string before the user can act.
 *
 * Composed here instead: same facts, ordered as a sentence, with the movement markers dropped. A
 * screen-reader user gets "up two" from the recap itself; what this control needs to convey is
 * where it goes and why it is worth going.
 */
const linkLabel = computed(() => {
  const parts = [
    t('recap.promptTitle'),
    listened.value,
    t('recap.promptEpisodes', recap.value?.distinct_episodes ?? 0, {
      named: { count: recap.value?.distinct_episodes ?? 0 },
    }),
    ...extras.value,
    ...moreTopics.value.map((topic) => topic.label),
  ]
  return parts.join(', ')
})
/** The tallest day reaches this share of the card: a backdrop, under the words, not over them. */
const BAR_MAX = 0.45
const dayBars = computed<number[]>(() => {
  const days = Object.keys(recap.value?.by_day ?? {}).sort()
  // Both kinds of listening, as on Profile's chart.
  const values = days.map(
    (d) => (recap.value?.by_day[d] ?? 0) + (recap.value?.moments_by_day?.[d] ?? 0),
  )
  const peak = Math.max(...values, 0)
  // A shape needs two days (operator 2026-10-12): with one day heard, that day was the peak, drawn
  // the card's full height — a dark block down the right side rather than a week.
  if (!peak || values.filter((v) => v > 0).length < 2) return []
  return values.map((v) => Math.max(0.04, v / peak))
})
</script>

<template>
  <RouterLink
    v-if="worthShowing"
    :to="{ name: 'profile', query: { tab: 'stats' } }"
    :aria-label="linkLabel"
    class="lp-recap-prompt relative mt-7 flex items-center justify-between gap-3 overflow-hidden rounded-2xl border border-border bg-surface px-4 py-3 transition hover:border-accent"
  >
    <!-- The week, behind the words. Bars are bottom-anchored and run the full width of the card,
         fading out under the text so the line stays the thing you read. `aria-hidden`: it restates
         the hours the text already gives, and a screen reader does not need the silhouette. -->
    <div
      v-if="dayBars.length"
      class="pointer-events-none absolute inset-0 flex items-end gap-px opacity-[0.18]"
      aria-hidden="true"
    >
      <span
        v-for="(h, i) in dayBars"
        :key="i"
        class="flex-1 rounded-t-sm bg-accent"
        :style="{ height: `${Math.round(h * BAR_MAX * 100)}%` }"
      />
    </div>
    <div class="relative min-w-0">
      <span class="lp-kicker">{{ t('recap.promptTitle') }}</span>
      <p class="mt-1 text-sm">
        <span class="font-bold">{{ listened }}</span>
        <span class="text-muted"> · </span>
        <span>{{ t('recap.promptEpisodes', recap!.distinct_episodes, { named: { count: recap!.distinct_episodes } }) }}</span>
        <template v-for="e in extras" :key="e"><span class="text-muted"> · </span><span>{{ e }}</span></template>
      </p>
      <!-- Second row: the week's top subjects, as pills.
           Pills rather than more running text — at plain weight the marker ran into the label it
           annotates ("expert interviewsNEW") and the whole row read as one grey sentence. A bordered
           chip gives each topic an edge to end at, which is what was actually missing.
           They are TEXT, not links: the card is already a RouterLink and an anchor inside an anchor
           is invalid and breaks assistive tech. Tapping anywhere lands on the full recap, which is
           where a topic is followable. -->
      <div
        v-if="moreTopics.length"
        class="mt-2 flex flex-wrap items-center gap-1.5"
        data-testid="recap-prompt-topics"
      >
        <span
          v-for="topic in moreTopics"
          :key="topic.key"
          class="inline-flex max-w-full items-center gap-1.5 rounded-full py-1 pl-2.5 pr-1.5 text-xs"
          :class="kindPill(topic.kind)"
          data-testid="recap-prompt-subject"
        >
          <!-- In its kind's colours, naming its kind (operator 2026-10-12): the row mixes topics,
               people, themes and storylines now. -->
          <span class="truncate"><KindWord :kind="topic.kind" />{{ topic.label }}</span>
          <!-- NEW is a filled accent badge; a delta is a quieter outlined one. "Appeared this week"
               and "moved two places" are different kinds of news and should not look identical. -->
          <span
            v-if="topic.trend"
            class="shrink-0 rounded-full px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase leading-none tracking-wide"
            :class="
              topic.isNew
                ? 'bg-accent text-canvas'
                : topic.trend.startsWith('↓')
                  ? 'bg-overlay text-muted'
                  : 'bg-overlay text-canvas-foreground'
            "
            >{{ topic.trend }}</span
          >
        </span>
      </div>
    </div>
    <span class="relative shrink-0 text-sm font-medium text-muted" aria-hidden="true">→</span>
  </RouterLink>
</template>
