<script setup lang="ts">
/**
 * How a grouping's member has moved over the grouping's own lifetime.
 *
 * A theme or storyline is not a static set — topics join it, carry it for a while, and drop out.
 * The member list said none of that: four words in a fixed order, identical whether a topic had
 * been there from the first episode or arrived last month.
 *
 * Renders NOTHING for `steady`, which is most members most of the time. A badge on every row is a
 * badge that says nothing; these earn attention by being rare.
 *
 * `new`/`growing` in the accent, `fading` in the app's cooling colour (`--lp-trend-cooling`, the
 * "↓ Cooling" pill's) and `gone` muted (operator 2026-10-15: fading sits above gone in the list and
 * needs a mark that tells the two apart — the warning token read as the accent in dark mode). Growing
 * and fading carry an arrow, so the direction reads without the colour. Neither is a value
 * judgement: a topic leaving a storyline is how a storyline moves on, not a failure.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { formatMonthYear } from '../utils/format'

const props = defineProps<{
  trend?: 'new' | 'growing' | 'steady' | 'fading' | 'gone'
  /** Dates behind the badge, so the title can say WHEN rather than only WHAT. */
  firstSeen?: string | null
  lastSeen?: string | null
}>()

const { t, locale } = useI18n()

const LABEL: Record<string, string> = {
  new: 'home.memberNew',
  growing: 'home.memberGrowing',
  fading: 'home.memberFading',
  gone: 'home.memberGone',
}

const show = computed(() => !!props.trend && props.trend !== 'steady')
const label = computed(() => (props.trend ? t(LABEL[props.trend] ?? '') : ''))
const ARROW: Record<string, string> = { growing: '↗', fading: '↘' }
const arrow = computed(() => (props.trend ? ARROW[props.trend] ?? '' : ''))

/** Accent for what is arriving, the cooling colour for what is leaving, muted for what has left. */
const tone = computed(() => {
  if (props.trend === 'new' || props.trend === 'growing') return 'text-accent ring-accent/30'
  if (props.trend === 'fading') return 'ring-current'
  return 'text-muted ring-border'
})
const toneStyle = computed(() => (props.trend === 'fading' ? { color: 'var(--lp-trend-cooling)' } : undefined))

/** A date a reader can read, not an ISO string. Falls back to the raw value if it will not parse. */
function human(iso?: string | null): string {
  return formatMonthYear(iso, locale.value) ?? ''
}

/**
 * The hover/long-press explanation. `new` is defined by when it STARTED, `gone` by when it last
 * appeared — each badge names the date that makes it true, rather than a generic tooltip.
 */
const title = computed(() => {
  if (props.trend === 'new' || props.trend === 'growing') {
    return props.firstSeen ? t('home.memberSince', { date: human(props.firstSeen) }) : label.value
  }
  return props.lastSeen ? t('home.memberUntil', { date: human(props.lastSeen) }) : label.value
})
</script>

<template>
  <span
    v-if="show"
    class="shrink-0 rounded-full px-1.5 py-px text-[0.625rem] font-bold uppercase tracking-wide ring-1 ring-inset"
    :class="tone"
    :style="toneStyle"
    :title="title"
    :data-trend="trend"
    data-testid="member-trend"
  >
    <span v-if="arrow" aria-hidden="true" class="mr-0.5">{{ arrow }}</span>{{ label }}
  </span>
</template>
