<script setup lang="ts">
/**
 * Topic conversation arc (ADR-108) — the aggregate-first "shape" of a topic's conversation over
 * time on the topic card. A generic topic (e.g. "AI") can carry 1000s of insights; instead of a
 * flat list we show a compact row of weekly stacked bars (height = volume, colour = neg/neu/pos
 * sentiment mix) from GET /api/app/topics/{id}/conversation-arc. Self-fetches; renders nothing when
 * the topic has no dated insights.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import SectionStatus from './SectionStatus.vue'
import { useSectionState } from '../composables/useSectionState'
import { getTopicConversationArc } from '../services/api'
import type { TopicConversationArcWeek } from '../services/types'

const props = defineProps<{ id: string; scope?: 'all' | 'mine' }>()
const { t } = useI18n()

/**
 * `useSectionState` so a failed fetch is not indistinguishable from a topic with no dated
 * insights. This section previously caught into `[]` and hid itself when empty — #1591's defect,
 * fixed there for the Home sections but never carried to the topic card.
 */
const section = useSectionState<TopicConversationArcWeek[]>([])
const weeks = computed(() => section.data.value)
/** Drops a slow reply for a topic (or scope) the reader has already moved off. */
const requestSeq = ref(0)

async function load(): Promise<void> {
  // The arc is a corpus-wide aggregate with no per-user cut, so under "My corpus" it renders
  // nothing like the rest of the card rather than showing all-corpus data. That is a deliberate
  // empty, not a failure — so it resolves to `[]` through the ready path, never `error`.
  const mine = requestSeq.value + 1
  requestSeq.value = mine
  await section.load(async () => {
    if (props.scope === 'mine') return []
    const r = await getTopicConversationArc(props.id)
    if (mine !== requestSeq.value) throw new Error('superseded')
    return r.weeks
  })
}

watch([() => props.id, () => props.scope], () => void load(), { immediate: true })

const maxVolume = computed(() => Math.max(1, ...weeks.value.map((w) => w.volume)))
const totalInsights = computed(() => weeks.value.reduce((n, w) => n + w.volume, 0))

const SENT_CLASS: Record<'negative' | 'neutral' | 'positive', string> = {
  negative: 'bg-rose-500/70',
  neutral: 'bg-slate-400/50',
  positive: 'bg-emerald-500/70',
}
</script>

<template>
  <!-- Error says so and offers retry; a genuinely arc-less topic still renders nothing. No
       skeleton — this sits inside an already-loading card. -->
  <SectionStatus v-if="section.isError.value" :phase="section.phase.value" @retry="load()" />

  <section v-else-if="weeks.length" class="mb-4" data-testid="topic-conversation-arc">
    <div class="mb-2 flex items-baseline justify-between gap-2">
      <h3 class="lp-section">{{ t('ec.conversationArc') }}</h3>
      <span class="text-xs text-muted">
        {{ t('ec.convArcInsights', totalInsights, { named: { count: totalInsights } }) }}
      </span>
    </div>
    <div
      class="flex items-end gap-px overflow-x-auto rounded-lg border border-border bg-overlay p-2"
      style="height: 64px"
      data-testid="tca-bars"
    >
      <div
        v-for="w in weeks"
        :key="w.week"
        class="flex shrink-0 flex-col justify-end rounded-sm"
        style="width: 8px"
        :style="{ height: Math.round((w.volume / maxVolume) * 48) + 6 + 'px' }"
        :title="`${w.week} · ${w.volume} · ${w.negative} ${t('ec.convNeg')} / ${w.neutral} ${t('ec.convNeu')} / ${w.positive} ${t('ec.convPos')} · avg ${w.avg_compound.toFixed(2)}`"
        :data-testid="`tca-bar-${w.week}`"
      >
        <span
          v-if="w.positive"
          class="w-full"
          :class="SENT_CLASS.positive"
          :style="{ height: (w.positive / w.volume) * 100 + '%' }"
        />
        <span
          v-if="w.neutral"
          class="w-full"
          :class="SENT_CLASS.neutral"
          :style="{ height: (w.neutral / w.volume) * 100 + '%' }"
        />
        <span
          v-if="w.negative"
          class="w-full"
          :class="SENT_CLASS.negative"
          :style="{ height: (w.negative / w.volume) * 100 + '%' }"
        />
      </div>
    </div>
    <div class="mt-1 flex items-center gap-3 text-[10px] text-muted">
      <span class="inline-flex items-center gap-1"><span class="inline-block h-2 w-2 rounded-sm bg-rose-500/70" />{{ t('ec.convNeg') }}</span>
      <span class="inline-flex items-center gap-1"><span class="inline-block h-2 w-2 rounded-sm bg-slate-400/50" />{{ t('ec.convNeu') }}</span>
      <span class="inline-flex items-center gap-1"><span class="inline-block h-2 w-2 rounded-sm bg-emerald-500/70" />{{ t('ec.convPos') }}</span>
    </div>
  </section>
</template>
