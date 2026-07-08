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

import { getTopicConversationArc } from '../services/api'
import type { TopicConversationArcWeek } from '../services/types'

const props = defineProps<{ id: string }>()
const { t } = useI18n()

const weeks = ref<TopicConversationArcWeek[]>([])
watch(
  () => props.id,
  () => {
    weeks.value = []
    void getTopicConversationArc(props.id)
      .then((r) => {
        weeks.value = r.weeks
      })
      .catch(() => {
        weeks.value = []
      })
  },
  { immediate: true },
)

const maxVolume = computed(() => Math.max(1, ...weeks.value.map((w) => w.volume)))
const totalInsights = computed(() => weeks.value.reduce((n, w) => n + w.volume, 0))

const SENT_CLASS: Record<'negative' | 'neutral' | 'positive', string> = {
  negative: 'bg-rose-500/70',
  neutral: 'bg-slate-400/50',
  positive: 'bg-emerald-500/70',
}
</script>

<template>
  <section v-if="weeks.length" class="mb-4" data-testid="topic-conversation-arc">
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
