<script setup lang="ts">
/**
 * Topic perspectives (#1146) — multi-perspective synthesis on the topic card.
 * Each guest who spoke on the topic, with their grounded insights. Self-fetches
 * from GET /api/app/topics/{id}/perspectives; renders nothing when the topic has
 * none. Speaker names tap through to their person card (the same `open` contract
 * the card's other people rows use).
 */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import { getTopicPerspectives } from '../services/api'
import type { TopicPerspective } from '../services/types'

const props = defineProps<{ id: string; scope?: 'all' | 'mine' }>()
const emit = defineEmits<{ (e: 'open', payload: { kind: 'person' | 'topic'; id: string }): void }>()

const { t } = useI18n()

const perspectives = ref<TopicPerspective[]>([])
watch(
  () => [props.id, props.scope] as const,
  () => {
    perspectives.value = []
    void getTopicPerspectives(props.id, props.scope)
      .then((r) => {
        perspectives.value = r.perspectives
      })
      .catch(() => {
        perspectives.value = []
      })
  },
  { immediate: true },
)

// Show up to PREVIEW insights per speaker; the rest sit behind a per-speaker toggle.
const PREVIEW = 3
const expanded = ref<Set<string>>(new Set())
function toggle(personId: string): void {
  const next = new Set(expanded.value)
  if (next.has(personId)) next.delete(personId)
  else next.add(personId)
  expanded.value = next
}
</script>

<template>
  <section v-if="perspectives.length" class="mb-4" data-testid="topic-perspectives">
    <h3 class="lp-section mb-2">
      {{ t('ec.perspectives', perspectives.length, { named: { count: perspectives.length } }) }}
    </h3>
    <ul class="flex flex-col gap-2.5">
      <li
        v-for="p in perspectives"
        :key="p.person_id"
        class="rounded-lg border border-border bg-overlay p-3"
        data-testid="topic-perspective"
      >
        <div class="flex items-baseline gap-2">
          <button
            type="button"
            class="text-sm font-bold text-person hover:underline"
            @click="emit('open', { kind: 'person', id: p.person_id })"
          >
            {{ p.person_name }}
          </button>
          <span class="lp-kicker">{{
            t('ec.perspectiveInsights', p.insight_count, { named: { count: p.insight_count } })
          }}</span>
        </div>
        <ul class="mt-1.5 flex flex-col gap-1">
          <li
            v-for="ins in expanded.has(p.person_id) ? p.insights : p.insights.slice(0, PREVIEW)"
            :key="ins.id"
            class="flex gap-1.5 text-sm text-canvas-foreground"
          >
            <span aria-hidden="true" class="text-muted">•</span>
            <span>{{ ins.text }}</span>
          </li>
        </ul>
        <button
          v-if="p.insights.length > PREVIEW"
          type="button"
          class="mt-1 text-xs font-semibold text-accent hover:underline"
          @click="toggle(p.person_id)"
        >
          {{
            expanded.has(p.person_id)
              ? t('ec.perspectiveLess')
              : t('ec.perspectiveMore', { count: p.insights.length - PREVIEW })
          }}
        </button>
      </li>
    </ul>
  </section>
</template>
