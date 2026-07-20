<script setup lang="ts">
/**
 * Topic perspectives (#1146) — operator NodeDetail rail panel. Each speaker's grounded
 * insights on the focused topic, grouped by speaker (most-insights first). Self-fetches
 * the operator route GET /api/topics/{id}/perspectives; shows a graceful empty state
 * when the topic has no speaker-attributed insight. Each speaker previews PREVIEW
 * insights with a per-speaker "+N more" toggle so nothing is silently dropped (parity
 * with the consumer TopicPerspectives.vue).
 */
import { reactive, ref, watch } from 'vue'

import { fetchTopicPerspectives, type CilTopicPerspective } from '../../api/cilApi'

const props = defineProps<{ corpusPath: string; topicId: string }>()

const PREVIEW = 4

const perspectives = ref<CilTopicPerspective[]>([])
const loading = ref(false)
const failed = ref(false)
const expanded = reactive(new Set<string>())

function insightText(node: Record<string, unknown>): string {
  const p = (node.properties ?? {}) as Record<string, unknown>
  return String(p.text ?? p.title ?? '')
}

function insightKey(node: Record<string, unknown>, i: number): string {
  return String((node.id as string | undefined) ?? i)
}

function visibleInsights(p: CilTopicPerspective) {
  return expanded.has(p.person_id) ? p.insights : p.insights.slice(0, PREVIEW)
}

function toggle(personId: string): void {
  if (expanded.has(personId)) expanded.delete(personId)
  else expanded.add(personId)
}

watch(
  () => [props.corpusPath, props.topicId] as const,
  () => {
    const path = props.corpusPath.trim()
    const tid = props.topicId.trim()
    perspectives.value = []
    failed.value = false
    expanded.clear()
    if (!path || !tid) return
    loading.value = true
    fetchTopicPerspectives(path, tid)
      .then((r) => {
        perspectives.value = Array.isArray(r.perspectives) ? r.perspectives : []
      })
      .catch(() => {
        failed.value = true
      })
      .finally(() => {
        loading.value = false
      })
  },
  { immediate: true },
)
</script>

<template>
  <div data-testid="node-topic-perspectives" class="flex flex-col gap-2 text-sm">
    <p v-if="loading" class="text-slate-400">Loading perspectives…</p>
    <p v-else-if="failed" class="text-slate-400">Couldn't load perspectives.</p>
    <p v-else-if="!perspectives.length" class="text-slate-400">
      No attributed perspectives for this topic.
    </p>
    <ul v-else role="list" class="flex flex-col gap-2">
      <li
        v-for="p in perspectives"
        :key="p.person_id"
        data-testid="node-topic-perspective"
        class="rounded border border-slate-700 bg-slate-800/50 p-2"
      >
        <div class="mb-1 flex items-baseline justify-between gap-2">
          <span class="font-semibold text-slate-100">{{ p.person_name }}</span>
          <span class="shrink-0 text-xs text-slate-400">
            {{ p.insight_count }} insights · {{ p.episode_count }} ep
          </span>
        </div>
        <ul role="list" class="flex flex-col gap-0.5">
          <li
            v-for="(ins, i) in visibleInsights(p)"
            :key="insightKey(ins, i)"
            class="flex gap-1.5 text-slate-300"
          >
            <span aria-hidden="true" class="text-slate-500">•</span>
            <span>{{ insightText(ins) }}</span>
          </li>
        </ul>
        <button
          v-if="p.insights.length > PREVIEW"
          type="button"
          data-testid="node-topic-perspective-toggle"
          class="mt-1 text-xs text-sky-400 hover:text-sky-300"
          @click="toggle(p.person_id)"
        >
          {{ expanded.has(p.person_id) ? 'Show fewer' : `+${p.insights.length - PREVIEW} more` }}
        </button>
      </li>
    </ul>
  </div>
</template>
