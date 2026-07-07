<script setup lang="ts">
/**
 * Multi-perspective topics (#1146) — Dashboard card. Topics ranked by how many distinct
 * speakers have an attributed insight on them (cross-speaker discussion richness).
 * Clicking a topic opens its node view — where the Perspectives tab shows each take.
 */
import { ref, watch } from 'vue'

import { fetchTopicPerspectiveLeaders, type CilTopicPerspectiveLeader } from '../../api/cilApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

const shell = useShellStore()
const subject = useSubjectStore()

const topics = ref<CilTopicPerspectiveLeader[]>([])
const loading = ref(false)
const failed = ref(false)

watch(
  () => shell.corpusPath,
  () => {
    const root = shell.corpusPath.trim()
    topics.value = []
    failed.value = false
    if (!root) return
    loading.value = true
    fetchTopicPerspectiveLeaders(root, 10)
      .then((r) => {
        topics.value = r.topics
      })
      .catch(() => {
        // Distinguish a load failure from a genuinely empty corpus — otherwise a 503
        // reads to the operator as "no multi-perspective topics" (#1146 review M2).
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
  <section
    data-testid="dashboard-topic-perspectives"
    class="rounded-lg border border-slate-700 bg-slate-800/40 p-3"
  >
    <h3 class="text-sm font-semibold text-slate-100">Multi-perspective topics</h3>
    <p class="mb-2 text-xs text-slate-400">Where the most guests weigh in — click to see each take.</p>
    <p v-if="loading" class="text-xs text-slate-400">Loading…</p>
    <p v-else-if="failed" class="text-xs text-slate-400">Couldn't load perspectives.</p>
    <p v-else-if="!topics.length" class="text-xs text-slate-400">No multi-perspective topics yet.</p>
    <ul v-else role="list" class="flex flex-col gap-0.5">
      <li v-for="t in topics" :key="t.topic_id">
        <button
          type="button"
          data-testid="dashboard-topic-perspective-row"
          class="flex w-full items-center justify-between gap-2 rounded px-2 py-1 text-left text-sm text-slate-200 transition-colors hover:bg-slate-700/50"
          @click="subject.focusTopic(t.topic_id)"
        >
          <span class="min-w-0 truncate">{{ t.topic_label }}</span>
          <span class="shrink-0 text-xs text-slate-400">{{ t.speaker_count }} speakers</span>
        </button>
      </li>
    </ul>
  </section>
</template>
