<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { fetchCorpusManifest, type CorpusCostRollup } from '../../api/corpusMetricsApi'
import { useShellStore } from '../../stores/shell'

// Surfaces corpus_manifest.json cost_rollup so cost review is click-not-curl (incremental-add P2.8).
const shell = useShellStore()

const rollup = ref<CorpusCostRollup | null>(null)
const state = ref<'idle' | 'ok' | 'missing' | 'error'>('idle')

const showBlock = computed(() => Boolean(shell.healthStatus) && shell.hasCorpusPath)

function usd(v: number | undefined): string {
  return v == null ? '—' : `$${v.toFixed(2)}`
}

const total = computed(() => usd(rollup.value?.total_cost_usd))
const transcription = computed(() => usd(rollup.value?.total_transcription_cost_usd))
const llm = computed(() => usd(rollup.value?.total_llm_cost_usd))
const runCount = computed(() => rollup.value?.run_count ?? 0)
const uninstrumented = computed(() => rollup.value?.cost_appears_uninstrumented === true)

async function load(): Promise<void> {
  const root = shell.corpusPath.trim()
  if (!root) {
    return
  }
  try {
    const doc = await fetchCorpusManifest(root)
    if (doc.cost_rollup) {
      rollup.value = doc.cost_rollup
      state.value = 'ok'
    } else {
      state.value = 'missing'
    }
  } catch {
    state.value = 'error' // manifest absent (404) or unreadable
  }
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    if (showBlock.value) void load()
  },
  { immediate: true },
)
</script>

<template>
  <div
    v-if="showBlock"
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="cost-rollup-card"
  >
    <h3 class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
      Corpus cost
    </h3>
    <template v-if="state === 'ok'">
      <p class="text-sm">
        Total: <span class="font-medium" data-testid="cost-total">{{ total }}</span>
        <span class="ml-2 text-muted">over {{ runCount }} run{{ runCount === 1 ? '' : 's' }}</span>
      </p>
      <p class="mt-1 text-xs text-muted">
        Transcription {{ transcription }} · LLM {{ llm }}
      </p>
      <p
        v-if="uninstrumented"
        class="mt-2 text-xs text-warning"
        data-testid="cost-uninstrumented"
      >
        ⚠ Metrics present but cost recorded as $0 — the per-call cost path dropped data (not free).
      </p>
    </template>
    <p
      v-else-if="state === 'missing'"
      class="text-sm text-muted"
    >
      No cost_rollup in corpus_manifest.json yet.
    </p>
    <p
      v-else-if="state === 'error'"
      class="text-sm text-muted"
    >
      corpus_manifest.json not available for this corpus.
    </p>
    <p
      v-else
      class="text-sm text-muted"
    >
      Loading…
    </p>
  </div>
</template>
