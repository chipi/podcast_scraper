<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { fetchOpsSummary, type OpsSourceEnvelope, type OpsSummary } from '../../api/opsApi'

// Render order — matches the control plane's source list (#803).
const SOURCE_ORDER = [
  'health',
  'version',
  'runs',
  'deploys',
  'cost',
  'logs',
  'errors',
  'alerts',
] as const

const data = ref<OpsSummary | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

type Bucket = 'live' | 'unconfigured' | 'failed'

function bucketOf(name: string): Bucket {
  const d = data.value
  if (!d) return 'failed'
  if (d.live.includes(name)) return 'live'
  if (d.unconfigured.includes(name)) return 'unconfigured'
  return 'failed'
}

const asRecord = (v: unknown): Record<string, unknown> =>
  v && typeof v === 'object' ? (v as Record<string, unknown>) : {}
const str = (v: unknown): string => (v == null ? '?' : String(v))
const num = (v: unknown): number => (typeof v === 'number' ? v : 0)

/** A compact, per-source one-liner — the "cool stat" for each card. */
function summaryLine(name: string, env: OpsSourceEnvelope | undefined): string {
  if (!env) return '—'
  if (!env.ok) return env.configured === false ? 'not configured' : (env.error ?? 'error')
  const d = asRecord(env.data)
  switch (name) {
    case 'health':
      return `status: ${str(d.status)}`
    case 'version':
      return `${str(d.code_version)} · corpus ${str(d.corpus_git_sha)}`
    case 'runs':
      return `${num(d.count)} recent runs`
    case 'deploys':
      return d.failure_rate == null
        ? `${num(d.count)} deploys`
        : `${num(d.count)} deploys · ${Math.round(num(d.failure_rate) * 100)}% fail`
    case 'cost':
      return d.estimated_cost_usd == null ? 'n/a' : `$${num(d.estimated_cost_usd).toFixed(4)} (24h)`
    case 'logs':
      return `${num(d.count)} error lines (1h)`
    case 'errors':
      return `${num(d.total_issues)} unresolved`
    case 'alerts':
      return `${num(d.firing)} firing / ${num(d.count)}`
    default:
      return 'ok'
  }
}

async function refresh(): Promise<void> {
  loading.value = true
  error.value = null
  try {
    data.value = await fetchOpsSummary()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  void refresh()
})
</script>

<template>
  <div class="space-y-3" data-testid="ops-view">
    <div class="flex items-center justify-between">
      <h2 class="text-sm font-semibold text-surface-foreground">Prod ops</h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-50"
        data-testid="ops-refresh"
        :disabled="loading"
        @click="refresh"
      >
        {{ loading ? 'Refreshing…' : 'Refresh' }}
      </button>
    </div>

    <p v-if="loading && !data" class="text-xs text-muted" data-testid="ops-loading">Loading…</p>
    <p v-if="error" class="text-xs text-danger" data-testid="ops-error">{{ error }}</p>

    <div v-if="data" class="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
      <div
        v-for="name in SOURCE_ORDER"
        :key="name"
        class="rounded-sm border border-border bg-elevated p-3"
        :data-testid="`ops-source-${name}`"
      >
        <div class="flex items-center justify-between">
          <span class="text-xs font-medium text-surface-foreground">{{ name }}</span>
          <span
            class="text-xs font-semibold"
            :class="{
              'text-success': bucketOf(name) === 'live',
              'text-muted': bucketOf(name) === 'unconfigured',
              'text-danger': bucketOf(name) === 'failed',
            }"
            :data-testid="`ops-status-${name}`"
          >
            {{ bucketOf(name) }}
          </span>
        </div>
        <p class="mt-1 text-xs text-muted">{{ summaryLine(name, data.sources[name]) }}</p>
      </div>
    </div>
  </div>
</template>
