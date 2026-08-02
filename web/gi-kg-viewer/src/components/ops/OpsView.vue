<script setup lang="ts">
import { onMounted, ref } from 'vue'

import {
  fetchLlmGateway,
  fetchOpsSummary,
  fetchResilience,
  fetchUsage,
  resetResilience,
  type LlmGatewaySnapshot,
  type OpsSourceEnvelope,
  type OpsSummary,
  type ResilienceSnapshot,
  type UsageSnapshot,
} from '../../api/opsApi'

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
  'traces',
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
    case 'traces':
      return `${num(d.count)} recent traces`
    default:
      return 'ok'
  }
}

// --- Resilience (ADR-113): open breakers + fuse budgets, with an operator reset ---
const resilience = ref<ResilienceSnapshot | null>(null)
const resilienceError = ref<string | null>(null)
const resetting = ref(false)

const rssOpenFeeds = (): string[] => {
  const feeds = (resilience.value?.rss?.circuit_breaker_open_feeds as unknown) ?? []
  return Array.isArray(feeds) ? (feeds as string[]) : []
}

async function refreshResilience(): Promise<void> {
  try {
    resilience.value = await fetchResilience()
    resilienceError.value = null
  } catch (e) {
    resilienceError.value = e instanceof Error ? e.message : String(e)
  }
}

async function onResetBreakers(): Promise<void> {
  resetting.value = true
  try {
    await resetResilience('all')
    await refreshResilience()
  } catch (e) {
    resilienceError.value = e instanceof Error ? e.message : String(e)
  } finally {
    resetting.value = false
  }
}

const usage = ref<UsageSnapshot | null>(null)
const usageError = ref<string | null>(null)
const usageGroupBy = ref<'provider,model' | 'operation' | 'episode_id' | 'model'>('provider,model')

const usdFmt = (n: number): string => `$${(n ?? 0).toFixed(4)}`
const tokFmt = (n: number): string => (n ?? 0).toLocaleString()
const usageLabel = (g: Record<string, string | number>): string =>
  usageGroupBy.value
    .split(',')
    .map((d) => String(g[d] ?? '(none)'))
    .join(' · ')

async function refreshUsage(): Promise<void> {
  try {
    usage.value = await fetchUsage(usageGroupBy.value)
    usageError.value = null
  } catch (e) {
    usageError.value = e instanceof Error ? e.message : String(e)
  }
}

async function onUsageGroupByChange(dim: typeof usageGroupBy.value): Promise<void> {
  usageGroupBy.value = dim
  await refreshUsage()
}

// --- Prod LiteLLM gateway (#53 / ADR-142): per-key spend/budget/burn from VictoriaMetrics ---
const llmGateway = ref<LlmGatewaySnapshot | null>(null)
const llmGatewayError = ref<string | null>(null)

// Spend runs from sub-cent (a few test calls) to dollars — show precision on tiny values.
const spendFmt = (n: number | undefined): string =>
  n == null ? '—' : `$${n > 0 && n < 0.01 ? n.toPrecision(2) : n.toFixed(2)}`
const burnFmt = (n: number | undefined): string => (n == null ? '—' : `${(n * 100).toFixed(1)}%`)

async function refreshLlmGateway(): Promise<void> {
  try {
    llmGateway.value = await fetchLlmGateway()
    llmGatewayError.value = null
  } catch (e) {
    llmGatewayError.value = e instanceof Error ? e.message : String(e)
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
  void refreshResilience()
  void refreshUsage()
  void refreshLlmGateway()
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

    <!-- Prod LiteLLM gateway (#53 / ADR-142): per-key spend + the budget wall, from the
         gateway's metered spend pushed to homelab VictoriaMetrics. -->
    <div class="rounded-sm border border-border bg-elevated p-3" data-testid="llm-gateway-panel">
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold text-surface-foreground">Prod LLM gateway</span>
        <span
          class="text-xs font-semibold"
          :class="{
            'text-success': llmGateway?.configured && llmGateway?.reachable,
            'text-muted': !llmGateway || !llmGateway.configured,
            'text-danger': llmGateway?.configured && !llmGateway?.reachable,
          }"
          data-testid="llm-gateway-status"
        >
          {{
            !llmGateway
              ? '—'
              : !llmGateway.configured
                ? 'not configured'
                : llmGateway.reachable
                  ? 'live'
                  : 'unreachable'
          }}
        </span>
      </div>
      <p v-if="llmGatewayError" class="mt-1 text-xs text-danger" data-testid="llm-gateway-error">
        {{ llmGatewayError }}
      </p>
      <p
        v-else-if="llmGateway && !llmGateway.configured"
        class="mt-1 text-xs text-muted"
      >
        VictoriaMetrics not wired for this deploy.
      </p>
      <p
        v-else-if="llmGateway && llmGateway.keys.length === 0"
        class="mt-1 text-xs text-muted"
      >
        No per-key spend recorded yet.
      </p>
      <table
        v-else-if="llmGateway"
        class="mt-2 w-full text-xs"
        data-testid="llm-gateway-keys"
      >
        <thead>
          <tr class="text-left text-muted">
            <th class="font-medium">key</th>
            <th class="font-medium text-right">spend</th>
            <th class="font-medium text-right">budget</th>
            <th class="font-medium text-right">burn</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="k in llmGateway.keys"
            :key="k.key_alias"
            :data-testid="`llm-key-${k.key_alias}`"
          >
            <td class="py-0.5 text-surface-foreground">{{ k.key_alias }}</td>
            <td class="py-0.5 text-right tabular-nums">{{ spendFmt(k.spend_usd) }}</td>
            <td class="py-0.5 text-right tabular-nums text-muted">{{ spendFmt(k.max_budget_usd) }}</td>
            <td
              class="py-0.5 text-right tabular-nums"
              :class="{ 'text-danger': (k.burn_ratio ?? 0) >= 0.9 }"
            >
              {{ burnFmt(k.burn_ratio) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Resilience (ADR-113): circuit breakers + call-fuse budgets, with an operator reset. -->
    <div class="rounded-sm border border-border bg-elevated p-3" data-testid="resilience-panel">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-surface-foreground">Resilience</span>
          <span
            class="text-xs font-semibold"
            :class="resilience?.any_open ? 'text-danger' : 'text-success'"
            data-testid="resilience-status"
          >
            {{ resilience?.any_open ? 'backing off' : 'all clear' }}
          </span>
        </div>
        <button
          v-if="resilience?.any_open"
          type="button"
          class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-50"
          data-testid="resilience-reset"
          :disabled="resetting"
          title="Force-close open breakers early instead of waiting out the cooldown"
          @click="onResetBreakers"
        >
          {{ resetting ? 'Resetting…' : 'Reset breakers' }}
        </button>
      </div>

      <p v-if="resilienceError" class="mt-1 text-xs text-danger" data-testid="resilience-error">
        {{ resilienceError }}
      </p>

      <div v-if="resilience" class="mt-2 space-y-2">
        <!-- LLM breakers: a chip per provider; open ones are highlighted with their cooldown. -->
        <div class="flex flex-wrap gap-1">
          <span
            v-for="(state, provider) in resilience.llm_breakers"
            :key="provider"
            class="rounded px-1.5 py-0.5 text-xs"
            :class="
              state.open
                ? 'bg-danger/10 text-danger'
                : 'bg-overlay text-muted'
            "
            :data-testid="`resilience-breaker-${provider}`"
            :title="`trips: ${state.trips_total} · recent failures: ${state.recent_failures}`"
          >
            {{ provider }}<template v-if="state.open">
              · {{ Math.ceil(state.cooldown_remaining_seconds) }}s</template>
          </span>
        </div>

        <p v-if="rssOpenFeeds().length" class="text-xs text-danger" data-testid="resilience-rss-open">
          RSS breaker open for: {{ rssOpenFeeds().join(', ') }}
        </p>

        <p class="text-xs text-muted" data-testid="resilience-fuses">
          Call fuse:
          {{ resilience.fuses.llm_max_calls_per_episode ?? '—' }}/episode ·
          {{ resilience.fuses.llm_max_calls_per_run ?? '—' }}/run
          <span class="opacity-70">(per-run hard stop; not resettable — fix &amp; rerun)</span>
        </p>
      </div>
    </div>

    <!-- LLM token/cost usage rollup — tokens are ground truth, cost derived; sliceable + de-duped. -->
    <div class="rounded-sm border border-border bg-elevated p-3" data-testid="usage-panel">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-semibold">LLM token &amp; cost usage</h3>
        <div class="flex gap-1" data-testid="usage-groupby">
          <button
            v-for="dim in ['provider,model', 'model', 'operation', 'episode_id']"
            :key="dim"
            class="rounded px-1.5 py-0.5 text-xs"
            :class="usageGroupBy === dim ? 'bg-accent text-on-accent' : 'bg-overlay text-muted'"
            :data-testid="`usage-groupby-${dim}`"
            @click="onUsageGroupByChange(dim as typeof usageGroupBy)"
          >
            {{ dim === 'provider,model' ? 'provider·model' : dim }}
          </button>
        </div>
      </div>

      <p v-if="usageError" class="mt-1 text-xs text-danger" data-testid="usage-error">
        {{ usageError }}
      </p>

      <div v-else-if="usage && usage.uninstrumented" class="mt-2 text-xs text-danger" data-testid="usage-uninstrumented">
        Telemetry files found but no token events recorded — cost is unknown, not zero.
      </div>

      <div v-else-if="usage" class="mt-2 space-y-2">
        <p class="text-xs text-muted" data-testid="usage-total">
          {{ usage.total.calls }} calls ·
          {{ tokFmt(usage.total.input_tokens) }} in ·
          {{ tokFmt(usage.total.output_tokens) }} out ·
          {{ tokFmt(usage.total.cached_input_tokens) }} cached ·
          <span class="font-semibold text-fg">{{ usdFmt(usage.total.estimated_cost_usd) }}</span>
        </p>
        <div class="overflow-x-auto">
          <table class="w-full text-xs" data-testid="usage-table">
            <thead>
              <tr class="text-left text-muted">
                <th class="pr-3">{{ usageGroupBy.replace(',', ' · ') }}</th>
                <th class="pr-3 text-right">calls</th>
                <th class="pr-3 text-right">in</th>
                <th class="pr-3 text-right">out</th>
                <th class="pr-3 text-right">cached</th>
                <th class="text-right">cost</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(g, i) in usage.groups.slice(0, 20)"
                :key="i"
                class="border-t border-border/50"
                data-testid="usage-row"
              >
                <td class="pr-3 py-0.5">{{ usageLabel(g) }}</td>
                <td class="pr-3 text-right">{{ g.calls }}</td>
                <td class="pr-3 text-right">{{ tokFmt(g.input_tokens) }}</td>
                <td class="pr-3 text-right">{{ tokFmt(g.output_tokens) }}</td>
                <td class="pr-3 text-right">{{ tokFmt(g.cached_input_tokens) }}</td>
                <td class="text-right">{{ usdFmt(g.estimated_cost_usd) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>
