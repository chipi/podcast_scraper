<script setup lang="ts">
import { ref, watch } from 'vue'
import {
  createMcpToken,
  getMcpConfig,
  listMcpConnections,
  listMcpTokens,
  revokeMcpConnection,
  revokeMcpToken,
  type McpConnection,
  type McpConnectionConfig,
  type McpTokenMeta,
} from '../../api/mcpApi'
import { StaleGeneration } from '../../utils/staleGeneration'

/**
 * "Connected agents" (RFC-112 §5) — the operator's MCP surface inside the shell Configuration
 * dialog. Mirrors the player's Profile section: the connector URL (for per-user OAuth clients)
 * plus personal-access token CRUD (for CLI clients). Every call is `mcp_access`-gated server-side;
 * the parent only renders this tab when the signed-in operator holds the entitlement.
 *
 * Loads lazily when its tab becomes `active` (same pattern as ScheduledJobsSection).
 */
const props = withDefaults(defineProps<{ active: boolean }>(), { active: false })

const config = ref<McpConnectionConfig | null>(null)
const connections = ref<McpConnection[]>([])
const tokens = ref<McpTokenMeta[]>([])
const newLabel = ref('')
const freshSecret = ref<string | null>(null)
const copied = ref<'url' | 'secret' | null>(null)
const busy = ref(false)
const error = ref<string | null>(null)
const loaded = ref(false)
const gate = new StaleGeneration()

async function load(): Promise<void> {
  const seq = gate.bump()
  busy.value = true
  error.value = null
  try {
    const [cfg, conns, toks] = await Promise.all([
      getMcpConfig(),
      listMcpConnections(),
      listMcpTokens(),
    ])
    if (gate.isStale(seq)) return
    config.value = cfg
    connections.value = conns
    tokens.value = toks
    loaded.value = true
  } catch (e) {
    if (gate.isCurrent(seq)) error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gate.isCurrent(seq)) busy.value = false
  }
}

async function revokeConnection(clientId: string): Promise<void> {
  if (busy.value) return
  busy.value = true
  error.value = null
  try {
    connections.value = await revokeMcpConnection(clientId)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function copy(text: string, which: 'url' | 'secret'): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    copied.value = which
    setTimeout(() => (copied.value = copied.value === which ? null : copied.value), 1500)
  } catch {
    /* clipboard denied — the value stays visible for a manual copy */
  }
}

async function create(): Promise<void> {
  const label = newLabel.value.trim()
  if (!label || busy.value) return
  busy.value = true
  error.value = null
  try {
    const created = await createMcpToken(label)
    freshSecret.value = created.token // shown ONCE
    tokens.value = [created.meta, ...tokens.value]
    newLabel.value = ''
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function revoke(id: string): Promise<void> {
  if (busy.value) return
  busy.value = true
  error.value = null
  try {
    tokens.value = await revokeMcpToken(id)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

function fmt(ts: number | null): string {
  return ts ? new Date(ts * 1000).toLocaleDateString() : ''
}

// Lazy-load on first activation.
watch(
  () => props.active,
  (active) => {
    if (active && !loaded.value) void load()
  },
  { immediate: true },
)
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto text-canvas-foreground">
    <div>
      <h3 class="text-sm font-medium text-surface-foreground">Connected agents</h3>
      <p class="mt-0.5 text-[11px] leading-snug text-muted">
        Let an AI agent (Claude, Cursor, an MCP client) search and read the corpus as you, over MCP.
      </p>
    </div>

    <p
      v-if="error"
      class="rounded border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-danger"
      data-testid="agents-error"
    >
      {{ error }}
    </p>

    <!-- Connector URL (per-user OAuth clients) -->
    <div v-if="config?.connector_url" class="rounded border border-border bg-surface p-3">
      <div class="mb-1 text-xs font-medium text-surface-foreground">Connector URL</div>
      <div class="flex items-center gap-2">
        <code
          class="min-w-0 flex-1 truncate rounded bg-canvas px-2 py-1 font-mono text-[11px]"
          :title="config.connector_url"
        >{{ config.connector_url }}</code>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay"
          @click="copy(config.connector_url, 'url')"
        >{{ copied === 'url' ? 'Copied' : 'Copy' }}</button>
      </div>
      <p class="mt-1 text-[11px] text-muted">
        Add as a custom connector in a client that supports per-user sign-in — approve access once.
      </p>
    </div>
    <p v-else-if="loaded" class="text-[11px] text-muted">
      The remote connector isn't configured on this deployment yet.
    </p>

    <!-- Connected OAuth apps (claude.ai etc.) — revocable -->
    <div v-if="connections.length" class="rounded border border-border bg-surface p-3">
      <div class="text-xs font-medium text-surface-foreground">Connected apps</div>
      <ul class="mt-2 divide-y divide-border">
        <li
          v-for="c in connections"
          :key="c.client_id"
          class="flex items-center justify-between gap-3 py-1.5"
        >
          <div class="min-w-0">
            <div class="truncate text-sm text-canvas-foreground">{{ c.client_name }}</div>
            <div class="text-[11px] text-muted">Access: {{ c.scopes.join(', ') }}</div>
          </div>
          <button
            type="button"
            class="shrink-0 rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay hover:text-danger disabled:opacity-50"
            :disabled="busy"
            data-testid="agents-connection-revoke"
            @click="revokeConnection(c.client_id)"
          >Disconnect</button>
        </li>
      </ul>
    </div>

    <!-- Personal-access tokens (CLI clients) -->
    <div class="rounded border border-border bg-surface p-3">
      <div class="text-xs font-medium text-surface-foreground">Access tokens</div>
      <p class="mb-2 mt-0.5 text-[11px] text-muted">
        For CLI clients (Claude Code, Cursor). Treat a token like a password.
      </p>

      <div class="flex items-center gap-2">
        <input
          v-model="newLabel"
          type="text"
          maxlength="120"
          placeholder="Token name (e.g. &quot;Claude Code&quot;)"
          class="min-w-0 flex-1 rounded border border-border bg-canvas px-2 py-1 text-sm text-canvas-foreground"
          data-testid="agents-token-label"
          @keyup.enter="create"
        />
        <button
          type="button"
          class="shrink-0 rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          :disabled="busy || !newLabel.trim()"
          data-testid="agents-token-create"
          @click="create"
        >Create token</button>
      </div>

      <!-- The freshly-minted secret — shown once. -->
      <div
        v-if="freshSecret"
        class="mt-2 rounded border border-primary/40 bg-overlay p-2"
        data-testid="agents-fresh-secret"
      >
        <p class="mb-1 text-[11px] font-medium text-primary">
          Copy this token now — it won't be shown again.
        </p>
        <div class="flex items-center gap-2">
          <code class="min-w-0 flex-1 truncate rounded bg-canvas px-2 py-1 font-mono text-[11px]">{{ freshSecret }}</code>
          <button
            type="button"
            class="shrink-0 rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay"
            @click="copy(freshSecret, 'secret')"
          >{{ copied === 'secret' ? 'Copied' : 'Copy' }}</button>
        </div>
      </div>

      <ul v-if="tokens.length" class="mt-2 divide-y divide-border">
        <li
          v-for="tok in tokens"
          :key="tok.id"
          class="flex items-center justify-between gap-3 py-1.5"
        >
          <div class="min-w-0">
            <div class="truncate text-sm text-canvas-foreground">{{ tok.label }}</div>
            <div class="text-[11px] text-muted">
              Created {{ fmt(tok.created_at) }} ·
              {{ tok.last_used_at ? `last used ${fmt(tok.last_used_at)}` : 'never used' }}
            </div>
          </div>
          <button
            type="button"
            class="shrink-0 rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay hover:text-danger disabled:opacity-50"
            :disabled="busy"
            data-testid="agents-token-revoke"
            @click="revoke(tok.id)"
          >Revoke</button>
        </li>
      </ul>
      <p v-else-if="loaded" class="mt-2 text-xs text-muted">No tokens yet.</p>
    </div>
  </div>
</template>
