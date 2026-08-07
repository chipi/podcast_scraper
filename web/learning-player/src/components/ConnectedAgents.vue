<script setup lang="ts">
/**
 * "Connected agents" (RFC-112 §5) — the Profile section, shown only to users who hold the
 * `mcp_access` entitlement, that wires an external AI agent to this user's corpus over MCP:
 *  - the connector URL (paste into a client that supports per-user OAuth sign-in),
 *  - personal-access tokens for CLI clients (create / copy-once / list / revoke).
 * All calls are `mcp_access`-gated server-side; this component is only mounted when the entitlement
 * is present, so it never renders for an unentitled user.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  createMcpToken,
  getMcpConfig,
  getMcpConnections,
  getMcpTokens,
  revokeMcpConnection,
  revokeMcpToken,
} from '../services/api'
import type { McpConnection, McpConnectionConfig, McpTokenMeta } from '../services/types'

const { t } = useI18n()

const config = ref<McpConnectionConfig | null>(null)
const connections = ref<McpConnection[]>([])
const tokens = ref<McpTokenMeta[]>([])
const newLabel = ref('')
const freshSecret = ref<string | null>(null)
const copiedUrl = ref(false)
const copiedSecret = ref(false)
const busy = ref(false)
const error = ref(false)

async function load(): Promise<void> {
  error.value = false
  try {
    const [cfg, conns, toks] = await Promise.all([
      getMcpConfig(),
      getMcpConnections(),
      getMcpTokens(),
    ])
    config.value = cfg
    connections.value = conns
    tokens.value = toks
  } catch {
    error.value = true
  }
}

async function revokeConnection(clientId: string): Promise<void> {
  if (busy.value) return
  busy.value = true
  error.value = false
  try {
    connections.value = await revokeMcpConnection(clientId)
  } catch {
    error.value = true
  } finally {
    busy.value = false
  }
}

async function copy(text: string, flag: 'url' | 'secret'): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    if (flag === 'url') {
      copiedUrl.value = true
      setTimeout(() => (copiedUrl.value = false), 1500)
    } else {
      copiedSecret.value = true
      setTimeout(() => (copiedSecret.value = false), 1500)
    }
  } catch {
    /* clipboard denied — the value stays visible for a manual copy */
  }
}

async function create(): Promise<void> {
  const label = newLabel.value.trim()
  if (!label || busy.value) return
  busy.value = true
  error.value = false
  try {
    const created = await createMcpToken(label)
    freshSecret.value = created.token // shown ONCE
    tokens.value = [created.meta, ...tokens.value]
    newLabel.value = ''
  } catch {
    error.value = true
  } finally {
    busy.value = false
  }
}

async function revoke(id: string): Promise<void> {
  if (busy.value) return
  busy.value = true
  error.value = false
  try {
    tokens.value = await revokeMcpToken(id)
  } catch {
    error.value = true
  } finally {
    busy.value = false
  }
}

function fmt(ts: number | null): string {
  return ts ? new Date(ts * 1000).toLocaleDateString() : ''
}

onMounted(load)
</script>

<template>
  <section class="mt-6 rounded-2xl border border-border p-5">
    <h2 class="lp-section mb-1">{{ t('agents.title') }}</h2>
    <p class="mb-4 text-sm text-muted">{{ t('agents.help') }}</p>

    <!-- Connector URL (per-user OAuth clients) -->
    <div v-if="config?.connector_url" class="mb-5">
      <div class="mb-1 text-sm font-medium">{{ t('agents.connectorUrl') }}</div>
      <div class="flex items-center gap-2">
        <code
          class="flex-1 truncate rounded-lg bg-overlay px-3 py-2 text-xs"
          :title="config.connector_url"
        >{{ config.connector_url }}</code>
        <button
          type="button"
          class="shrink-0 rounded-lg border border-border px-3 py-2 text-xs font-bold"
          @click="copy(config.connector_url, 'url')"
        >{{ copiedUrl ? t('agents.copied') : t('agents.copy') }}</button>
      </div>
      <p class="mt-1 text-xs text-muted">{{ t('agents.connectorHelp') }}</p>
    </div>
    <p v-else class="mb-5 text-xs text-muted">{{ t('agents.connectorUnset') }}</p>

    <!-- Connected OAuth apps (claude.ai etc.) — revocable -->
    <div v-if="connections.length" class="mb-5 border-t border-border pt-4">
      <div class="mb-2 text-sm font-medium">{{ t('agents.connectedApps') }}</div>
      <ul class="divide-y divide-border">
        <li
          v-for="c in connections"
          :key="c.client_id"
          class="flex items-center justify-between gap-3 py-2"
        >
          <div class="min-w-0">
            <div class="truncate text-sm font-medium">{{ c.client_name }}</div>
            <div class="text-xs text-muted">{{ t('agents.connectedScopes', { scopes: c.scopes.join(', ') }) }}</div>
          </div>
          <button
            type="button"
            class="shrink-0 text-xs font-bold text-danger disabled:opacity-50"
            :disabled="busy"
            @click="revokeConnection(c.client_id)"
          >{{ t('agents.disconnect') }}</button>
        </li>
      </ul>
    </div>

    <!-- Personal-access tokens (CLI clients) -->
    <div class="border-t border-border pt-4">
      <div class="mb-1 text-sm font-medium">{{ t('agents.tokensTitle') }}</div>
      <p class="mb-3 text-xs text-muted">{{ t('agents.tokensHelp') }}</p>

      <div class="flex items-center gap-2">
        <input
          v-model="newLabel"
          type="text"
          maxlength="120"
          :placeholder="t('agents.newTokenLabel')"
          class="flex-1 rounded-lg border border-border bg-overlay px-3 py-2 text-sm"
          @keyup.enter="create"
        />
        <button
          type="button"
          class="shrink-0 rounded-lg bg-accent px-3 py-2 text-xs font-bold text-white disabled:opacity-50"
          :disabled="busy || !newLabel.trim()"
          @click="create"
        >{{ t('agents.create') }}</button>
      </div>

      <!-- The freshly-minted secret — shown once. -->
      <div v-if="freshSecret" class="mt-3 rounded-lg border border-accent/40 bg-accent/5 p-3">
        <p class="mb-2 text-xs font-medium text-accent">{{ t('agents.createdOnce') }}</p>
        <div class="flex items-center gap-2">
          <code class="flex-1 truncate rounded bg-overlay px-2 py-1 text-xs">{{ freshSecret }}</code>
          <button
            type="button"
            class="shrink-0 rounded-lg border border-border px-3 py-1.5 text-xs font-bold"
            @click="copy(freshSecret, 'secret')"
          >{{ copiedSecret ? t('agents.copied') : t('agents.copy') }}</button>
        </div>
      </div>

      <ul v-if="tokens.length" class="mt-3 divide-y divide-border">
        <li v-for="tok in tokens" :key="tok.id" class="flex items-center justify-between gap-3 py-2">
          <div class="min-w-0">
            <div class="truncate text-sm font-medium">{{ tok.label }}</div>
            <div class="text-xs text-muted">
              {{ t('agents.created', { date: fmt(tok.created_at) }) }} ·
              {{ tok.last_used_at ? t('agents.lastUsed', { date: fmt(tok.last_used_at) }) : t('agents.neverUsed') }}
            </div>
          </div>
          <button
            type="button"
            class="shrink-0 text-xs font-bold text-danger disabled:opacity-50"
            :disabled="busy"
            @click="revoke(tok.id)"
          >{{ t('agents.revoke') }}</button>
        </li>
      </ul>
      <p v-else class="mt-3 text-sm text-muted">{{ t('agents.noTokens') }}</p>
    </div>

    <p v-if="error" class="mt-3 text-xs text-danger">{{ t('agents.error') }}</p>
  </section>
</template>
