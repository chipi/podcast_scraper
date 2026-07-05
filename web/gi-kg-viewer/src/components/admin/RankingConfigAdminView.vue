<script setup lang="ts">
/**
 * Admin → Discovery ranking (#11 / B2). Edit the discovery ranking-signal registry in one place:
 * toggle each signal on/off, tune its weight, and edit any signal-specific params (e.g. the trend
 * cap). Saves to the admin ranking-config endpoint; the discovery feed loads the active config per
 * request, so changes take effect immediately. A malformed save can't empty ranking — the backend
 * merges onto the defaults.
 */
import { onMounted, ref } from 'vue'
import {
  fetchRankingConfig,
  saveRankingConfig,
  type RankingSignalDTO,
} from '../../api/authApi'

const signals = ref<RankingSignalDTO[]>([])
const loading = ref(false)
const saving = ref(false)
const error = ref<string | null>(null)
const saved = ref(false)

async function load(): Promise<void> {
  loading.value = true
  error.value = null
  try {
    signals.value = (await fetchRankingConfig()).signals
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load ranking config'
  } finally {
    loading.value = false
  }
}
onMounted(load)

function paramKeys(sig: RankingSignalDTO): string[] {
  return Object.keys(sig.params ?? {})
}

function setParam(sig: RankingSignalDTO, key: string, raw: string): void {
  const n = Number(raw)
  sig.params[key] = raw.trim() !== '' && Number.isFinite(n) ? n : raw
  saved.value = false
}

async function save(): Promise<void> {
  saving.value = true
  error.value = null
  saved.value = false
  try {
    signals.value = (await saveRankingConfig({ signals: signals.value })).signals
    saved.value = true
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to save ranking config'
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="mx-auto max-w-2xl" data-testid="ranking-config-admin">
    <h2 class="text-base font-semibold text-surface-foreground">Discovery ranking</h2>
    <p class="mb-3 text-xs text-muted">
      Toggle signals on/off and tune their weights; a disabled signal contributes nothing. Changes
      apply to the discovery feed immediately. Clicks vs configured rank are logged for A/B.
    </p>
    <p v-if="loading" class="text-xs text-muted" data-testid="ranking-config-loading">Loading…</p>
    <p
      v-else-if="error"
      class="rounded border border-danger/40 bg-danger/10 px-2 py-1 text-xs text-danger"
      role="alert"
      data-testid="ranking-config-error"
    >
      {{ error }}
    </p>
    <ul v-else class="space-y-2" data-testid="ranking-config-signals">
      <li
        v-for="sig in signals"
        :key="sig.name"
        class="rounded border border-border bg-elevated/40 p-2"
        :class="sig.enabled ? '' : 'opacity-60'"
        :data-testid="`ranking-signal-${sig.name}`"
      >
        <div class="flex items-center justify-between gap-2">
          <label class="flex items-center gap-2 text-sm font-medium text-surface-foreground">
            <input
              v-model="sig.enabled"
              type="checkbox"
              :data-testid="`ranking-enabled-${sig.name}`"
              @change="saved = false"
            />
            {{ sig.name }}
          </label>
          <label class="flex items-center gap-1 text-xs text-muted">
            weight
            <input
              v-model.number="sig.weight"
              type="number"
              step="0.1"
              class="w-20 rounded border border-border bg-surface px-1 py-0.5 text-right text-surface-foreground"
              :data-testid="`ranking-weight-${sig.name}`"
              @input="saved = false"
            />
          </label>
        </div>
        <div v-if="paramKeys(sig).length" class="mt-1.5 flex flex-wrap gap-2 pl-6">
          <label
            v-for="k in paramKeys(sig)"
            :key="k"
            class="flex items-center gap-1 text-[11px] text-muted"
          >
            {{ k }}
            <input
              type="text"
              :value="String(sig.params[k])"
              class="w-16 rounded border border-border bg-surface px-1 py-0.5 text-surface-foreground"
              :data-testid="`ranking-param-${sig.name}-${k}`"
              @input="setParam(sig, k, ($event.target as HTMLInputElement).value)"
            />
          </label>
        </div>
      </li>
    </ul>
    <div class="mt-3 flex items-center gap-3">
      <button
        type="button"
        class="rounded bg-primary px-3 py-1 text-sm font-medium text-primary-foreground disabled:opacity-50"
        :disabled="saving || loading"
        data-testid="ranking-config-save"
        @click="save"
      >
        {{ saving ? 'Saving…' : 'Save' }}
      </button>
      <span v-if="saved" class="text-xs text-grounded" data-testid="ranking-config-saved">Saved ✓</span>
    </div>
  </div>
</template>
