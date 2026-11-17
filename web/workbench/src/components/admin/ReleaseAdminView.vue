<script setup lang="ts">
/**
 * Admin → Released app version. The newest app version people can install from TestFlight / Play;
 * the native app shows "update available" when it is higher than its own. Saves to the admin
 * release endpoint, which the player API reads per request — so a native-only release moves the
 * prompt with no deploy and no restart. "Use deploy default" clears the override and the
 * deployment's value (the PLAYER_RELEASED_VERSION variable) applies again.
 */
import { onMounted, ref } from 'vue'
import { fetchRelease, saveRelease, type ReleaseDTO } from '../../api/authApi'

const release = ref<ReleaseDTO | null>(null)
const draft = ref('')
const loading = ref(false)
const saving = ref(false)
const error = ref<string | null>(null)
const saved = ref(false)

/** The same rule the server enforces: dotted numbers, e.g. 1.0.2. */
const VERSION = /^\d{1,4}(\.\d{1,4}){0,3}$/

function adopt(r: ReleaseDTO): void {
  release.value = r
  draft.value = r.override ?? ''
}

async function load(): Promise<void> {
  loading.value = true
  error.value = null
  try {
    adopt(await fetchRelease())
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load the released version'
  } finally {
    loading.value = false
  }
}
onMounted(load)

async function save(version: string | null): Promise<void> {
  if (version !== null && !VERSION.test(version)) {
    error.value = 'Use a dotted number, e.g. 1.0.2.'
    return
  }
  saving.value = true
  error.value = null
  saved.value = false
  try {
    adopt(await saveRelease(version))
    saved.value = true
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to save the released version'
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="mx-auto max-w-2xl" data-testid="release-admin">
    <h2 class="text-base font-semibold text-surface-foreground">Released app version</h2>
    <p class="mb-3 text-xs text-muted">
      The newest version people can install from TestFlight / Play. Apps below it show "update
      available". Takes effect immediately — no deploy.
    </p>
    <p v-if="loading" class="text-xs text-muted" data-testid="release-loading">Loading…</p>
    <template v-else>
      <dl v-if="release" class="mb-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt class="text-muted">Served now</dt>
        <dd class="font-mono text-surface-foreground" data-testid="release-served">
          {{ release.player_version ?? '— (no prompt)' }}
        </dd>
        <dt class="text-muted">Deploy default</dt>
        <dd class="font-mono text-surface-foreground" data-testid="release-deploy-default">
          {{ release.deploy_default ?? '—' }}
        </dd>
      </dl>
      <div class="flex flex-wrap items-center gap-2">
        <label class="flex items-center gap-1 text-xs text-muted">
          Override
          <input
            v-model.trim="draft"
            type="text"
            inputmode="decimal"
            placeholder="e.g. 1.0.2"
            class="w-24 rounded border border-border bg-surface px-1 py-0.5 font-mono text-surface-foreground"
            data-testid="release-input"
            @input="saved = false"
          />
        </label>
        <button
          type="button"
          class="rounded bg-primary px-3 py-1 text-sm font-medium text-primary-foreground disabled:opacity-50"
          :disabled="saving || !draft"
          data-testid="release-save"
          @click="save(draft)"
        >
          {{ saving ? 'Saving…' : 'Save' }}
        </button>
        <button
          type="button"
          class="rounded border border-border px-3 py-1 text-sm text-surface-foreground disabled:opacity-50"
          :disabled="saving || !release?.override"
          data-testid="release-clear"
          @click="save(null)"
        >
          Use deploy default
        </button>
        <span v-if="saved" class="text-xs text-grounded" data-testid="release-saved">Saved ✓</span>
      </div>
    </template>
    <p
      v-if="error"
      class="mt-2 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-xs text-danger"
      role="alert"
      data-testid="release-error"
    >
      {{ error }}
    </p>
  </div>
</template>
