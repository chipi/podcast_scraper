<script setup lang="ts">
import { computed, ref, useTemplateRef } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const localFileInputRef = useTemplateRef<HTMLInputElement>('localFileInputRef')
const healthDialogRef = useTemplateRef<HTMLDialogElement>('healthDialogRef')
const indexDialogRef = useTemplateRef<HTMLDialogElement>('indexDialogRef')

const healthDotClass = computed(() => {
  if (shell.healthError) {
    return 'bg-danger'
  }
  if (!shell.healthStatus) {
    return 'bg-muted'
  }
  const st = String(shell.healthStatus).toLowerCase()
  if (st === 'ok') {
    return 'bg-success'
  }
  return 'bg-warning'
})

const showRebuildBolt = computed(
  () => Boolean(indexStats.indexEnvelope?.reindex_recommended),
)

function openHealthDialog(): void {
  healthDialogRef.value?.showModal()
}

function openIndexDialog(): void {
  indexDialogRef.value?.showModal()
}

function triggerLocalFilePick(): void {
  localFileInputRef.value?.click()
}

const emit = defineEmits<{
  localArtifactsLoaded: [loaded: boolean]
}>()

async function onLocalFilesChange(ev: Event): Promise<void> {
  const el = ev.target as HTMLInputElement
  await artifacts.loadFromLocalFiles(el.files)
  el.value = ''
  emit('localArtifactsLoaded', Boolean(artifacts.displayArtifact))
}

const corpusPathModel = computed({
  get: () => shell.corpusPath,
  set: (v: string) => {
    shell.corpusPath = v
  },
})
</script>

<template>
  <footer
    class="flex h-9 shrink-0 items-center gap-2 border-t border-border bg-canvas px-2 text-xs text-canvas-foreground"
    data-testid="app-status-bar"
  >
    <label class="sr-only" for="status-bar-corpus-path-input">Corpus path</label>
    <input
      id="status-bar-corpus-path-input"
      v-model="corpusPathModel"
      type="text"
      data-testid="status-bar-corpus-path"
      class="min-w-0 flex-1 rounded border border-border bg-elevated px-2 py-0.5 text-[11px] text-elevated-foreground placeholder:text-muted"
      placeholder="/path/to/output"
      autocomplete="off"
    >
    <input
      ref="localFileInputRef"
      type="file"
      class="sr-only"
      multiple
      accept=".gi.json,.kg.json,application/json"
      data-testid="status-bar-local-file-input"
      @change="onLocalFilesChange"
    >
    <button
      type="button"
      class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
      title="Choose GI/KG JSON files (offline)"
      aria-label="Choose corpus files"
      @click="triggerLocalFilePick"
    >
      Files
    </button>
    <button
      type="button"
      class="flex shrink-0 items-center gap-1 rounded border border-border px-1.5 py-0.5 hover:bg-overlay"
      data-testid="status-bar-health-trigger"
      title="Health details"
      @click="openHealthDialog"
    >
      <span
        class="inline-block h-2 w-2 shrink-0 rounded-full"
        :class="healthDotClass"
        aria-hidden="true"
      />
      <span class="hidden text-[10px] sm:inline">Health</span>
    </button>
    <button
      v-if="showRebuildBolt"
      type="button"
      class="shrink-0 rounded border border-warning/50 px-1.5 py-0.5 text-[10px] text-warning hover:bg-warning/10"
      data-testid="status-bar-rebuild-indicator"
      title="Index refresh recommended"
      @click="openIndexDialog"
    >
      Index
    </button>
  </footer>

  <dialog
    ref="healthDialogRef"
    class="max-w-md rounded-lg border border-border bg-surface p-4 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-health-dialog-title"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 id="status-bar-health-dialog-title" class="text-sm font-semibold">
        API health
      </h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        @click="healthDialogRef?.close()"
      >
        Close
      </button>
    </div>
    <p class="mb-2 text-[10px] text-muted leading-snug">
      Flags from
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/health</code>
    </p>
    <div class="rounded border border-border bg-elevated p-2 text-[10px]">
      <dl class="space-y-1">
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Health
          </dt>
          <dd v-if="shell.healthStatus" class="font-medium text-success">
            {{ shell.healthStatusDisplay }}
          </dd>
          <dd v-else-if="shell.healthError" class="font-medium text-danger">
            {{ shell.healthError }}
          </dd>
          <dd v-else class="text-muted">
            Checking…
          </dd>
        </div>
      </dl>
      <dl
        v-if="shell.healthStatus"
        class="mt-1.5 space-y-1 border-t border-border/60 pt-1.5"
      >
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Artifacts (graph)
          </dt>
          <dd :class="shell.artifactsApiAvailable !== false ? 'text-success' : 'text-danger'">
            {{ shell.artifactsApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Semantic search
          </dt>
          <dd :class="shell.searchApiAvailable !== false ? 'text-success' : 'text-danger'">
            {{ shell.searchApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Graph explore
          </dt>
          <dd :class="shell.exploreApiAvailable !== false ? 'text-success' : 'text-danger'">
            {{ shell.exploreApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Index routes
          </dt>
          <dd :class="shell.indexRoutesApiAvailable !== false ? 'text-success' : 'text-danger'">
            {{ shell.indexRoutesApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Library API
          </dt>
          <dd :class="shell.corpusLibraryApiAvailable ? 'text-success' : 'text-danger'">
            {{ shell.corpusLibraryApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Digest API
          </dt>
          <dd :class="shell.corpusDigestApiAvailable ? 'text-success' : 'text-danger'">
            {{ shell.corpusDigestApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
      </dl>
    </div>
    <button
      type="button"
      class="mt-2 rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay"
      @click="shell.fetchHealth()"
    >
      Retry health
    </button>
    <div
      v-if="shell.healthError"
      class="mt-2 rounded border border-border bg-overlay p-2 text-[10px]"
    >
      <p class="mb-1 text-muted">
        Load files directly (no API):
      </p>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 hover:bg-canvas"
        @click="triggerLocalFilePick"
      >
        Choose files…
      </button>
    </div>
  </dialog>

  <dialog
    ref="indexDialogRef"
    class="max-w-md rounded-lg border border-border bg-surface p-4 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-index-dialog-title"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 id="status-bar-index-dialog-title" class="text-sm font-semibold">
        Vector index
      </h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        @click="indexDialogRef?.close()"
      >
        Close
      </button>
    </div>
    <p
      v-if="indexStats.indexEnvelope?.rebuild_in_progress"
      class="mb-1 text-[10px] text-muted leading-snug"
    >
      Background index job running — stats refresh automatically.
    </p>
    <p
      v-if="indexStats.indexEnvelope?.rebuild_last_error"
      class="mb-1 text-[10px] text-danger leading-snug"
    >
      Last rebuild error: {{ indexStats.indexEnvelope.rebuild_last_error }}
    </p>
    <div class="mt-2 flex flex-wrap gap-1">
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="!shell.healthStatus || indexStats.indexLoading"
        @click="indexStats.refreshIndexStats()"
      >
        {{ indexStats.indexLoading ? 'Loading…' : 'Refresh' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="indexStats.rebuildActionsDisabled"
        @click="indexStats.requestIndexRebuild(false)"
      >
        {{ indexStats.rebuildSubmitting ? 'Queueing…' : 'Update index' }}
      </button>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="indexStats.rebuildActionsDisabled"
        @click="indexStats.requestIndexRebuild(true)"
      >
        Full rebuild
      </button>
    </div>
  </dialog>
</template>
