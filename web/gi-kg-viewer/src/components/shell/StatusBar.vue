<script setup lang="ts">
import { computed, ref, useTemplateRef } from 'vue'
import { getFeeds, putFeeds, type FeedApiEntry } from '../../api/feedsApi'
import { getOperatorConfig, putOperatorConfig } from '../../api/operatorConfigApi'
import { mergeOperatorYamlProfile, splitOperatorYamlProfile } from '../../utils/operatorYamlProfile'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const localFileInputRef = useTemplateRef<HTMLInputElement>('localFileInputRef')
const healthDialogRef = useTemplateRef<HTMLDialogElement>('healthDialogRef')
const indexDialogRef = useTemplateRef<HTMLDialogElement>('indexDialogRef')
const artifactListDialogRef = useTemplateRef<HTMLDialogElement>('artifactListDialogRef')
const sourcesDialogRef = useTemplateRef<HTMLDialogElement>('sourcesDialogRef')

const sourcesTab = ref<'feeds' | 'operator'>('feeds')
const feedsEditorText = ref('')
const operatorYamlBody = ref('')
const operatorProfileSelected = ref('')
const availableProfiles = ref<string[]>([])
const operatorFileHint = ref('')
const sourcesBusy = ref(false)
const sourcesError = ref<string | null>(null)
/** One RSS URL per line — merged into the JSON ``feeds`` array (legacy ``--rss-file`` style). */
const feedsLinePaste = ref('')

const healthDotClass = computed(() => {
  if (shell.healthError) {
    return 'bg-danger'
  }
  /** VIEWER_IA: no corpus configured — show danger even if health is still loading. */
  if (!shell.hasCorpusPath) {
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
  'go-graph': []
}>()

async function onListArtifactsClick(): Promise<void> {
  await shell.fetchArtifactList()
  artifactListDialogRef.value?.showModal()
}

async function onLoadIntoGraphFromDialog(): Promise<void> {
  await artifacts.loadSelected()
  if (artifacts.displayArtifact) {
    emit('go-graph')
    artifactListDialogRef.value?.close()
  }
}

async function onLocalFilesChange(ev: Event): Promise<void> {
  const el = ev.target as HTMLInputElement
  await artifacts.loadFromLocalFiles(el.files)
  el.value = ''
  emit('localArtifactsLoaded', Boolean(artifacts.displayArtifact))
}

/** Load only the active tab so a broken operator file does not block the Feeds editor. */
async function loadSourcesTab(tab: 'feeds' | 'operator'): Promise<void> {
  const p = shell.corpusPath.trim()
  sourcesBusy.value = true
  sourcesError.value = null
  try {
    if (tab === 'feeds' && shell.feedsApiAvailable) {
      const f = await getFeeds(p)
      feedsEditorText.value = JSON.stringify({ feeds: f.feeds }, null, 2)
    } else if (tab === 'operator' && shell.operatorConfigApiAvailable) {
      const o = await getOperatorConfig(p)
      operatorFileHint.value = o.operator_config_path
      availableProfiles.value = o.available_profiles ?? []
      const sp = splitOperatorYamlProfile(o.content)
      operatorProfileSelected.value = sp.profile
      operatorYamlBody.value = sp.body
    }
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}

async function openSourcesDialog(tab: 'feeds' | 'operator'): Promise<void> {
  sourcesTab.value = tab
  await loadSourcesTab(tab)
  sourcesDialogRef.value?.showModal()
}

async function selectSourcesTab(tab: 'feeds' | 'operator'): Promise<void> {
  if (sourcesTab.value === tab) {
    return
  }
  sourcesTab.value = tab
  await loadSourcesTab(tab)
}

function appendFeedUrlsFromLines(): void {
  sourcesError.value = null
  const rawLines = feedsLinePaste.value.split(/\r?\n/)
  const urls = rawLines.map((s) => s.trim()).filter((s) => s.length > 0)
  if (urls.length === 0) {
    sourcesError.value = 'Paste at least one URL line first.'
    return
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(feedsEditorText.value) as unknown
  } catch {
    sourcesError.value = 'Fix JSON in the editor before merging lines (or clear and try again).'
    return
  }
  const root = parsed as { feeds?: unknown }
  if (!root || typeof root !== 'object' || !Array.isArray(root.feeds)) {
    sourcesError.value = 'Root must be an object with a "feeds" array.'
    return
  }
  const existing = root.feeds as FeedApiEntry[]
  const seen = new Set<string>()
  for (const e of existing) {
    if (typeof e === 'string') {
      seen.add(e.trim())
    } else if (e && typeof e === 'object' && typeof (e as { url?: unknown }).url === 'string') {
      seen.add(String((e as { url: string }).url).trim())
    }
  }
  const merged: FeedApiEntry[] = [...existing]
  for (const u of urls) {
    if (seen.has(u)) {
      continue
    }
    seen.add(u)
    merged.push(u)
  }
  feedsEditorText.value = JSON.stringify({ feeds: merged }, null, 2)
  feedsLinePaste.value = ''
}

async function saveFeedsFromDialog(): Promise<void> {
  sourcesBusy.value = true
  sourcesError.value = null
  const p = shell.corpusPath.trim()
  try {
    let parsed: unknown
    try {
      parsed = JSON.parse(feedsEditorText.value) as unknown
    } catch {
      sourcesError.value = 'Invalid JSON — fix syntax before saving.'
      return
    }
    const root = parsed as { feeds?: unknown }
    if (!root || typeof root !== 'object' || !Array.isArray(root.feeds)) {
      sourcesError.value = 'Root must be an object with a "feeds" array.'
      return
    }
    const out = await putFeeds(p, root.feeds as FeedApiEntry[])
    feedsEditorText.value = JSON.stringify({ feeds: out.feeds }, null, 2)
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}

async function saveOperatorFromDialog(): Promise<void> {
  sourcesBusy.value = true
  sourcesError.value = null
  const p = shell.corpusPath.trim()
  try {
    // Dropdown is the only source of truth for `profile:` (None = omit line; ignore pasted profile: in textarea).
    const inner = splitOperatorYamlProfile(operatorYamlBody.value)
    const prof = operatorProfileSelected.value.trim()
    const merged = mergeOperatorYamlProfile(prof, inner.body)
    const out = await putOperatorConfig(p, merged)
    availableProfiles.value = out.available_profiles ?? availableProfiles.value
    const sp = splitOperatorYamlProfile(out.content)
    operatorProfileSelected.value = sp.profile
    operatorYamlBody.value = sp.body
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
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
      placeholder="Set corpus path…"
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
      v-if="shell.healthStatus && shell.corpusPath.trim()"
      type="button"
      class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
      title="List GI/KG artifacts from the API for this corpus path"
      data-testid="status-bar-list-artifacts"
      @click="void onListArtifactsClick()"
    >
      List
    </button>
    <button
      v-if="shell.healthStatus && shell.hasCorpusPath && shell.feedsApiAvailable"
      type="button"
      class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
      title="Edit feeds.spec.yaml (JSON in UI) for this corpus"
      data-testid="status-bar-feeds-trigger"
      @click="void openSourcesDialog('feeds')"
    >
      Feeds
    </button>
    <button
      v-if="shell.healthStatus && shell.hasCorpusPath && shell.operatorConfigApiAvailable"
      type="button"
      class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
      title="Edit operator YAML (no API keys — use environment variables)"
      data-testid="status-bar-operator-config-trigger"
      @click="void openSourcesDialog('operator')"
    >
      Config
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
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Feeds file API
          </dt>
          <dd :class="shell.feedsApiAvailable ? 'text-success' : 'text-muted'">
            {{ shell.feedsApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Operator YAML API
          </dt>
          <dd :class="shell.operatorConfigApiAvailable ? 'text-success' : 'text-muted'">
            {{ shell.operatorConfigApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Pipeline jobs API
          </dt>
          <dd :class="shell.jobsApiAvailable ? 'text-success' : 'text-muted'">
            {{ shell.jobsApiAvailable ? 'Yes' : 'No' }}
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
    ref="sourcesDialogRef"
    class="max-h-[min(85vh,36rem)] w-[min(96vw,28rem)] overflow-hidden rounded-lg border border-border bg-surface p-3 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-sources-dialog-title"
    data-testid="status-bar-sources-dialog"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 id="status-bar-sources-dialog-title" class="text-sm font-semibold">
        Corpus sources
      </h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        @click="sourcesDialogRef?.close()"
      >
        Close
      </button>
    </div>
    <div class="mb-2 flex gap-1 border-b border-border pb-2">
      <button
        v-if="shell.feedsApiAvailable"
        type="button"
        class="rounded px-2 py-0.5 text-[10px] hover:bg-overlay"
        :class="sourcesTab === 'feeds' ? 'bg-overlay font-medium' : 'text-muted'"
        data-testid="sources-dialog-tab-feeds"
        @click="void selectSourcesTab('feeds')"
      >
        Feeds
      </button>
      <button
        v-if="shell.operatorConfigApiAvailable"
        type="button"
        class="rounded px-2 py-0.5 text-[10px] hover:bg-overlay"
        :class="sourcesTab === 'operator' ? 'bg-overlay font-medium' : 'text-muted'"
        data-testid="sources-dialog-tab-operator"
        @click="void selectSourcesTab('operator')"
      >
        Operator YAML
      </button>
    </div>
    <p v-if="sourcesBusy" class="mb-2 text-[10px] text-muted">
      Loading…
    </p>
    <p v-if="sourcesError" class="mb-2 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[10px] text-danger">
      {{ sourcesError }}
    </p>
    <div v-show="sourcesTab === 'feeds' && shell.feedsApiAvailable" class="flex max-h-[min(60vh,22rem)] flex-col gap-2">
      <p class="text-[10px] text-muted leading-snug">
        Structured <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">feeds.spec.yaml</code> on disk (RFC-077 / #626). Edit JSON here: root <code class="font-mono text-[9px]">feeds</code> is an array of URL strings or objects with <code class="font-mono text-[9px]">url</code> plus optional per-feed overrides (same shape as CLI <code class="font-mono text-[9px]">--feeds-spec</code>). Legacy one-URL-per-line lists belong with <code class="font-mono text-[9px]">--rss-file</code> on the CLI — use the box below to turn lines into this JSON shape. Do not duplicate feeds in Operator YAML.
      </p>
      <textarea
        v-model="feedsLinePaste"
        data-testid="sources-dialog-feeds-lines-textarea"
        class="min-h-[4rem] w-full resize-y rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
        spellcheck="false"
        aria-label="Paste RSS URLs one per line"
        placeholder="https://example.com/feed.xml (one per line)"
      />
      <button
        type="button"
        class="self-start rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="sourcesBusy"
        data-testid="sources-dialog-feeds-merge-lines"
        @click="appendFeedUrlsFromLines"
      >
        Append lines to feeds JSON
      </button>
      <textarea
        v-model="feedsEditorText"
        data-testid="sources-dialog-feeds-textarea"
        class="min-h-[10rem] w-full resize-y rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
        spellcheck="false"
        aria-label="Feeds spec JSON (feeds array)"
      />
      <button
        type="button"
        class="self-start rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="sourcesBusy"
        @click="void saveFeedsFromDialog()"
      >
        Save feeds
      </button>
    </div>
    <div v-show="sourcesTab === 'operator' && shell.operatorConfigApiAvailable" class="flex max-h-[min(60vh,22rem)] flex-col gap-2">
      <p v-if="operatorFileHint" class="break-all text-[9px] text-muted leading-snug">
        {{ operatorFileHint }}
      </p>
      <p class="text-[10px] text-muted leading-snug">
        Packaged preset <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">profile:</code> defaults merge first; keys below override. The <strong>Profile</strong> menu is what Save writes for <code class="font-mono text-[9px]">profile:</code> — <strong>None</strong> removes it even if you pasted a <code class="font-mono text-[9px]">profile:</code> line in the box. When the operator file is missing or empty, the server creates <code class="font-mono text-[9px]">profile: cloud_balanced</code> on first load if that preset exists under packaged <code class="font-mono text-[9px]">config/profiles</code>. If the menu is empty, no packaged presets were found (check <code class="font-mono text-[9px]">config/profiles</code>). Do not put API keys here — use environment variables. RSS / feed lists belong in Feeds (<code class="font-mono text-[9px]">feeds.spec.yaml</code>); the server rejects feed keys and secrets on save (top-level only).
      </p>
      <div class="flex flex-wrap items-center gap-2">
        <label
          for="sources-dialog-profile-select"
          class="text-[10px] text-muted shrink-0"
        >Profile</label>
        <select
          id="sources-dialog-profile-select"
          v-model="operatorProfileSelected"
          data-testid="sources-dialog-profile-select"
          class="max-w-[min(100%,14rem)] rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
          aria-label="Pipeline profile preset"
        >
          <option value="">
            None
          </option>
          <option
            v-if="operatorProfileSelected && !availableProfiles.includes(operatorProfileSelected)"
            :value="operatorProfileSelected"
          >
            {{ operatorProfileSelected }} (custom)
          </option>
          <option
            v-for="n in availableProfiles"
            :key="n"
            :value="n"
          >
            {{ n }}
          </option>
        </select>
      </div>
      <p class="text-[10px] text-muted leading-snug">
        Overrides (YAML, without top-level <code class="font-mono text-[9px]">profile:</code> — use the menu above):
      </p>
      <textarea
        v-model="operatorYamlBody"
        data-testid="sources-dialog-operator-textarea"
        class="min-h-[10rem] w-full resize-y rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
        spellcheck="false"
        aria-label="Operator YAML overrides"
      />
      <button
        type="button"
        class="self-start rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="sourcesBusy"
        @click="void saveOperatorFromDialog()"
      >
        Save YAML
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

  <dialog
    ref="artifactListDialogRef"
    class="max-h-[min(80vh,32rem)] max-w-lg overflow-y-auto rounded-lg border border-border bg-surface p-4 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-artifact-list-title"
    data-testid="artifact-list-dialog"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h2 id="status-bar-artifact-list-title" class="text-sm font-semibold">
        Corpus artifacts
      </h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
        @click="artifactListDialogRef?.close()"
      >
        Close
      </button>
    </div>
    <p v-if="shell.artifactsLoading" class="mt-1 text-[10px] text-muted">
      Loading…
    </p>
    <template v-else>
      <div
        v-if="shell.corpusHints.length"
        class="mt-1.5 rounded border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10px] text-surface-foreground"
        role="status"
      >
        <p class="font-medium text-warning">
          Corpus path hint
        </p>
        <ul class="mt-0.5 list-inside list-disc text-muted">
          <li v-for="(h, i) in shell.corpusHints" :key="i">
            {{ h }}
          </li>
        </ul>
      </div>
      <div
        v-if="shell.artifactList.length"
        class="mt-1.5 space-y-1"
      >
        <div class="flex flex-wrap items-center gap-1">
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
            @click="artifacts.selectAllListed(shell.artifactList.map((a) => a.relative_path))"
          >
            All
          </button>
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
            @click="artifacts.deselectAllListed()"
          >
            None
          </button>
          <button
            type="button"
            class="rounded border border-border px-2 py-0.5 text-[10px] font-medium hover:bg-overlay disabled:opacity-40"
            :disabled="artifacts.selectedRelPaths.length === 0 || artifacts.loading"
            @click="void onLoadIntoGraphFromDialog()"
          >
            {{ artifacts.loading ? 'Loading…' : 'Load into graph' }}
          </button>
        </div>
        <div class="max-h-48 overflow-y-auto rounded border border-border bg-elevated p-1 text-[11px]">
          <label
            v-for="a in shell.artifactList"
            :key="a.relative_path"
            class="flex cursor-pointer items-start gap-1 py-0.5 hover:bg-overlay"
          >
            <input
              type="checkbox"
              class="mt-0.5 rounded border-border"
              :checked="artifacts.selectedRelPaths.includes(a.relative_path)"
              @change="artifacts.toggleSelection(a.relative_path)"
            >
            <span class="break-all">
              <span :class="a.kind === 'gi' ? 'text-gi' : 'text-kg'">{{ a.kind }}</span>
              {{ a.relative_path }}
            </span>
          </label>
        </div>
      </div>
      <p
        v-else-if="shell.artifactCount !== null && shell.artifactCount === 0"
        class="mt-1 text-[10px] text-muted"
      >
        No artifacts found.
      </p>
    </template>
    <p v-if="shell.artifactsError" class="mt-1 text-[10px] text-danger">
      {{ shell.artifactsError }}
    </p>
    <p v-if="artifacts.loadError" class="mt-1 text-[10px] text-danger">
      {{ artifacts.loadError }}
    </p>
  </dialog>
</template>
