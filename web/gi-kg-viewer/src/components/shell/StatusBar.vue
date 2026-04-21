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

const sourcesTab = ref<'feeds' | 'profile' | 'operator'>('feeds')
const feedsEditorText = ref('')
const operatorYamlBody = ref('')
const operatorProfileSelected = ref('')
const availableProfiles = ref<string[]>([])
const operatorFileHint = ref('')
/** Corpus path (trimmed) we last GET for operator YAML; skip re-fetch when switching Profile ↔ Overrides. */
const operatorSourcesLoadedForPath = ref('')
const sourcesBusy = ref(false)
const sourcesError = ref<string | null>(null)
/** One RSS URL per line — merged into the JSON ``feeds`` array (legacy ``--rss-file`` style). */
const feedsLinePaste = ref('')

/** Advertised in GET /api/health but often off in minimal server builds. */
const optionalHealthCapsLimited = computed(
  () =>
    !shell.feedsApiAvailable ||
    !shell.operatorConfigApiAvailable ||
    !shell.jobsApiAvailable,
)

const healthDotClass = computed(() => {
  if (shell.healthError) {
    return 'bg-danger'
  }
  if (!shell.healthStatus) {
    return 'bg-muted'
  }
  const st = String(shell.healthStatus).toLowerCase()
  if (st !== 'ok') {
    return 'bg-warning'
  }
  /** Reachable + OK, but viewer cannot drive corpus APIs until path is set. */
  if (!shell.hasCorpusPath) {
    return 'bg-warning'
  }
  if (optionalHealthCapsLimited.value) {
    return 'bg-warning'
  }
  return 'bg-success'
})

/** Single summary line: server status + viewer context (path / optional routes). */
const healthDialogSummary = computed(() => {
  if (shell.healthError) {
    return { text: shell.healthError, textClass: 'text-danger' }
  }
  if (!shell.healthStatus) {
    return { text: 'Checking…', textClass: 'text-muted' }
  }
  const st = String(shell.healthStatus).toLowerCase()
  if (st !== 'ok') {
    return { text: shell.healthStatusDisplay, textClass: 'text-warning' }
  }
  if (!shell.hasCorpusPath) {
    return {
      text: 'OK — corpus path not set',
      textClass: 'text-warning',
    }
  }
  if (optionalHealthCapsLimited.value) {
    return {
      text: 'OK — limited (optional APIs off)',
      textClass: 'text-warning',
    }
  }
  return { text: shell.healthStatusDisplay, textClass: 'text-success' }
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
async function loadSourcesTab(tab: 'feeds' | 'profile' | 'operator'): Promise<void> {
  const p = shell.corpusPath.trim()
  sourcesBusy.value = true
  sourcesError.value = null
  try {
    if (tab === 'feeds' && shell.feedsApiAvailable) {
      const f = await getFeeds(p)
      feedsEditorText.value = JSON.stringify({ feeds: f.feeds }, null, 2)
    } else if (
      (tab === 'profile' || tab === 'operator') &&
      shell.operatorConfigApiAvailable
    ) {
      if (operatorSourcesLoadedForPath.value === p) {
        return
      }
      const o = await getOperatorConfig(p)
      operatorFileHint.value = o.operator_config_path
      availableProfiles.value = o.available_profiles ?? []
      const sp = splitOperatorYamlProfile(o.content)
      operatorProfileSelected.value = sp.profile
      operatorYamlBody.value = sp.body
      operatorSourcesLoadedForPath.value = p
    }
  } catch (e) {
    sourcesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    sourcesBusy.value = false
  }
}

async function openSourcesDialog(tab: 'feeds' | 'profile' | 'operator'): Promise<void> {
  sourcesTab.value = tab
  await loadSourcesTab(tab)
  sourcesDialogRef.value?.showModal()
}

/** Open configuration dialog: Feed list tab if feeds API is on, else operator (YAML) tab. */
async function openSourcesDialogDefault(): Promise<void> {
  const feedsOn = shell.feedsApiAvailable
  const tab: 'feeds' | 'profile' | 'operator' = feedsOn ? 'feeds' : 'operator'
  await openSourcesDialog(tab)
}

function closeSourcesDialog(): void {
  operatorSourcesLoadedForPath.value = ''
  sourcesDialogRef.value?.close()
}

async function selectSourcesTab(tab: 'feeds' | 'profile' | 'operator'): Promise<void> {
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
    operatorSourcesLoadedForPath.value = p
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
    class="flex h-9 min-w-0 shrink-0 items-center gap-2 border-t border-border bg-canvas px-2 text-xs text-canvas-foreground"
    data-testid="app-status-bar"
  >
    <label class="sr-only" for="status-bar-corpus-path-input">Corpus path</label>
    <input
      id="status-bar-corpus-path-input"
      v-model="corpusPathModel"
      type="text"
      data-testid="status-bar-corpus-path"
      class="h-7 min-w-[12rem] w-[min(56.25rem,calc(100vw-14rem))] shrink-0 rounded border border-border bg-elevated px-2 py-0.5 font-mono text-[11px] text-elevated-foreground placeholder:text-muted"
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
    <div class="ml-auto flex min-w-0 shrink-0 items-center gap-2">
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
      <button
        v-if="
          shell.healthStatus &&
            shell.hasCorpusPath &&
            (shell.feedsApiAvailable || shell.operatorConfigApiAvailable)
        "
        type="button"
        class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
        title="Feed list, pipeline profile, and YAML overrides"
        data-testid="status-bar-sources-trigger"
        @click="void openSourcesDialogDefault()"
      >
        Configuration
      </button>
      <button
        type="button"
        class="flex shrink-0 items-center gap-1 rounded border border-border px-1.5 py-0.5 hover:bg-overlay"
        data-testid="status-bar-health-trigger"
        title="Health: server status, corpus path, and advertised API routes"
        @click="openHealthDialog"
      >
        <span
          class="inline-block h-2 w-2 shrink-0 rounded-full"
          :class="healthDotClass"
          aria-hidden="true"
        />
        <span class="hidden text-[10px] sm:inline">Health</span>
      </button>
    </div>
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
          <dd class="text-right font-medium leading-snug">
            <span :class="healthDialogSummary.textClass">{{ healthDialogSummary.text }}</span>
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
          <dd :class="shell.feedsApiAvailable ? 'text-success' : 'text-danger'">
            {{ shell.feedsApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Operator YAML API
          </dt>
          <dd :class="shell.operatorConfigApiAvailable ? 'text-success' : 'text-danger'">
            {{ shell.operatorConfigApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Pipeline jobs API
          </dt>
          <dd :class="shell.jobsApiAvailable ? 'text-success' : 'text-danger'">
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
    class="w-[min(96vw,32rem)] overflow-hidden rounded-lg border border-border bg-surface p-0 text-xs text-surface-foreground shadow-lg backdrop:bg-black/40"
    aria-labelledby="status-bar-configuration-title"
    data-testid="status-bar-sources-dialog"
  >
    <!-- Inner flex wrapper: avoid ``display:flex`` on ``<dialog>`` — it overrides UA ``display:none`` when closed. -->
    <div class="flex max-h-[min(92vh,44rem)] min-h-0 flex-col gap-2 p-3">
    <div class="shrink-0 space-y-2 border-b border-border pb-2">
      <div class="flex items-start justify-between gap-2">
        <div class="min-w-0">
          <h2 id="status-bar-configuration-title" class="text-sm font-semibold">
            Configuration
          </h2>
        </div>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-2 py-0.5 text-[10px] hover:bg-overlay"
          @click="closeSourcesDialog"
        >
          Close
        </button>
      </div>
      <div class="flex flex-wrap gap-1">
        <button
          v-if="shell.feedsApiAvailable"
          type="button"
          class="rounded px-2 py-0.5 text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'feeds' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-feeds"
          @click="void selectSourcesTab('feeds')"
        >
          Feed list
        </button>
        <button
          v-if="shell.operatorConfigApiAvailable"
          type="button"
          class="rounded px-2 py-0.5 text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'profile' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-profile"
          @click="void selectSourcesTab('profile')"
        >
          Profile
        </button>
        <button
          v-if="shell.operatorConfigApiAvailable"
          type="button"
          class="rounded px-2 py-0.5 text-[10px] hover:bg-overlay"
          :class="sourcesTab === 'operator' ? 'bg-overlay font-medium' : 'text-muted'"
          data-testid="sources-dialog-tab-operator"
          @click="void selectSourcesTab('operator')"
        >
          Overrides
        </button>
      </div>
    </div>
    <p v-if="sourcesBusy" class="shrink-0 text-[10px] text-muted">
      Loading…
    </p>
    <p v-if="sourcesError" class="shrink-0 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-[10px] text-danger">
      {{ sourcesError }}
    </p>
    <!-- Explicit max-height (not flex-1) so the tab row never loses space to the scroll sibling. -->
    <div class="min-h-0 max-h-[min(58vh,32rem)] overflow-y-auto overscroll-contain pr-0.5">
    <div v-show="sourcesTab === 'feeds' && shell.feedsApiAvailable" class="flex flex-col gap-2">
      <p
        v-if="shell.operatorConfigApiAvailable"
        class="text-[10px] text-muted leading-snug"
      >
        Pipeline preset + YAML (<code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/operator-config</code>):
        <button
          type="button"
          class="font-medium text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
          data-testid="sources-dialog-jump-to-profile"
          @click="void selectSourcesTab('profile')"
        >
          Profile
        </button>
        (packaged <code class="font-mono text-[9px]">profile:</code>)
        or
        <button
          type="button"
          class="font-medium text-surface-foreground underline decoration-dotted underline-offset-2 hover:decoration-solid"
          data-testid="sources-dialog-jump-to-operator"
          @click="void selectSourcesTab('operator')"
        >
          Overrides
        </button>
        (YAML without top-level <code class="font-mono text-[9px]">profile:</code>).
      </p>
      <p class="text-[10px] text-muted leading-snug">
        Structured <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">feeds.spec.yaml</code> on disk under the corpus root. Edit JSON here: root <code class="font-mono text-[9px]">feeds</code> is an array of URL strings or objects with <code class="font-mono text-[9px]">url</code> plus optional per-feed overrides (same shape as CLI <code class="font-mono text-[9px]">--feeds-spec</code>). Legacy one-URL-per-line lists belong with <code class="font-mono text-[9px]">--rss-file</code> on the CLI — use the box below to turn lines into this JSON shape. Do not duplicate feeds in operator config.
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
    <div v-show="sourcesTab === 'profile' && shell.operatorConfigApiAvailable" class="flex flex-col gap-2">
      <p v-if="operatorFileHint" class="break-all text-[9px] text-muted leading-snug">
        {{ operatorFileHint }}
      </p>
      <p class="text-[10px] text-muted leading-snug">
        Packaged preset <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">profile:</code> merges first; keys in <strong class="text-surface-foreground">Overrides</strong> win — same idea as CLI <code class="font-mono text-[9px]">--profile</code> plus <code class="font-mono text-[9px]">--config</code>. This menu is the source of truth for top-level <code class="font-mono text-[9px]">profile:</code>: <strong>None</strong> removes it even if a stale line exists on disk. If the menu is empty, no packaged presets were found (check <code class="font-mono text-[9px]">config/profiles</code>). Do not put API keys in YAML — use environment variables. RSS / feeds belong in <strong class="text-surface-foreground">Feed list</strong>; the server rejects feed keys and secrets on save.
      </p>
      <div class="flex flex-wrap items-center gap-2">
        <label
          for="sources-dialog-profile-select"
          class="text-[10px] text-muted shrink-0"
        >Preset</label>
        <select
          id="sources-dialog-profile-select"
          v-model="operatorProfileSelected"
          data-testid="sources-dialog-profile-select"
          class="max-w-[min(100%,16rem)] rounded border border-border bg-elevated px-2 py-1 text-[11px] text-elevated-foreground"
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
      <button
        type="button"
        class="self-start rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="sourcesBusy"
        data-testid="sources-dialog-save-profile"
        @click="void saveOperatorFromDialog()"
      >
        Save (applies preset + overrides on disk)
      </button>
    </div>
    <div v-show="sourcesTab === 'operator' && shell.operatorConfigApiAvailable" class="flex flex-col gap-2">
      <p v-if="operatorFileHint" class="break-all text-[9px] text-muted leading-snug">
        {{ operatorFileHint }}
      </p>
      <p class="text-[10px] text-muted leading-snug">
        YAML overrides only (no top-level <code class="font-mono text-[9px]">profile:</code> — use the <strong class="text-surface-foreground">Profile</strong> tab for the preset line). Same file as the API response; saving merges with the current preset from the Profile tab.
      </p>
      <textarea
        v-model="operatorYamlBody"
        data-testid="sources-dialog-operator-textarea"
        class="min-h-[10rem] w-full resize-y rounded border border-border bg-elevated p-2 font-mono text-[11px] text-elevated-foreground"
        spellcheck="false"
        aria-label="Operator config overrides (YAML)"
      />
      <button
        type="button"
        class="self-start rounded border border-border px-2 py-1 text-[10px] hover:bg-overlay disabled:opacity-40"
        :disabled="sourcesBusy"
        data-testid="sources-dialog-save-overrides"
        @click="void saveOperatorFromDialog()"
      >
        Save YAML
      </button>
    </div>
    </div>
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
