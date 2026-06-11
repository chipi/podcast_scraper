// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Telemetry: the shell store fires posthog.capture on corpus-path change. Mock
// the SDK so v-model edits don't reach the network.
vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

// ── API mocks ────────────────────────────────────────────────────────────────
// The Configuration dialog loads feeds / operator-config over HTTP. Mock those
// modules so dialog interactions exercise render + store wiring offline.
const getFeeds = vi.fn()
const putFeeds = vi.fn()
vi.mock('../../api/feedsApi', () => ({
  getFeeds: (...a: unknown[]) => getFeeds(...a),
  putFeeds: (...a: unknown[]) => putFeeds(...a),
}))

const getOperatorConfig = vi.fn()
const putOperatorConfig = vi.fn()
vi.mock('../../api/operatorConfigApi', () => ({
  getOperatorConfig: (...a: unknown[]) => getOperatorConfig(...a),
  putOperatorConfig: (...a: unknown[]) => putOperatorConfig(...a),
}))

// indexStats store calls fetchIndexStats on its corpusPath/health watcher.
// Stub the network module so mounting StatusBar (which instantiates the store)
// never hits fetch.
const fetchIndexStats = vi.fn()
const postIndexRebuild = vi.fn()
vi.mock('../../api/indexStatsApi', () => ({
  fetchIndexStats: (...a: unknown[]) => fetchIndexStats(...a),
  postIndexRebuild: (...a: unknown[]) => postIndexRebuild(...a),
}))

import StatusBar from './StatusBar.vue'
import { useShellStore } from '../../stores/shell'
import { useIndexStatsStore } from '../../stores/indexStats'
import type { IndexStatsEnvelope } from '../../api/indexStatsApi'

function mountBar() {
  return mount(StatusBar, { attachTo: document.body })
}

const BAR = '[data-testid="app-status-bar"]'
const PATH_INPUT = '[data-testid="status-bar-corpus-path"]'
const FILES_BTN = 'button[aria-label="Choose corpus files"]'
const LIST_BTN = '[data-testid="status-bar-list-artifacts"]'
const HEALTH_TRIGGER = '[data-testid="status-bar-health-trigger"]'
const SOURCES_TRIGGER = '[data-testid="status-bar-sources-trigger"]'
const REBUILD_BTN = '[data-testid="status-bar-rebuild-indicator"]'
const BUILD_LABEL = '[data-testid="status-bar-build-label"]'
const SOURCES_DIALOG = '[data-testid="status-bar-sources-dialog"]'
const VERSION_BANNER = '[data-testid="corpus-version-warning-banner"]'

function healthDot(w: ReturnType<typeof mountBar>): string[] {
  return w.get(`${HEALTH_TRIGGER} span[aria-hidden="true"]`).classes()
}

describe('StatusBar', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchIndexStats.mockResolvedValue(null)
  })
  afterEach(() => {
    vi.clearAllMocks()
  })

  // ── Base render ─────────────────────────────────────────────────────────────

  it('always renders the status bar with the corpus-path input and Files button', () => {
    const w = mountBar()
    expect(w.find(BAR).exists()).toBe(true)
    expect(w.find(PATH_INPUT).exists()).toBe(true)
    expect(w.find(FILES_BTN).exists()).toBe(true)
    // Health pill is always present (status reachable or not).
    expect(w.find(HEALTH_TRIGGER).exists()).toBe(true)
  })

  it('renders the build label from the injected build hint', () => {
    const w = mountBar()
    expect(w.get(BUILD_LABEL).text()).toContain('Build:')
  })

  // ── Corpus-path input ↔ store v-model ──────────────────────────────────────

  it('seeds the path input from the store value', () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.corpusPath = '/corpus/root'
    return w.vm.$nextTick().then(() => {
      expect((w.get(PATH_INPUT).element as HTMLInputElement).value).toBe('/corpus/root')
    })
  })

  it('writes the path input back to the store via v-model', async () => {
    const w = mountBar()
    const shell = useShellStore()
    await w.get(PATH_INPUT).setValue('/new/path')
    expect(shell.corpusPath).toBe('/new/path')
  })

  // ── Corpus version warning banner ──────────────────────────────────────────

  it('hides the version-warning banner when none is set', () => {
    const w = mountBar()
    expect(w.find(VERSION_BANNER).exists()).toBe(false)
  })

  it('shows the version-warning banner with the store message', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.corpusVersionWarning = 'Corpus built with an older schema.'
    await w.vm.$nextTick()
    const banner = w.get(VERSION_BANNER)
    expect(banner.attributes('role')).toBe('alert')
    expect(banner.text()).toContain('older schema')
  })

  // ── Health dot branches ─────────────────────────────────────────────────────

  it('health dot is muted before any health status arrives', () => {
    const w = mountBar()
    expect(healthDot(w)).toContain('bg-muted')
  })

  it('health dot is danger when the store has a health error', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthError = 'connection refused'
    await w.vm.$nextTick()
    expect(healthDot(w)).toContain('bg-danger')
  })

  it('health dot is warning when status is reachable but not "ok"', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'degraded'
    await w.vm.$nextTick()
    expect(healthDot(w)).toContain('bg-warning')
  })

  it('health dot is warning when ok but no corpus path is set', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = ''
    await w.vm.$nextTick()
    expect(healthDot(w)).toContain('bg-warning')
  })

  it('health dot is warning when ok + path set but optional APIs are off', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    // feeds/operator/jobs all default false → optionalHealthCapsLimited true.
    await w.vm.$nextTick()
    expect(healthDot(w)).toContain('bg-warning')
  })

  it('health dot is success when ok, path set, and all optional APIs are on', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    shell.operatorConfigApiAvailable = true
    shell.jobsApiAvailable = true
    await w.vm.$nextTick()
    expect(healthDot(w)).toContain('bg-success')
  })

  // ── Conditional action buttons ──────────────────────────────────────────────

  it('hides the List button until health + corpus path are both present', async () => {
    const w = mountBar()
    const shell = useShellStore()
    expect(w.find(LIST_BTN).exists()).toBe(false)
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    expect(w.find(LIST_BTN).exists()).toBe(true)
  })

  it('hides the Configuration button until feeds or operator API is advertised', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    // hasCorpusPath true + health set, but no feeds/operator API yet.
    expect(w.find(SOURCES_TRIGGER).exists()).toBe(false)
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    expect(w.find(SOURCES_TRIGGER).exists()).toBe(true)
  })

  it('hides the rebuild indicator unless the index envelope recommends reindex', async () => {
    const w = mountBar()
    expect(w.find(REBUILD_BTN).exists()).toBe(false)
    const index = useIndexStatsStore()
    index.indexEnvelope = { reindex_recommended: true } as unknown as IndexStatsEnvelope
    await w.vm.$nextTick()
    expect(w.find(REBUILD_BTN).exists()).toBe(true)
  })

  // ── List artifacts action → store + dialog ─────────────────────────────────

  it('List button fetches the artifact list and opens the dialog', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    await w.vm.$nextTick()
    const spy = vi.spyOn(shell, 'fetchArtifactList').mockResolvedValue()
    await w.get(LIST_BTN).trigger('click')
    await w.vm.$nextTick()
    expect(spy).toHaveBeenCalledTimes(1)
    expect((w.get('[data-testid="artifact-list-dialog"]').element as HTMLDialogElement).open).toBe(
      true,
    )
  })

  // ── Health trigger opens the sources dialog on the Health tab ───────────────

  it('Health pill opens the sources dialog on the Health tab', async () => {
    const w = mountBar()
    await w.get(HEALTH_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    const dialog = w.get(SOURCES_DIALOG).element as HTMLDialogElement
    expect(dialog.open).toBe(true)
    // Health tab is active (font-medium highlight).
    expect(w.get('[data-testid="sources-dialog-tab-health"]').classes()).toContain('font-medium')
    expect(w.find('[data-testid="sources-dialog-health-panel"]').exists()).toBe(true)
  })

  it('health panel row reflects "Checking…" before status, OK after', async () => {
    const w = mountBar()
    await w.get(HEALTH_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    const panel = w.get('[data-testid="sources-dialog-health-panel"]')
    expect(panel.text()).toContain('Checking…')
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    shell.operatorConfigApiAvailable = true
    shell.jobsApiAvailable = true
    await w.vm.$nextTick()
    expect(panel.text()).toContain('OK')
  })

  it('health panel API-route rows render once a status is present', async () => {
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.get(HEALTH_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    const panel = w.get('[data-testid="sources-dialog-health-panel"]')
    expect(panel.text()).toContain('Feeds file API')
    expect(panel.text()).toContain('Pipeline jobs API')
  })

  it('Retry health in the health panel calls the store fetchHealth', async () => {
    const w = mountBar()
    const shell = useShellStore()
    const spy = vi.spyOn(shell, 'fetchHealth').mockResolvedValue()
    await w.get(HEALTH_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    // Retry button has text "Retry health".
    const retry = w
      .findAll('[data-testid="sources-dialog-health-panel"] button')
      .find((b) => b.text().includes('Retry health'))
    expect(retry).toBeTruthy()
    await retry!.trigger('click')
    expect(spy).toHaveBeenCalledTimes(1)
  })

  // ── Configuration trigger opens dialog + loads the right default tab ────────

  it('Configuration button opens the dialog on Feeds tab and loads feeds', async () => {
    getFeeds.mockResolvedValue({ path: '/corpus', file_relpath: 'feeds.spec.yaml', feeds: [] })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(getFeeds).toHaveBeenCalledTimes(1)
    expect((w.get(SOURCES_DIALOG).element as HTMLDialogElement).open).toBe(true)
    expect(w.get('[data-testid="sources-dialog-tab-feeds"]').classes()).toContain('font-medium')
  })

  it('Configuration defaults to the Job Configuration tab when feeds API is off', async () => {
    getOperatorConfig.mockResolvedValue({
      corpus_path: '/corpus',
      operator_config_path: '/corpus/operator.yaml',
      content: 'profile: fast\nfoo: bar\n',
      available_profiles: ['fast', 'thorough'],
    })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.operatorConfigApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(getOperatorConfig).toHaveBeenCalledTimes(1)
    expect(w.get('[data-testid="sources-dialog-tab-operator"]').classes()).toContain('font-medium')
  })

  // ── Feeds CRUD inside the dialog ────────────────────────────────────────────

  it('renders the empty-feeds hint when the feeds list is empty', async () => {
    getFeeds.mockResolvedValue({ path: '/corpus', file_relpath: 'feeds.spec.yaml', feeds: [] })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(w.text()).toContain('No feeds yet')
  })

  it('renders loaded feed rows and adds a new feed (persist via putFeeds)', async () => {
    getFeeds.mockResolvedValue({
      path: '/corpus',
      file_relpath: 'feeds.spec.yaml',
      feeds: ['https://a.example/feed.xml'],
    })
    putFeeds.mockResolvedValue({
      path: '/corpus',
      file_relpath: 'feeds.spec.yaml',
      feeds: ['https://a.example/feed.xml', 'https://b.example/feed.xml'],
    })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(w.find('[data-testid="sources-dialog-feeds-row-0"]').exists()).toBe(true)
    await w.get('[data-testid="sources-dialog-feeds-add-url"]').setValue('https://b.example/feed.xml')
    await w.get('[data-testid="sources-dialog-feeds-add-btn"]').trigger('click')
    await w.vm.$nextTick()
    expect(putFeeds).toHaveBeenCalledTimes(1)
    expect(putFeeds.mock.calls[0][1]).toContain('https://b.example/feed.xml')
  })

  it('rejects a duplicate feed URL without calling putFeeds', async () => {
    getFeeds.mockResolvedValue({
      path: '/corpus',
      file_relpath: 'feeds.spec.yaml',
      feeds: ['https://a.example/feed.xml'],
    })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    await w.get('[data-testid="sources-dialog-feeds-add-url"]').setValue('https://a.example/feed.xml')
    await w.get('[data-testid="sources-dialog-feeds-add-btn"]').trigger('click')
    await w.vm.$nextTick()
    expect(putFeeds).not.toHaveBeenCalled()
    expect(w.text()).toContain('already in the list')
  })

  it('surfaces a load error when getFeeds rejects', async () => {
    getFeeds.mockRejectedValue(new Error('feeds boom'))
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(w.text()).toContain('feeds boom')
  })

  it('Raw JSON sub-tab rejects malformed JSON without calling putFeeds', async () => {
    getFeeds.mockResolvedValue({ path: '/corpus', file_relpath: 'feeds.spec.yaml', feeds: [] })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    await w.get('[data-testid="sources-dialog-feeds-panel-json"]').trigger('click')
    await w.vm.$nextTick()
    await w.get('[data-testid="sources-dialog-feeds-textarea"]').setValue('{ not json ')
    await w.get('[data-testid="sources-dialog-feeds-apply-json"]').trigger('click')
    await w.vm.$nextTick()
    expect(putFeeds).not.toHaveBeenCalled()
    expect(w.text()).toContain('Invalid JSON')
  })

  // ── Sources dialog tab switching ────────────────────────────────────────────

  it('switching to the Job Profile tab loads operator config and lists presets', async () => {
    getFeeds.mockResolvedValue({ path: '/corpus', file_relpath: 'feeds.spec.yaml', feeds: [] })
    getOperatorConfig.mockResolvedValue({
      corpus_path: '/corpus',
      operator_config_path: '/corpus/operator.yaml',
      content: 'profile: fast\nfoo: bar\n',
      available_profiles: ['fast', 'thorough'],
    })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    shell.operatorConfigApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    await w.get('[data-testid="sources-dialog-tab-profile"]').trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    expect(getOperatorConfig).toHaveBeenCalledTimes(1)
    const select = w.get('[data-testid="sources-dialog-profile-select"]')
    expect(select.text()).toContain('fast')
    expect(select.text()).toContain('thorough')
    expect((select.element as HTMLSelectElement).value).toBe('fast')
  })

  it('close button resets the operator-load cache and closes the dialog', async () => {
    getFeeds.mockResolvedValue({ path: '/corpus', file_relpath: 'feeds.spec.yaml', feeds: [] })
    const w = mountBar()
    const shell = useShellStore()
    shell.healthStatus = 'ok'
    shell.corpusPath = '/corpus'
    shell.feedsApiAvailable = true
    await w.vm.$nextTick()
    await w.get(SOURCES_TRIGGER).trigger('click')
    await w.vm.$nextTick()
    await w.vm.$nextTick()
    const dialog = w.get(SOURCES_DIALOG).element as HTMLDialogElement
    expect(dialog.open).toBe(true)
    await w.get('[data-testid="sources-dialog-close"]').trigger('click')
    expect(dialog.open).toBe(false)
  })

  // ── Local file picker → artifacts store + emit ─────────────────────────────

  it('local file input change loads files and emits localArtifactsLoaded', async () => {
    const w = mountBar()
    // No files selected → artifacts store leaves displayArtifact null → false.
    const input = w.get('[data-testid="status-bar-local-file-input"]')
    await input.trigger('change')
    await w.vm.$nextTick()
    const emitted = w.emitted('localArtifactsLoaded')
    expect(emitted).toBeTruthy()
    expect(emitted![0]).toEqual([false])
  })

  // ── Rebuild indicator opens the index dialog ───────────────────────────────

  it('rebuild indicator opens the vector-index dialog', async () => {
    const w = mountBar()
    const index = useIndexStatsStore()
    index.indexEnvelope = { reindex_recommended: true } as unknown as IndexStatsEnvelope
    await w.vm.$nextTick()
    await w.get(REBUILD_BTN).trigger('click')
    await w.vm.$nextTick()
    const indexDialog = w
      .findAll('dialog')
      .find((d) => d.text().includes('Vector index'))
    expect(indexDialog).toBeTruthy()
    expect((indexDialog!.element as HTMLDialogElement).open).toBe(true)
  })
})
