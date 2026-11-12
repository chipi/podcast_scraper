/**
 * Extension points: places where a build of this viewer can add features without editing the
 * components that host them. This build registers nothing, so every list is empty and the viewer
 * renders its own features only.
 *
 * A downstream build registers its entries in a module that runs before `main.ts` (imported ahead
 * of it from its own entry), the same way installed backend extensions add routes to the server.
 * Each point is an ordered list; entries carry a unique `id` and an optional `order` (ascending,
 * default 0). Hosts read the list when they render.
 */
import { shallowReactive, type Component } from 'vue'

export interface Entry {
  id: string
  order?: number
}

export interface ExtensionPoint<T extends Entry> {
  register(entry: T): void
  list(): T[]
  /** Tests only: forget every registered entry. */
  reset(): void
}

export function extensionPoint<T extends Entry>(name: string): ExtensionPoint<T> {
  const entries = shallowReactive<T[]>([])
  return {
    register(entry) {
      if (entries.some((e) => e.id === entry.id)) {
        throw new Error(`extension point ${name}: "${entry.id}" is already registered`)
      }
      entries.push(entry)
    },
    list() {
      return [...entries].sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    },
    reset() {
      entries.splice(0)
    },
  }
}

/**
 * A card on the dashboard's Intelligence tab. `region` picks the row: `overview` is the
 * full-width row under the snapshot, `topics` the topic grid. `span` is the card's width in the
 * 4-column grid (topics only). The dashboard forwards a card's `go-graph` event.
 */
export interface DashboardCard extends Entry {
  region: 'overview' | 'topics'
  span?: 1 | 2 | 4
  component: Component
  props?: Record<string, unknown>
}

export const dashboardCards = extensionPoint<DashboardCard>('dashboardCards')

/** Extra query parameters for every `/api/search` request the search store sends. */
export interface SearchRequestParams extends Entry {
  params(): Record<string, string> | undefined
}

export const searchRequestParams = extensionPoint<SearchRequestParams>('searchRequestParams')

/**
 * Told about each search: `onResponse` with the raw response body of a successful run (fields
 * this viewer does not read included), `onClear` when results are cleared or a new run starts,
 * `onFiltersReset` when the user resets the filters.
 */
export interface SearchListener extends Entry {
  onResponse?(body: Record<string, unknown>): void
  onClear?(): void
  /** The user reset every search filter to its default. */
  onFiltersReset?(): void
}

export const searchListeners = extensionPoint<SearchListener>('searchListeners')

/**
 * A result-set operator chip on the operator bar. The viewer's own chips are timeline (order
 * 10), graph (20) and compare (30). `activate` runs when the chip turns on; `panel` renders under
 * the bar while it is on; `busy` shows the chip's loading state.
 */
export interface SearchOperator extends Entry {
  label: string
  title?: string
  busy?(): boolean
  /** The operator's last error; the bar shows it under the chips whichever chip is open. */
  error?(): string | null
  activate?(): void
  panel?: Component
  /** Adds the palette command `operator.<id>-last`: load the last query, then open this operator. */
  palette?: { label: string; keywords?: string[] }
}

export const searchOperators = extensionPoint<SearchOperator>('searchOperators')

/**
 * A section of the search panel: `header` sits in the title row, `above-results` above the
 * operator bar and hit cards.
 */
export interface SearchPanelSection extends Entry {
  region: 'header' | 'above-results'
  component: Component
}

export const searchPanelSections = extensionPoint<SearchPanelSection>('searchPanelSections')

/** A chip on the search filter bar, after the viewer's own. */
export interface SearchFilterChip extends Entry {
  component: Component
}

export const searchFilterChips = extensionPoint<SearchFilterChip>('searchFilterChips')

/** A section at the bottom of each search result card; rendered with the card's `hit` prop. */
export interface ResultCardSection extends Entry {
  component: Component
}

export const resultCardSections = extensionPoint<ResultCardSection>('resultCardSections')

/**
 * A section of the episode detail panel, rendered with `detail` (the episode detail payload) and
 * `corpusPath` props: `summary-head` under the summary title, `end` after the panel's own lists
 * (the details tab), `enrichment` in the Enrichment tab.
 */
export interface EpisodeDetailSection extends Entry {
  region: 'summary-head' | 'end' | 'enrichment'
  component: Component
}

export const episodeDetailSections = extensionPoint<EpisodeDetailSection>('episodeDetailSections')

/**
 * A section of the show rail. `signals` renders in the show-level signals block with `signals`
 * (the feed-signals payload, fields this viewer does not read included) and `feedId` props;
 * `hasContent` makes the block show for a show with no signal of the viewer's own. `episode-row`
 * renders under each episode row with `episode` (the list item) and `index` props.
 */
export interface ShowRailSection extends Entry {
  region: 'signals' | 'episode-row'
  component: Component
  hasContent?(signals: Record<string, unknown>): boolean
}

export const showRailSections = extensionPoint<ShowRailSection>('showRailSections')

/**
 * A section of the digest. `band` renders at the end of each topic band with `band` (the band
 * payload) and `topicId` props; `row` renders under each recent episode row with a `row` prop.
 */
export interface DigestSection extends Entry {
  region: 'band' | 'row'
  component: Component
}

export const digestSections = extensionPoint<DigestSection>('digestSections')

/**
 * A section of a person's view, rendered with `personId` and `view` props: `header` under the
 * name and role, `profile` at the end of the profile, `positions` at the end of the positions
 * view, `full` at the end of the full view.
 */
export interface PersonProfileSection extends Entry {
  region: 'header' | 'profile' | 'positions' | 'full'
  component: Component
}

export const personProfileSections = extensionPoint<PersonProfileSection>('personProfileSections')

/** A command in the command palette. */
export interface PaletteCommandEntry extends Entry {
  label: string
  category: string
  keywords?: string[]
  run(): void | Promise<void>
}

export const paletteCommands = extensionPoint<PaletteCommandEntry>('paletteCommands')

/**
 * A tab in the configuration dialog, after the viewer's own. `visible` hides it (for example
 * when the signed-in user lacks an entitlement); the panel gets `active`.
 */
export interface ConfigTab extends Entry {
  label: string
  visible?(): boolean
  component: Component
}

export const configTabs = extensionPoint<ConfigTab>('configTabs')

/**
 * A section on the Admin tab, between the user table and the release settings. For admin surfaces
 * whose backend routes come from an installed extension, so a build without it shows no panel
 * rather than one that cannot load.
 */
export interface AdminPanel extends Entry {
  component: Component
}

export const adminPanels = extensionPoint<AdminPanel>('adminPanels')

/** A panel on the graph tab: `notice` lines under the error strip, `above-canvas` over the graph. */
export interface GraphPanel extends Entry {
  region: 'notice' | 'above-canvas'
  component: Component
}

export const graphPanels = extensionPoint<GraphPanel>('graphPanels')

/** A chip in the graph bottom bar, after the lenses chip. */
export interface GraphBottomBarChip extends Entry {
  component: Component
}

export const graphBottomBarChips = extensionPoint<GraphBottomBarChip>('graphBottomBarChips')

/** A chip in the graph filter bar, after the viewer's filter chips (before the divider). */
export interface GraphFilterBarChip extends Entry {
  component: Component
}

export const graphFilterBarChips = extensionPoint<GraphFilterBarChip>('graphFilterBarChips')

/** A banner across the top of the app, under the viewer's own alerts. Renders itself or nothing. */
export interface AppBanner extends Entry {
  component: Component
}

export const appBanners = extensionPoint<AppBanner>('appBanners')

/**
 * A graph lens: a checkbox in the lenses popover, a flag persisted with the viewer's own lens
 * flags, and a canvas overlay. `available` gates the checkbox for a corpus; `apply` paints (it is
 * called after every layout pass and when the flag or `watchSource` changes); `clear` removes the
 * paint. `legacyKeys` are older storage names read once, so a renamed lens keeps a user's choice.
 */
export interface GraphLens extends Entry {
  label: string
  description: string
  testid: string
  defaultOn?: boolean
  legacyKeys?: string[]
  available?(corpusPath: string): boolean | Promise<boolean>
  apply(core: import('cytoscape').Core, corpusPath: string): void | Promise<void>
  clear(core: import('cytoscape').Core): void
  watchSource?(): unknown
}

export const graphLenses = extensionPoint<GraphLens>('graphLenses')

/**
 * Cytoscape style rules for the main graph and its previews. `base` rules go right after the
 * generic node rule (selection and state rules declared later still win); `end` rules go last.
 */
export interface GraphStylesheetEntry extends Entry {
  layer: 'base' | 'end'
  rules(options: { compact: boolean }): Record<string, unknown>[]
}

export const graphStylesheet = extensionPoint<GraphStylesheetEntry>('graphStylesheet')

/** What a canvas plugin gets: the live Cytoscape core (null between graphs) and the canvas's guard. */
export interface GraphCanvasContext {
  core(): import('cytoscape').Core | null
  /** Runs `fn` on the live core inside the canvas's error guard (a styling error never aborts a redraw). */
  guarded(label: string, fn: (core: import('cytoscape').Core) => void): void
}

/**
 * Behaviour added to the main graph canvas. Every hook is optional.
 *
 * - `setup` runs when the canvas mounts (watchers belong here) and may return a disposer.
 * - `onLayoutPass` runs at the end of every layout pass; `afterRestore` after selection dimming is
 *   restored.
 * - `onNodeTap`, `onNodeOneTap` and `onNodeDblTap` return true to take over the gesture (the viewer
 *   then does nothing more with it); `onEmptyTap` runs on a tap on the empty canvas.
 * - `onKeydown` returns true when it handled the key.
 * - `onFocusTargetMissing` runs when a pending focus id is not on the canvas (the target may be
 *   hidden by the plugin's own view).
 * - `onInteractionReset` runs when the canvas clears its interaction state.
 * - `bridgeTitle` gives the hover title for a bridge node (first non-null wins).
 * - `overlay` renders over the canvas; `topicLikeTypes` are node types placed and banded like Topic.
 */
export interface GraphCanvasPlugin extends Entry {
  setup?(ctx: GraphCanvasContext): void | (() => void)
  onLayoutPass?(core: import('cytoscape').Core): void
  afterRestore?(core: import('cytoscape').Core): void
  onNodeTap?(core: import('cytoscape').Core, node: import('cytoscape').NodeSingular): boolean
  onNodeOneTap?(node: import('cytoscape').NodeSingular): boolean
  onNodeDblTap?(node: import('cytoscape').NodeSingular): boolean
  onEmptyTap?(core: import('cytoscape').Core): void
  onKeydown?(e: KeyboardEvent): boolean
  onFocusTargetMissing?(rawId: string): void
  onInteractionReset?(): void
  bridgeTitle?(node: import('cytoscape').NodeSingular): string | null
  overlay?: Component
  topicLikeTypes?: string[]
}

export const graphCanvasPlugins = extensionPoint<GraphCanvasPlugin>('graphCanvasPlugins')

/** Rewrites the merged graph the viewer displays, in `order` (each gets the previous one's output). */
export interface ArtifactTransform extends Entry {
  transform(
    artifact: import('../types/artifact').ParsedArtifact | null,
  ): import('../types/artifact').ParsedArtifact | null
}

export const artifactTransforms = extensionPoint<ArtifactTransform>('artifactTransforms')

/**
 * Told about the artifacts store's loads. `onLoadStart` runs as a graph load begins
 * (`preserveExpansion` true when the selection only grew or shrank); `beforeAssign` is awaited
 * after the files are read and before the graph swaps to them; `onReset` runs when the selection
 * is cleared (`clear`) or replaced by local files (`local-files`); `onCorpusPathChange` when the
 * corpus path changes.
 */
export interface ArtifactLoadHook extends Entry {
  onLoadStart?(opts: { preserveExpansion: boolean }): void
  beforeAssign?(): Promise<void>
  onReset?(reason: 'clear' | 'local-files'): void
  onCorpusPathChange?(next: string, previous: string): void
}

export const artifactLoadHooks = extensionPoint<ArtifactLoadHook>('artifactLoadHooks')

/**
 * A different graph for the canvas while `active()` is true and `artifact()` is not null (the first
 * such override wins). `forceVisibleTypes` stay visible whatever the type filter says.
 */
export interface DisplayOverride extends Entry {
  active(): boolean
  artifact(): import('../types/artifact').ParsedArtifact | null
  forceVisibleTypes?: string[]
}

export const displayOverrides = extensionPoint<DisplayOverride>('displayOverrides')

/**
 * The neighbourhood shown when a node is focused (ego view). Returns null to leave it to the next
 * entry; the viewer's own one-hop view is the fallback.
 */
export interface GraphEgoView extends Entry {
  view(
    base: import('../types/artifact').ParsedArtifact,
    focusId: string | null,
  ): import('../types/artifact').ParsedArtifact | null
}

export const graphEgoViews = extensionPoint<GraphEgoView>('graphEgoViews')

/**
 * Told about the graph tab's loads. `beforeSync` runs when the tab opens on a target node and may
 * build the graph itself (return true and the default corpus sync is skipped); `afterSync` runs
 * after the default sync; `beforeCorpusGraphSync` before the default slice is chosen;
 * `onGraphShown` when the graph tab shows a loaded graph.
 */
export interface GraphActivationHook extends Entry {
  beforeSync?(target: string): Promise<boolean>
  afterSync?(target: string | undefined): Promise<void>
  beforeCorpusGraphSync?(): Promise<void>
  onGraphShown?(): void | Promise<void>
}

export const graphActivationHooks = extensionPoint<GraphActivationHook>('graphActivationHooks')

/** Extra score for an episode (by artifact path stem) when the default graph slice is chosen. */
export interface EpisodeScoreBonus extends Entry {
  bonus(stem: string): number
}

export const episodeScoreBonuses = extensionPoint<EpisodeScoreBonus>('episodeScoreBonuses')

/** The node a detail panel shows, as extensions see it. */
export interface NodeDetailContext {
  nodeId: string
  nodeType: string
  isTopicNode: boolean
  viewArtifact: import('../types/artifact').ParsedArtifact | null
  embedInRail: boolean
  /** Ask the panel's host to show the graph (the panel's own `go-graph`). */
  goGraph(): void
}

/**
 * Additions to the node detail panel. `knowsNode` shows the panel for a node the graph slice does
 * not hold; `title` overrides the header title; `avatarGroup` overrides the header glyph group; `sections` render with a `ctx` prop
 * (`header` under the title, `overview` after the topic search shortcuts, `after-aliases` after
 * the topic aliases); `timeline` returns the topic ids whose merged timeline the Timeline tab
 * shows for this node, `'hidden'` for no Timeline tab, or null for the default (a topic's own
 * timeline), and `timelineToggle`
 * adds an on/off button that `timeline` receives; `connections` replaces the neighbour list (and
 * the minimap's neighbourhood) for this node.
 */
export interface NodeDetailExtension extends Entry {
  knowsNode?(nodeId: string): boolean
  /** The panel title for this node (first non-null wins over the node's own label). */
  title?(ctx: NodeDetailContext): string | null
  avatarGroup?(ctx: NodeDetailContext): string | null
  sections?: Array<{ region: 'header' | 'overview' | 'after-aliases'; component: Component }>
  timeline?(ctx: NodeDetailContext, toggled: boolean): string[] | 'hidden' | null
  timelineToggle?(ctx: NodeDetailContext): {
    label: string
    titleOn: string
    titleOff: string
    testid: string
    /** CSS colour for the button while on (text and border). */
    color?: string
  } | null
  connections?(ctx: NodeDetailContext): {
    rows: import('../utils/graphNeighbors').GraphNeighborRow[]
    emptyHint?: string
    compound?: import('../utils/parsing').CompoundNeighborhood
  } | null
}

export const nodeDetailExtensions = extensionPoint<NodeDetailExtension>('nodeDetailExtensions')

/**
 * Topic groupings a build knows (one provider usually). `themeMemberTopicIds` are highlighted in
 * topic pills; `groupTitle` marks a topic as grouped (tooltip text) on person pages;
 * `episodeIdsForTopic` is tried before the corpus timeline when a topic is handed to the graph;
 * `nodeKind` names a node the provider owns (node rail); `ensureLoaded` runs when a view that
 * shows groupings opens.
 */
export interface TopicGroupingProvider extends Entry {
  themeMemberTopicIds?(): Set<string>
  groupTitle?(topicId: string): string | undefined
  episodeIdsForTopic?(topicId: string): string[]
  nodeKind?(nodeId: string): string | undefined
  ensureLoaded?(): void
}

export const topicGrouping = extensionPoint<TopicGroupingProvider>('topicGrouping')

/** A badge after a digest topic band's label; rendered with a `topicId` prop. */
export interface TopicBadge extends Entry {
  component: Component
}

export const topicBadges = extensionPoint<TopicBadge>('topicBadges')

/**
 * A row in a topic's or person's enrichment section, from the same per-entity signals payload the
 * section loads (`/api/corpus/entity-signals`, fields this viewer does not read included).
 * `placement` puts it before the viewer's rows (`first`) or after them (`last`); `hasContent`
 * counts it towards showing the Enrichment tab. Rendered with `nodeId` and `signals` props.
 */
export interface NodeEnrichmentRow extends Entry {
  kind: 'topic' | 'person'
  placement: 'first' | 'last'
  hasContent(signals: Record<string, unknown>, nodeId: string): boolean
  component: Component
}

export const nodeEnrichmentRows = extensionPoint<NodeEnrichmentRow>('nodeEnrichmentRows')
