/**
 * Command palette — command mode entries (#1259-1).
 *
 * Palette command mode surfaces global nav + session actions above the
 * corpus-search hit list. This module defines the command entries as
 * pure data (id, label, category, keywords, run-handle) so
 * ``CommandPalette.vue`` stays a thin renderer.
 *
 * Trigger contract (matches ``matchCommands`` below):
 *   - Input starts with ``>`` → command mode ONLY (input after ``>`` is
 *     the query; empty input after ``>`` shows all commands).
 *   - Otherwise → commands appear ABOVE hits when the query fuzz-matches
 *     a command label or a keyword. Corpus-hit fetch still runs in
 *     parallel.
 */

export type CommandCategory =
  | 'nav'
  | 'session'
  | 'modal'
  | 'admin'
  | 'operator'

export interface PaletteCommand {
  id: string
  label: string
  category: CommandCategory
  /** Extra keywords to match against — synonyms, mnemonics, section. */
  keywords: string[]
  /** Optional short hint rendered next to the label (e.g. ``⌘K``). */
  shortcut?: string | null
  /** True when the command should only appear for admin users. */
  adminOnly?: boolean
  /** Invoked when the user selects the command. */
  run: () => void | Promise<void>
}

/** Mainly for the "starts with ``>``" test — one place, one rule. */
export function stripCommandPrefix(input: string): {
  isCommandMode: boolean
  query: string
} {
  const trimmed = input.trim()
  if (trimmed.startsWith('>')) {
    return { isCommandMode: true, query: trimmed.slice(1).trim() }
  }
  return { isCommandMode: false, query: trimmed }
}

/**
 * Rank commands against a query. Simple weighted match:
 *   - label starts-with (3)
 *   - label contains (2)
 *   - keyword contains (1)
 *   - fuzzy substring (0.5 — every query char appears in order)
 * Commands with score 0 are dropped. Ties break by original order so the
 * caller's list dictates default ordering (nav first, then session, …).
 */
export function matchCommands(
  commands: readonly PaletteCommand[],
  rawQuery: string,
  { limit = 8 }: { limit?: number } = {},
): PaletteCommand[] {
  const q = rawQuery.trim().toLowerCase()
  if (!q) return commands.slice(0, limit)

  const scored: Array<{ cmd: PaletteCommand; score: number; idx: number }> = []
  commands.forEach((cmd, idx) => {
    const label = cmd.label.toLowerCase()
    let score = 0
    if (label.startsWith(q)) score = 3
    else if (label.includes(q)) score = 2
    else if (cmd.keywords.some((k) => k.toLowerCase().includes(q))) score = 1
    else if (fuzzySubstring(label, q)) score = 0.5
    if (score > 0) scored.push({ cmd, score, idx })
  })
  scored.sort((a, b) => (b.score - a.score) || (a.idx - b.idx))
  return scored.slice(0, limit).map((s) => s.cmd)
}

function fuzzySubstring(haystack: string, needle: string): boolean {
  let i = 0
  for (const ch of haystack) {
    if (ch === needle[i]) i += 1
    if (i === needle.length) return true
  }
  return false
}

export type MainTabId = 'digest' | 'library' | 'search' | 'graph' | 'dashboard'

export interface PaletteCommandDeps {
  /** Switch main tab. */
  goTab: (tab: MainTabId) => void
  /** Save the current query into ``search.savedQueries``. */
  saveCurrentQuery: () => void
  /** Clear the current search (query + results). */
  clearSearch: () => void
  /** Reset chip filters to defaults (topK / doc types / topic / speaker / etc.). */
  resetFilters: () => void
  /** Cycle theme (auto → dark → light → auto). */
  cycleTheme: () => void
  /** Copy the current corpus path to clipboard. */
  copyCorpusPath: () => Promise<void> | void
  /** Open Configuration dialog. */
  openConfiguration: () => void
  /** Open Health dialog. */
  openHealth: () => void
  /** Trigger a corpus re-index (admin only). */
  rebuildIndex: () => void
  /**
   * #1259-4 operator kickers — run a server-side result-set operator
   * (Cluster / Consensus) over the most recent query. Silent no-op when
   * there is no recent query yet.
   */
  runOperatorOnLastQuery: (op: 'cluster' | 'consensus') => Promise<void> | void
  /**
   * #1259-4 timeline kicker — open the Search tab on the most recent
   * query and toggle the Timeline operator on. Silent no-op when there
   * is no recent query.
   */
  openTimelineForLastQuery: () => Promise<void> | void
  /**
   * #1259-4 compare kicker — open the Search tab on the most recent
   * query and pop the Compare picker. Silent no-op when there is no
   * recent query.
   */
  openCompareForLastQuery: () => Promise<void> | void
  /** True when the signed-in user is an admin. */
  isAdmin: boolean
}

/** Build the full command catalog for the current session. */
export function buildPaletteCommands(deps: PaletteCommandDeps): PaletteCommand[] {
  const cmds: PaletteCommand[] = [
    // Navigation
    {
      id: 'nav.digest',
      label: 'Go to Digest',
      category: 'nav',
      keywords: ['digest', 'home', 'topics', '1'],
      shortcut: '1',
      run: () => deps.goTab('digest'),
    },
    {
      id: 'nav.library',
      label: 'Go to Library',
      category: 'nav',
      keywords: ['library', 'episodes', 'catalog', '2'],
      shortcut: '2',
      run: () => deps.goTab('library'),
    },
    {
      id: 'nav.search',
      label: 'Go to Search',
      category: 'nav',
      keywords: ['search', 'query', 'workspace', '3'],
      shortcut: '3',
      run: () => deps.goTab('search'),
    },
    {
      id: 'nav.graph',
      label: 'Go to Graph',
      category: 'nav',
      keywords: ['graph', 'canvas', 'network', '4'],
      shortcut: '4',
      run: () => deps.goTab('graph'),
    },
    {
      id: 'nav.dashboard',
      label: 'Go to Dashboard',
      category: 'nav',
      keywords: ['dashboard', 'analytics', 'intelligence', '5'],
      shortcut: '5',
      run: () => deps.goTab('dashboard'),
    },
    // Session actions
    {
      id: 'session.save-query',
      label: 'Save current query',
      category: 'session',
      keywords: ['save', 'pin', 'query', 'bookmark'],
      run: () => deps.saveCurrentQuery(),
    },
    {
      id: 'session.clear-search',
      label: 'Clear search',
      category: 'session',
      keywords: ['clear', 'reset', 'empty', 'query'],
      run: () => deps.clearSearch(),
    },
    {
      id: 'session.reset-filters',
      label: 'Reset filters',
      category: 'session',
      keywords: ['reset', 'filters', 'chips', 'defaults'],
      run: () => deps.resetFilters(),
    },
    {
      id: 'session.toggle-theme',
      label: 'Toggle theme',
      category: 'session',
      keywords: ['theme', 'dark', 'light', 'auto', 'colors', 'appearance'],
      run: () => deps.cycleTheme(),
    },
    {
      id: 'session.copy-corpus-path',
      label: 'Copy corpus path',
      category: 'session',
      keywords: ['corpus', 'path', 'clipboard', 'copy'],
      run: () => deps.copyCorpusPath(),
    },
    // Modal opens
    {
      id: 'modal.configuration',
      label: 'Open Configuration',
      category: 'modal',
      keywords: ['configuration', 'settings', 'feeds', 'operator', 'yaml'],
      run: () => deps.openConfiguration(),
    },
    {
      id: 'modal.health',
      label: 'Open Health',
      category: 'modal',
      keywords: ['health', 'status', 'diagnostics', 'server'],
      run: () => deps.openHealth(),
    },
    // Admin
    {
      id: 'admin.rebuild-index',
      label: 'Rebuild index',
      category: 'admin',
      keywords: ['index', 'reindex', 'rebuild', 'vector', 'lance'],
      adminOnly: true,
      run: () => deps.rebuildIndex(),
    },
    // #1259-4 operator kickers
    {
      id: 'operator.cluster-last',
      label: 'Cluster last query',
      category: 'operator',
      keywords: ['cluster', 'group', 'theme', 'operator', 's4b', 'last'],
      run: () => deps.runOperatorOnLastQuery('cluster'),
    },
    {
      id: 'operator.consensus-last',
      label: 'Consensus last query',
      category: 'operator',
      keywords: ['consensus', 'corroboration', 'agreement', 'operator', 's4b'],
      run: () => deps.runOperatorOnLastQuery('consensus'),
    },
    {
      id: 'operator.timeline-last',
      label: 'Timeline last query',
      category: 'operator',
      keywords: ['timeline', 'histogram', 'when', 'months', 'operator', 's4a'],
      run: () => deps.openTimelineForLastQuery(),
    },
    {
      id: 'operator.compare-last',
      label: 'Compare on last query',
      category: 'operator',
      keywords: ['compare', 'subjects', 'operator', 's8', 'briefing'],
      run: () => deps.openCompareForLastQuery(),
    },
  ]
  return cmds.filter((c) => !c.adminOnly || deps.isAdmin)
}
