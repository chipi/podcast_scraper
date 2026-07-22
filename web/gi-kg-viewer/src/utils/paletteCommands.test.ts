import { describe, expect, it, vi } from 'vitest'
import {
  buildPaletteCommands,
  matchCommands,
  stripCommandPrefix,
  type PaletteCommand,
  type PaletteCommandDeps,
} from './paletteCommands'

function noopDeps(overrides: Partial<PaletteCommandDeps> = {}): PaletteCommandDeps {
  return {
    goTab: vi.fn(),
    saveCurrentQuery: vi.fn(),
    clearSearch: vi.fn(),
    resetFilters: vi.fn(),
    cycleTheme: vi.fn(),
    copyCorpusPath: vi.fn(),
    openConfiguration: vi.fn(),
    openHealth: vi.fn(),
    rebuildIndex: vi.fn(),
    isAdmin: false,
    ...overrides,
  }
}

describe('stripCommandPrefix', () => {
  it('extracts the query after a leading > and marks command mode', () => {
    expect(stripCommandPrefix('> save')).toEqual({ isCommandMode: true, query: 'save' })
    expect(stripCommandPrefix('>save')).toEqual({ isCommandMode: true, query: 'save' })
    expect(stripCommandPrefix('>')).toEqual({ isCommandMode: true, query: '' })
    expect(stripCommandPrefix('   >  reset  ')).toEqual({
      isCommandMode: true,
      query: 'reset',
    })
  })

  it('leaves plain queries untouched', () => {
    expect(stripCommandPrefix('iran')).toEqual({ isCommandMode: false, query: 'iran' })
    expect(stripCommandPrefix('')).toEqual({ isCommandMode: false, query: '' })
  })
})

describe('matchCommands', () => {
  const catalog: PaletteCommand[] = buildPaletteCommands(noopDeps())

  it('returns the full catalog when the query is empty (default order)', () => {
    const out = matchCommands(catalog, '', { limit: 100 })
    expect(out.map((c) => c.id).slice(0, 3)).toEqual([
      'nav.digest',
      'nav.library',
      'nav.search',
    ])
    // Admin-only entry omitted for non-admin.
    expect(out.map((c) => c.id)).not.toContain('admin.rebuild-index')
  })

  it('ranks label prefix > label contains > keyword > fuzzy', () => {
    // "graph" — nav.graph starts with "graph" → top.
    expect(matchCommands(catalog, 'graph', { limit: 3 })[0].id).toBe('nav.graph')
    // "reset" — session.reset-filters label contains → top.
    expect(matchCommands(catalog, 'reset', { limit: 3 })[0].id).toBe(
      'session.reset-filters',
    )
    // "dark" — no label contains; keyword match on toggle-theme.
    expect(matchCommands(catalog, 'dark', { limit: 3 })[0].id).toBe(
      'session.toggle-theme',
    )
  })

  it('drops commands with zero-match', () => {
    const out = matchCommands(catalog, 'xxnothingxx', { limit: 100 })
    expect(out).toEqual([])
  })

  it('caps by the limit', () => {
    const out = matchCommands(catalog, '', { limit: 3 })
    expect(out).toHaveLength(3)
  })

  it('surfaces admin commands only when isAdmin is true', () => {
    const adminCatalog = buildPaletteCommands(noopDeps({ isAdmin: true }))
    const ids = matchCommands(adminCatalog, 'index', { limit: 100 }).map((c) => c.id)
    expect(ids).toContain('admin.rebuild-index')
  })
})

describe('buildPaletteCommands', () => {
  it('wires each dep through the correct command run() handler', async () => {
    const deps = noopDeps({ isAdmin: true })
    const catalog = buildPaletteCommands(deps)
    const byId = new Map(catalog.map((c) => [c.id, c]))

    await byId.get('nav.digest')!.run()
    expect(deps.goTab).toHaveBeenCalledWith('digest')

    await byId.get('session.save-query')!.run()
    expect(deps.saveCurrentQuery).toHaveBeenCalled()

    await byId.get('session.clear-search')!.run()
    expect(deps.clearSearch).toHaveBeenCalled()

    await byId.get('session.reset-filters')!.run()
    expect(deps.resetFilters).toHaveBeenCalled()

    await byId.get('session.toggle-theme')!.run()
    expect(deps.cycleTheme).toHaveBeenCalled()

    await byId.get('session.copy-corpus-path')!.run()
    expect(deps.copyCorpusPath).toHaveBeenCalled()

    await byId.get('modal.configuration')!.run()
    expect(deps.openConfiguration).toHaveBeenCalled()

    await byId.get('modal.health')!.run()
    expect(deps.openHealth).toHaveBeenCalled()

    await byId.get('admin.rebuild-index')!.run()
    expect(deps.rebuildIndex).toHaveBeenCalled()
  })

  it('excludes admin-only commands when isAdmin is false', () => {
    const catalog = buildPaletteCommands(noopDeps({ isAdmin: false }))
    expect(catalog.map((c) => c.id)).not.toContain('admin.rebuild-index')
  })
})
