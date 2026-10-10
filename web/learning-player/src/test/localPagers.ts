/**
 * The server's paging, done on a full list — the unit tests' fake server.
 *
 * These lived in services/api.ts as the fallback for a server that predated paging (a 1.0.2-era
 * server answered the whole list, with no `total`). That server is gone (removed 2026-10-10,
 * docs/wip/TODO-remove-pre-1.0.3-compat.md), but the specs still answer the paged reads from their
 * own full-list spies through exactly this code (`apiViaSpies.ts`), so it moved here unchanged.
 */
import type {
  FavoritesPageQuery,
  HighlightsPage,
  HighlightsPageQuery,
  NotesPage,
  NotesPageQuery,
  PodcastsPage,
  PodcastsQuery,
} from "../services/api"
import type {
  CollectionDetail,
  FavoriteKind,
  FavoritesResponse,
  Highlight,
  Note,
  Podcast,
  ResurfacingItem,
  ResurfacingResponse,
} from "../services/types"

export function pageFavoritesLocally(all: FavoritesResponse, query: FavoritesPageQuery): FavoritesResponse {
  const needle = (query.q ?? "").trim().toLocaleLowerCase()
  const has = (...texts: (string | null | undefined)[]) =>
    !needle || texts.some((t) => (t ?? "").toLocaleLowerCase().includes(needle))
  const colourOk = (c?: string | null) => !query.color || c === query.color
  const eps = all.episodes.filter((e) => colourOk(e.color) && has(e.title, e.podcast_title))
  const ents = (all.entities ?? []).filter((e) => colourOk(e.color) && has(e.label))
  const counts: Partial<Record<FavoriteKind, number>> = { episode: eps.length }
  for (const e of ents) counts[e.kind] = (counts[e.kind] ?? 0) + 1
  const byTitle = query.sort === "title"
  const start = query.offset ?? 0
  const end = start + query.limit
  if (query.kind === "episode") {
    const list = byTitle ? [...eps].sort((a, b) => a.title.localeCompare(b.title)) : eps
    return { episodes: list.slice(start, end), entities: [], total: list.length, counts }
  }
  const list = query.kind ? ents.filter((e) => e.kind === query.kind) : ents
  const sorted = byTitle ? [...list].sort((a, b) => a.label.localeCompare(b.label)) : list
  return { episodes: [], entities: sorted.slice(start, end), total: sorted.length, counts }
}

export function pagePodcastsLocally(all: Podcast[], query: PodcastsQuery): PodcastsPage {
  const ids = new Set(query.feedIds ?? [])
  const words = (query.q ?? "").toLowerCase().split(/\s+/).filter(Boolean)
  const title = (p: Podcast) => (p.title ?? p.feed_id).toLowerCase()
  const hay = (p: Podcast) => [title(p), ...(p.authors ?? [])].join(" ").toLowerCase()
  let list = all.filter(
    (p) =>
      p.feed_id &&
      (!ids.size || ids.has(p.feed_id)) &&
      (!query.category || p.category === query.category) &&
      words.every((w) => hay(p).includes(w)),
  )
  const byTitle = (a: Podcast, b: Podcast) => title(a).localeCompare(title(b))
  if (query.sort === "za") list = [...list].sort((a, b) => byTitle(b, a))
  else if (query.sort === "az" || query.sort === "trending") list = [...list].sort(byTitle)
  else {
    const dated = list.filter((p) => p.last_updated)
    const undated = list.filter((p) => !p.last_updated).sort(byTitle)
    dated.sort((a, b) => {
      const c = (a.last_updated ?? "").localeCompare(b.last_updated ?? "") || byTitle(a, b)
      return query.sort === "oldest" ? c : -c
    })
    list = [...dated, ...undated]
  }
  const start = query.offset ?? 0
  return {
    items: list.slice(start, start + query.limit),
    total: list.length,
    categories: [...new Set(all.map((p) => p.category).filter((c): c is string => !!c))].sort(),
  }
}

export function pageHighlightsLocally(
  all: Highlight[],
  notes: Note[],
  query: HighlightsPageQuery,
): HighlightsPage {
  const needle = (query.q ?? "").trim().toLocaleLowerCase()
  const has = (...texts: (string | null | undefined)[]) =>
    !needle || texts.some((t) => (t ?? "").toLocaleLowerCase().includes(needle))
  const groups = new Map<string, Highlight[]>()
  let total = 0
  for (const h of all) {
    if (query.episode && h.episode_slug !== query.episode) continue
    if (query.color && h.color !== query.color) continue
    if (query.muted && !h.retired) continue
    if (!has(h.quote_text, h.speaker)) continue
    total++
    groups.set(h.episode_slug, [...(groups.get(h.episode_slug) ?? []), h])
  }
  const newest = (hs: Highlight[]) => Math.max(...hs.map((h) => h.created_at ?? 0))
  for (const hs of groups.values()) hs.sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
  // An older server has no titles to sort by here; A–Z falls back to the slug.
  const order = [...groups.keys()].sort((a, b) =>
    query.sort === "title" ? a.localeCompare(b) : newest(groups.get(b)!) - newest(groups.get(a)!),
  )
  const start = query.offset ?? 0
  const page = order.slice(start, start + query.limit)
  const items = page.flatMap((slug) => groups.get(slug)!.slice(0, query.perEpisode ?? 5))
  const ids = new Set(items.map((h) => h.id))
  return {
    items,
    total,
    episode_total: order.length,
    episode_counts: Object.fromEntries(page.map((slug) => [slug, groups.get(slug)!.length])),
    notes: notes.filter((n) => n.target === "highlight" && ids.has(n.target_id)),
  }
}

export function pageNotesLocally(all: Note[], highlights: Highlight[], query: NotesPageQuery): NotesPage {
  const needle = (query.q ?? "").trim().toLocaleLowerCase()
  const words = query.words ? needle.split(/\s+/).filter(Boolean) : [needle]
  const hits = [...all]
    .sort((a, b) => b.created_at - a.created_at)
    .filter((n) => words.every((w) => !w || n.text.toLocaleLowerCase().includes(w)))
  const counts: Record<string, number> = {}
  for (const n of hits) counts[n.target] = (counts[n.target] ?? 0) + 1
  const kinds = query.kinds ?? []
  const selected = kinds.length ? hits.filter((n) => kinds.includes(n.target)) : hits
  const start = query.offset ?? 0
  const items = selected.slice(start, start + query.limit)
  const on = new Set(items.filter((n) => n.target === "highlight").map((n) => n.target_id))
  return { items, total: selected.length, counts, highlights: highlights.filter((h) => on.has(h.id)) }
}

export function pageResurfacingLocally(
  resp: ResurfacingResponse,
  query: { offset?: number; limit: number; perEpisode?: number },
): Required<ResurfacingResponse> {
  const groups = new Map<string, ResurfacingItem[]>()
  for (const it of resp.items) {
    const slug = it.highlight.episode_slug
    groups.set(slug, [...(groups.get(slug) ?? []), it])
  }
  const order = [...groups.keys()]
  const start = query.offset ?? 0
  const page = order.slice(start, start + query.limit)
  return {
    items: page.flatMap((slug) => groups.get(slug)!.slice(0, query.perEpisode ?? 100)),
    paused: resp.paused,
    total: resp.items.length,
    episode_total: order.length,
    episode_counts: Object.fromEntries(page.map((slug) => [slug, groups.get(slug)!.length])),
  }
}

export function pageCollectionLocally(
  resp: CollectionDetail,
  query: { limit: number; offset?: number; kind?: string },
): Required<CollectionDetail> {
  const kindCounts: Record<string, number> = {}
  for (const it of resp.items) kindCounts[it.kind] = (kindCounts[it.kind] ?? 0) + 1
  const selected = query.kind ? resp.items.filter((i) => i.kind === query.kind) : resp.items
  const start = query.offset ?? 0
  return {
    collection: resp.collection,
    items: selected.slice(start, start + query.limit),
    total: selected.length,
    kind_counts: kindCounts,
  }
}
