/**
 * Where a note's "Open" link goes — ONE rule, every surface.
 *
 * This lived as two copies, in CollectionsView and SearchView, and they had drifted: Search routed
 * a `storyline` note to its page, Collections returned null for the same note, so the identical
 * note showed "Open" on one screen and not the other (operator 2026-09-17). A note's target is a
 * property of the NOTE, not of the list it happens to be rendered in.
 *
 * Coverage is deliberately total: every target a note can carry resolves to a route, including
 * `highlight`, which previously had none anywhere. A highlight is anchored to a moment in an
 * episode, so its note opens the player at that moment — the same place the Highlights list jumps
 * to. `insight` is the one target with no page of its own; it is also the one target no UI can
 * create (no NoteComposer is mounted on an insight), so it resolves to the episode that carries it
 * when that is known, and null otherwise.
 */
import type { Highlight, NoteTarget } from "../services/types"

export interface NoteRoute {
  name: string
  params: Record<string, string>
  query?: Record<string, string>
}

/**
 * The route for a note, or null when the target cannot be opened.
 *
 * `highlights` lets a `highlight` note resolve through to its episode + timestamp; pass the capture
 * store's list. Without it a highlight note simply has no link, which is the old behaviour rather
 * than a wrong one.
 */
export function noteRoute(
  target: NoteTarget | string,
  id: string,
  highlights: Highlight[] = []
): NoteRoute | null {
  if (!id) return null
  switch (target) {
    case "episode":
      return { name: "player", params: { slug: id } }
    case "topic":
      return { name: "topic", params: { id } }
    case "person":
      return { name: "person", params: { id } }
    case "show":
      return { name: "podcast", params: { feedId: id } }
    case "storyline":
      return { name: "storyline", params: { id } }
    case "highlight":
    case "insight": {
      // Both hang off an episode rather than having a page. Resolve through the highlight so the
      // link lands on the moment, not just the episode top.
      const h = highlights.find((x) => x.id === id)
      if (!h?.episode_slug) return null
      const at = h.start_ms != null ? String(Math.floor(h.start_ms / 1000)) : undefined
      return {
        name: "player",
        params: { slug: h.episode_slug },
        ...(at ? { query: { t: at } } : {}),
      }
    }
    default:
      return null
  }
}

/**
 * WHAT a note is attached to, in words — "Risk Is a Systems Property", not `p09-a4bbb5dde3`.
 *
 * A note row showed only its target KIND and an "Open" link, so a list of notes read as "episode,
 * episode, topic" with no way to tell which episode without following each link (operator
 * 2026-09-17). The kind alone is the least useful half of the answer.
 *
 * Two classes of target, resolved differently and deliberately:
 *
 * - **Entity ids carry their own label.** `topic:systems-thinking` → "systems thinking". No lookup,
 *   no fetch, and it cannot go stale. This is the same de-slugging `FollowedInterests` does.
 * - **Episodes and shows do not.** A slug or a feed id says nothing, so the caller passes whatever
 *   titles it already holds (`titles`). Returns null when the title is genuinely unknown rather
 *   than echoing an id back — a raw slug on screen is worse than no subtitle, and the "Open" link
 *   is still there.
 *
 * `highlight` / `insight` resolve through to the episode they are anchored in, matching where
 * {@link noteRoute} sends them.
 */
export function noteTargetLabel(
  target: NoteTarget | string,
  id: string,
  titles: { episodes?: Map<string, string>; shows?: Map<string, string> } = {},
  highlights: Highlight[] = []
): string | null {
  if (!id) return null
  switch (target) {
    case "topic":
    case "person":
    case "storyline":
      return deslugEntityId(id)
    case "episode":
      return titles.episodes?.get(id) ?? null
    case "show":
      return titles.shows?.get(id) ?? null
    case "highlight":
    case "insight": {
      const slug = highlights.find((x) => x.id === id)?.episode_slug
      return slug ? (titles.episodes?.get(slug) ?? null) : null
    }
    default:
      return null
  }
}

/** `topic:systems-thinking` / `thc:managing-risk` → `systems thinking` / `managing risk`. */
function deslugEntityId(id: string): string {
  const bare = id.includes(":") ? id.slice(id.indexOf(":") + 1) : id
  return bare.replace(/[-_]+/g, " ").trim()
}
