import type { RouteLocationRaw } from 'vue-router'

/**
 * A learning day's "Play" (operator 2026-10-14): the episode page plays the day's clip and stops at
 * its end, marks the day done, and offers the way back to the week. The page finds the clip in the
 * path itself (`?learn=<subject>&day=<n>`). `from=learn`: a highlight saved there counts as saved on
 * a learning day (stats, 2026-10-12).
 */
export function learnDayRoute(subjectId: string, step: { day: number; episode: { slug: string } }): RouteLocationRaw {
  return {
    name: 'player',
    params: { slug: step.episode.slug },
    query: { learn: subjectId, day: String(step.day), from: 'learn' },
  }
}
