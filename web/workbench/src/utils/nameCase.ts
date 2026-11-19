/**
 * Title-case each word for display of person / show names that arrive from
 * slugs (e.g. ``jay powell`` → ``Jay Powell``, ``o'shaughnessy`` → ``O'Shaughnessy``).
 * Idempotent for already-cased names ("Katie Martin" stays "Katie Martin").
 */
export function titleCaseWords(s: string | null | undefined): string {
  return (s ?? '').replace(/(^|[^\p{L}\p{N}])(\p{L})/gu, (_m, sep, ch) => sep + ch.toUpperCase())
}
