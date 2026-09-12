/**
 * Case-insensitive substring match for the Library type-to-filter search (#2042 follow-up). One
 * place so Following and Saved match identically. An empty/blank query matches everything, so a
 * caller can pass the raw query straight through without a "searching?" branch.
 */
export function matchesQuery(text: string | null | undefined, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return (text ?? '').toLowerCase().includes(q)
}
