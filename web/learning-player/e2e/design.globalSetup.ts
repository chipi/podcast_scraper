import { buildIndexIfAbsent } from './globalSetup'

/**
 * The design harness's setup: only the search index. Not the e2e setup — that one refuses a `dist/`
 * older than `src/` (this harness serves the dev server, so it has no bundle to be stale) and wipes
 * the per-user state (the design account is CONSTANT on purpose, see surfaces.design.spec.ts).
 * Without the index the shots judged a Discover with no "15 minutes on" row, no "Learn this week" and
 * an empty search (2026-10-11).
 */
export default async function designGlobalSetup(): Promise<void> {
  await buildIndexIfAbsent()
}
