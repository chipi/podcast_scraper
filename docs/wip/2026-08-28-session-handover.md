# Session handover — 2026-08-27/28

Main is green at `0d0709c3`. Production corpus, bridges and search index are consistent. One
deploy is outstanding and one nightly is unverified. Everything else here is either done or filed.

## The thread that ran through this session

#1685: a single-token person id (`person:jensen`) identifies someone within ONE episode and nobody
globally, yet it rendered as a followable entity card and pooled several real people under one
token. The pipeline had minted scoped ids since 2026-08-21; the existing 678 episodes never got the
same treatment.

**That is now done and verified.** Bare-name occurrences went **215 → 0** — the audit's
"Bare person names" section no longer renders at all.

## What is DONE

| | Evidence |
|---|---|
| m0007 backfill applied | `170 episodes, 275 scoped, 0 healed` then `32 episodes, 35 scoped` after the fix |
| Bridges re-pointed | `147 bridges, 244 substitutions, 0 unresolved` |
| Search index rebuilt | `total_vectors=75968` |
| Bare names eliminated | audit section absent (was 215 occurrences / 178 ids) |
| Coexistence damage | **23 → 0** ([run 33135256168](https://github.com/chipi/podcast_scraper/actions/runs/33135256168)) |
| Cross-episode contamination | 0, throughout |
| #1686 (missing summaries) | 0 absent / 0 blank across 678, confirmed in prod |

## OUTSTANDING — do these

**1. Deploy `0d0709c3` (or later).** Not done. The migration reached prod via mounted source (see
the `prod-tooling-runs-mounted-source` memory note), but the **live pipeline runs the deployed
image** — so until this deploy, every newly ingested episode re-creates the quote-`speaker_id`
damage m0007 just cleaned. Nothing is broken meanwhile; it accrues.

**2. Watch the next nightly.** It has failed three nights running (08-25, 08-26, 08-27). The last
failure was `nightly-test-unit` / `test_real_workflows_pass_the_full_gate` — the ADR-115 secrets
gate, tripped by two prod workflows I added without the tmpfs-secrets step. Fixed by `afcf1260` at
**21:57 UTC**; the failing nightly ran at **12:25 UTC**, so **nightly has not run since the fix**.
The gate passes on main locally (`6 passed`). The 03:00 run is the first real verification.

## OPEN ISSUES, with what is known

- **#1865 — `index_corpus(rebuild=True)` deletes the index then indexes nothing.** Highest value.
  `rmtree(lance_path)` leaves the fingerprint sidecar beside it, so every episode is skipped and no
  index is written. This took prod search down for ~80 minutes. Fix: pass `drop_existing=True`
  through to `build_two_tier_index`, which already has the switch, and prefer its MVCC clear over
  destroy-then-attempt. Regression test: rebuild over an existing index+sidecar must yield non-zero
  vectors.
- **#1862 — the scoping pass writes GI but not KG, and is not durable across re-processing.** The
  edge/roster half is fixed (#1868); these two halves are not. m0007 repaired the existing KG-side
  damage, but the pipeline will keep re-creating it.
- **#1801 — the entity enricher.** Now has real numbers: 12 blocked heals with named targets
  (`brandon-anderson`, `gabrielle-steinhauser`, `sergei-yudinov`, +9), and of ~178 single-token
  names ~26 recur across 2+ episodes while ~152 appear exactly once. Only the recurring ones are
  worth resolving. Decide on that split, not on principle.
- **#1798** (mypy/numpy), **#1799** (test isolation) — unchanged, low priority.

## Incident

`docs/incidents/2026-08-27-prod-search-index-deleted.md`. Search was down ~80 minutes because I ran
a destructive rebuild whose delete succeeded and whose build could not run. Five dispatches for one
reindex. The report is weighted toward what to do differently — fix `--rebuild` at source, never
delete before the replacement is proven, ship a read-only sibling with every destructive operation,
never assert on a match string not observed in real output, read the call path before the first
dispatch.

## Things that will bite the next person

- **Prod tooling runs MOUNTED source, not the image.** A migration or audit fix reaches prod the
  moment it lands on main — no deploy. The live pipeline is the opposite. See the memory note.
- **678 episodes ≠ 953 `.gi.json` files.** The rest are stale copies in earlier `run_*` dirs.
  Migration counts and audit counts describe different populations; never compare them directly.
- **A prod reindex must delete `episode_fingerprints.json` first**, and the verb is
  `index --rebuild`, not `index-two-tier --rebuild`. Both are in `reindex-prod.yml` now.
- **`reindex-prod.yml` defaults to `mode=stats`** — read-only. `rebuild` must be chosen.

## New tooling added this session

- `scope-bare-names-prod.yml` — runs m0007 + the bridge re-point under one corpus lock
- `reindex-prod.yml` — `stats` (default, read-only) or `rebuild`, with real assertions on
  `total_vectors` and a moved `last_updated`
- `inspect-prod-corpus.yml` gained `m0007_dry_run` (opt-in, not in `all`)
- `capability_audit` gained `measure_placeholder_health` — contaminated ids, blocked heals,
  per-case location (`edge_endpoint` / `edge_speaker` / `node_speaker`), re-run flip risk, and
  convergence
