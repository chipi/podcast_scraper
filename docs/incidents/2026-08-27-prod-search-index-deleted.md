# Incident — prod search index deleted, ~80 minutes down (2026-08-27)

**Severity:** degraded surface, no data loss. Search returned nothing between 18:03 and 19:23 UTC.
**Cause:** operator (this assistant) ran a destructive rebuild whose delete step succeeded and whose
build step could not run. **Trigger:** a workflow I wrote the same evening, dispatched five times
before it worked.
**Data:** nothing unrecoverable. The index is derived from corpus artifacts, which were intact
throughout.

## Timeline (UTC)

| Time | Event |
| --- | --- |
| 17:29 | m0007 migration applied — 170 episodes, 275 ids scoped, 0 healed, 0 unparsable |
| 17:41 | Bridge re-point applied — 147 bridges, 244 substitutions, 0 unresolved |
| ~17:44 | Reindex attempt 1 — `index-two-tier --rebuild` → exit 2, argparse. **Wrong verb.** No effect |
| 17:54 | Attempt 2 — `index --rebuild` → exit 0, printed nothing. Index **was** rebuilt |
| 18:03 | Attempt 3 — read stats first (`last_updated=17:54:14`, healthy), then rebuild → **exit 3 after 58s** |
| 18:03 | **Index deleted here.** `rmtree` ran; the build did not |
| 18:13 | I reported "most likely the delete never ran" — an inference from timing + exit code. Wrong |
| 18:14 | Read-only stats: `<no stats returned>`. Index confirmed **gone** |
| 18:17 | Attempt 4 — rebuild → exit 3 in seconds. Still failing |
| ~18:20 | Read `index_corpus` → `build_two_tier_index` end to end. Found the fingerprint sidecar |
| 18:17→19:23 | Attempt 5 with the sidecar removed — **66 minutes**, `total_vectors=75,968` |

## Root cause

`index_corpus(rebuild=True)` (`search/indexer.py:551`) does:

```python
if rebuild and lance_path.exists():
    shutil.rmtree(lance_path)          # deletes search/lance_index
tt = build_two_tier_index(...)         # then rebuilds
```

The per-episode fingerprint sidecar lives **beside** the index, not inside it —
`_fingerprints_path() = Path(lance_path).parent / "episode_fingerprints.json"`
(`two_tier_indexer.py:173`). So `rmtree` leaves it. The builder then loads it
(`stored_fps = {} if clear_requested else _load_episode_fingerprints(...)`, `:378`), every one of the
678 episodes matches its stored fingerprint, every one is skipped, no rows are written, no index is
produced — and the process exits 3.

**So `--rebuild` is a delete followed by a guaranteed no-op** whenever the sidecar survives, which
is always. The index is destroyed and not rebuilt. That is a product bug, not just a workflow bug.

`build_two_tier_index` already has the correct switch — `drop_existing` → `clear_reason` →
`clear_requested` (`:366`), which makes it ignore fingerprints. `index_corpus` never passes it.

## What I did wrong, specifically

**1. I had the answer written down and did not read it.** `CORPUS_INTEGRITY_REPAIR_RUNBOOK.md` and
the deploy plan I wrote myself on 2026-08-21 both say: *"Delete `episode_fingerprints.json`
alongside `lance_index`, or every episode is skipped and you get a silent empty index."* There is
also a memory note on exactly this. I built the workflow without applying any of it.

**2. I wrote a guard that could not fail.** The first empty-index check grepped for
`"0 documents indexed"` — a string I invented and never compared against real output. The indexer
logs via `logger.info`, which emitted nothing in that container, so the grep matched an empty log
and passed. A guard added specifically to catch "empty index exits 0 and reads as success" instead
manufactured that exact reading.

**3. I reasoned instead of measuring, and reported the reasoning as near-fact.** After the delete I
argued from run duration (58s vs ~10min) and exit code (3 vs the 2 an exception returns) that the
`rmtree` had probably not run. Both premises were sound; the conclusion was wrong. One read-only
stats call would have settled it and I had already built the capability to make that call.

**4. I fixed each error in front of me instead of reading the path once.** Five dispatches: wrong
verb → invented verification → delete → still-failing → fingerprints. Reading `index_corpus`
through to `build_two_tier_index` took two minutes and produced the real answer. Doing that before
the first dispatch would have prevented all of it.

**5. The workflow could only do the destructive thing.** Until the fourth failure there was no way
to ask "what state is the index in?" without running the delete again. Investigation was more
dangerous than action.

## Fixes already landed

| Commit | Fix |
| --- | --- |
| `e91a5214` | Correct verb: `index --rebuild`, with the distinction recorded so it is not "corrected" back |
| `457b903d` | Verification reads `index --stats` JSON (stdout, unswallowable) and asserts **both** `total_vectors > 0` **and** `last_updated` changed — either alone is passable by a no-op |
| `2fc66d28` | `mode` defaults to **`stats`**: read-only. `rebuild` must be chosen deliberately |
| `90d09228` | Delete the fingerprint sidecar before rebuilding |

## What to do better — concrete, in priority order

**1. Fix `--rebuild` in the product, not just the workflow.** `index_corpus(rebuild=True)` should
pass `drop_existing=True` to `build_two_tier_index` and let the MVCC clear path do the work, rather
than `rmtree` + hope. Today the flag's contract ("delete the index directory and rebuild from
scratch") is only half true: it reliably deletes and reliably does not rebuild. Anyone who runs it
without knowing the sidecar trick destroys their index. **This is the single highest-value fix and
it removes the incident's cause rather than working around it.**

**2. Never delete before the replacement is proven.** Build to a temp directory, verify non-zero
vectors, then swap. The current order — destroy, then attempt — has no safe failure mode. LanceDB's
MVCC overwrite path (`_plan_reindex_clear`, "never rmtree — #1206") already exists precisely because
someone reached this conclusion before; `index_corpus` bypasses it.

**3. A destructive operation must have a read-only sibling, shipped at the same time.** The `stats`
mode should have existed before the first rebuild dispatch, not after the fourth failure. Rule:
if a workflow can destroy state, it ships with a way to inspect that state without destroying it.

**4. Never write an assertion whose match string has not been seen in real output.** The empty-index
guard was tested against nothing. Where the format cannot be observed first, assert on structured
output (JSON via `print`) rather than log prose — that is what the fix does and it is the reason the
final run could be trusted.

**5. Read the call path before the first dispatch, not after the fifth failure.** For any operation
that writes to prod: trace it end to end in the source first. Two minutes of reading versus 80
minutes of downtime and five approvals.

**6. Search the runbooks for the operation's name before building anything.** Both the sidecar trap
and the correct sequence were already documented. A single grep for `episode_fingerprints` would
have surfaced it.

**7. Treat an unexplained failure as unknown state, not as a state I can deduce.** "The timing says
the delete probably did not run" is a hypothesis. With a read-only check available, reporting it as
a near-conclusion was the wrong call and it delayed the real diagnosis by ten minutes.

## Follow-ups to file

- **[#1865](https://github.com/chipi/podcast_scraper/issues/1865):** `index_corpus(rebuild=True)` deletes the index and then skips every episode
  because the fingerprint sidecar survives `rmtree`. Pass `drop_existing=True` through, or delete the
  sidecar inside `index_corpus`. Include a regression test: rebuild over an existing index+sidecar
  must produce non-zero vectors.
- **#1862 (open):** scoping pass writes GI but not KG, and is not durable across re-processing.
  Unaffected by this incident; m0007 repaired the existing KG-side damage.

## What was never at risk

The corpus. m0007 and the bridge re-point both completed with verified counts before any of this,
and the index is derived data — every vector was rebuildable from artifacts that were never touched.
