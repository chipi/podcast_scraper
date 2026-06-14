# Post-migration graph validation — feat/967 (#967 + #974 + #876)

Validation of the re-diarized prod-v2 corpus (99 eps) graph after folding the
fcose graph work (#967) onto the two-artifact transcript model (#974) + diarized
roster (#876). Corpus: `.test_outputs/manual/prod-v2/corpus`.

## Bugs found + fixed (this branch)

1. **#967↔#974 focus reconciliation** (`GraphCanvas.vue`, commit e231ec2e).
   Folding cose→fcose exposed a clash: a `quote` search hit has no cy node, so
   focus resolves the *fallback* (its Episode) for the camera — but
   `tryApplyPendingFocus` then opened the Episode rail, clobbering the quote
   rail the search subject set. Pre-#967 cose stayed busy long enough that the
   idle-gated fallback never fired; fcose goes idle fast → clobber. Fix: only
   (re)open the subject rail when the PRIMARY resolves; on a fallback, anchor
   camera+selection but leave the rail on the user's actual pick. Also hardened
   the watcher's deferred resolve to a bounded poll.

2. **Prefix-tolerant rail resolution** (`GraphNodeRailPanel.vue` + `NodeDetail.vue`,
   commit 4d79e569). Found by real-corpus exploratory testing: clicking "Show on
   graph" on a quote opened an empty "Node" rail. On a merged GI+KG corpus
   `mergeGiKg` prefixes GI ids (`g:quote:…`) but the search hit carries the bare
   id (`quote:…`); the rail's exact-match `findRawNodeInArtifact` missed it.
   Switched both to `findRawNodeInArtifactByIdOrPrefixed` (already used by the
   subject Person/Topic views). The single-artifact e2e fixtures never carry the
   `g:` prefix, so the suite couldn't catch this — added a NodeDetail unit
   regression.

## Tier-3 data scan (all 99 eps) — ALL GREEN

- **Offset alignment** (`verify-gil-chunk-offsets --strict`): verdict **aligned**,
  overlap_rate **1.0**, all 2301 quotes verifiable + overlapping (was 0.54
  pre-#974). The #545 drift is fully fixed on real data.
- **transcript_ref**: 2301/2301 quotes point at `.adfree.txt` (the new base).
- **SPOKEN_BY**: 2710 edges, all valid Quote→Person, 2248/2301 quotes attributed.
- **Speakers**: 96/99 eps carry `content.speakers` (host 52 / guest 172 roles).

## Test tiers — ALL GREEN

- vitest: **2074** (133 files).
- Playwright graph e2e: graph mocks 14 + handoff 89 + handoff-production 17 +
  transcript-viewer (raw + adfree) 2 = ~**122**.
- Python: 23 (#974/#876) + 326 (gi/kg unit) + 37 (gi integration) = **386**.
- Real-corpus validation walk (prod-v2): V1 Library→graph, V3 search→graph,
  V5 episode hot-state, V6 LanceDB hybrid search — **pass**.

## Verified end-to-end on real data (browser)

Quote search → "Show on graph" → quote rail resolves (heading "Quote", full
passage, `SPOKEN_BY` speaker chip) → "View transcript" → dialog reads the
**ad-free base** + highlights the quote at the exact ad-free char offsets
(392–488). The full #974 chain renders correctly.

## Open findings (NOT graph-migration regressions)

- **V4 dashboard topic-landscape slow** (>30s under the validation test's
  headless concurrent burst; ~12s manually). Root cause: single-worker serve-api
  serializes the Intelligence tab's ~6 concurrent corpus-scan calls on a 99-ep
  corpus. Same root as an earlier **serve-api SIGSEGV under the concurrent
  first-load burst**. Performance/robustness item, separate from the graph.
- **Pushkin network leak** (POLISH 1, confirmed in detail): host intro
  "This is Unhedged… I'm Pushkin. I'm Katie Martin" → roster took the network
  bumper "Pushkin" as the host name instead of "Katie Martin". Surfaces as a
  `person:pushkin` speaker + a "Pushkin" entity node.
- Validator classes still present (need DGX re-diar to land fixes): 9
  direct_download "unattributed" (Odd Lots), 18 network-speaker, 42
  unnamed-voice panels.

## Still to do

- PHASE 2: graph edge cases (direct_download eps, unnamed-speaker panels,
  multi-run dedup correctness) + performance (fcose at 1.7k nodes; serve-api
  concurrent-burst robustness).
- POLISH: roster network-name filter + validator false-positive guards (land on
  corpus via DGX re-diar when available).
- Close #876 when the corpus is clean.
