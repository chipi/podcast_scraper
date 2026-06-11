# Next-session handoff — `feat/946` (lance + graph-testing + DGX resilience)

Pick-up note for a fresh session. Branch: **`feat/946-existing-only-rediarization`**,
rebased on `origin/main` `67bf819e`, working tree clean, **32 commits ahead**.

## Immediate next step: land this PR

1. **Confirm `make ci` is green.** The last run got to its final stage
   (`stack-test-ml-ci`, Docker + Playwright); every earlier stage passed
   (fast gates, unit 4444 / integration / e2e, test-ui, test-ui-e2e,
   build-viewer, coverage-enforce, docs, build). Re-run `make ci` (ML models are
   cached, so it skips preload) or just `make stack-test-ml-ci` to confirm the tail.
2. **Push + open ONE PR** (`git push -u origin feat/946-existing-only-rediarization`,
   then `gh pr create` — **ready, not draft**). The operator wants a single PR; DGX
   continues as a *new chapter after it lands*. **Do not push without the operator's
   explicit go** (standing rule).
3. Suggested PR body themes + links: lance search parity/self-heal, Tier-3
   viewer-validation, large-graph diagnosis (#967), DGX resilience (#954/#956),
   #876/#946/#947/#948/#953/#913/#914.

## What's on the branch (this session's work, all committed + green)

- **Lance/search**: `publish_date` + `source_id` FAISS-parity on the two-tier hybrid
  index (fixed digest topic-bands under lance + the viewer "Show on graph"), plus
  `LANCE_SCHEMA_VERSION` self-heal (stale index → read skips to FAISS, reindex
  rebuilds). Tier-3 walk **34/34**. Docs in RFC-090 §10 + SEMANTIC_SEARCH_GUIDE.
- **Tier-3 corpus**: walk CORPUS must point at the FIXTURES_VERSION dir
  (`.../viewer-validation-corpus/v2`), not the parent. `build-validation-index`
  builds FAISS + lance two-tier + topic_clusters. New **V6** spec asserts lance (not
  FAISS) is serving (RRF score < 0.1).
- **#956 DGX resilience**: shared `dgx_http_client` factory (TCP keepalive +
  `Connection: close`) on top of the #954 watchdog/breaker. Issue reconciled:
  Tier-2 done; per-read-timeout rejected (false-aborts long zero-byte POSTs);
  Tier-1 async-jobs deferred to the DGX chapter.
- **CI gap fixes** (found by the first full `make ci` on this branch): cap test
  15→25, `lint-markdown` ignore for `validation-results/`, 2 CircuitBreaker
  docstrings, friendly httpx error in `dgx_http_client`.

## Open follow-ups (future chapters, NOT this PR)

- **#967 — large-graph viewer**: replace `cose` (O(n²), 134s @ 2861 nodes) with
  **fcose** globally + persist selection across the ego→full reload (drops during the
  rebuild). TDD selection-persistence first. Episode cap is interim 25
  (`graphEpisodeSelection.ts`); raise after the layout fix. Full mechanism + file:lines
  in the issue.
- **#956 (remaining) — DGX Tier-1**: async job submission (`POST /jobs` → poll/SSE) on
  each DGX service — the "DGX as prod AI node" chapter.
- **#876 re-diarization batch**: PAUSED awaiting free GPU (DGX shared with autoresearch).

## Environment notes

- Other git worktrees: `podcast_scraper-FUTURE` (autoresearch), `podcast_scraper-infra`.
- Local dev servers (`make serve-api`/`serve-ui` on :8000/:5173) may be left running from
  the validation work — `make ci`'s `cleanup-processes` clears strays.
- The synthetic validation corpus's `search/lance_index` is gitignored — rebuild with
  `make build-validation-index` before a Tier-3 walk.
