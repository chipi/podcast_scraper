# Validation backlog & delivery plan — 2026-08-08

Session context: this session shipped the remote MCP Host-header fix + pip-audit ignore
(deployed `sha-0f3cd36`), the tailnet self-devices ACL, and the DGX diarization guardrail
cherry-pick (on main `24a41d3c`, direct-to-main). It also fixed the prod corpus's `no_index`
(reindex) + relational/enrichment tool gaps (enrich-edges + enrichment, run live on the May-05
corpus). Branch/stash hygiene done (5 merged branches + 3 stashes deleted).

This doc is the **validation backlog** — what we did but haven't fully validated — plus the
delivery order. **DGX is down; do not depend on it.** Focus order: **Player "Your Week" → Remote
MCP → Corpus prep (on operator go)**.

## Priority 1 — Player #1487 "Your Week" (live validation)
Merged + CI-green, but **prod is still `sha-0f3cd36`** (before it). Deliver:
- Redeploy prod to the `sha-24a41d3c` image (gated) — carries #1487 + the DGX guardrail (guardrail
  is inert on the player runtime).
- Verify: "Your Week" renders in-app; the hardened `digest-scheduler` sidecar auto-fires the weekly
  digest (heartbeat healthy, a digest actually enqueues).
- Agent-doable except the prod gate approval.

## Priority 2 — Remote MCP: the claude.ai OAuth connector flow
Validated with a **PAT** only. The **browser OAuth path is unproven**: DCR → authorize →
**consent** → token.
- Operator: add the connector at `https://mcp.closelistening.app/mcp`, complete the OAuth flow.
- Agent: watch the mcp/api logs during the flow; fix any DCR/consent/callback failure.

## Priority 3 — Corpus prep + deploy (ON OPERATOR GO; no DGX/Whisper)
Reuse the local **diarized DeepSeek corpus** `v2.5-deepseek-fixed-100ep` (in the
`podcast_scraper-ai-ml-improvements` worktree). All prep is deterministic:
- `enrich-edges` (adds `HAS_EPISODE`), deterministic enrichment stage (`enrichments/`), dedup to
  newest run per feed, regen `feeds.spec.yaml` if export needs it.
- `export-corpus` → transfer → `restore-corpus-prod` → **redeploy** (NEVER `docker restart` — it
  drops the tmpfs `/dev/shm/player-secrets` and crashed player-api this session).
- Then **re-validate all 38 MCP tools** on the DeepSeek corpus (speaker tools should light up —
  82% quotes have `speaker_id`).
- Note: the live May-05 prod corpus is currently **hand-modified** (enrich-edges + enrichment run
  directly on the volume); it has relational + enrichment data but **no diarization**. Replaced by
  this deploy.

## Priority 4 — `consensus_search` ML enricher
Needs the `topic_consensus` enricher, which needs (a) CLI **provider-wiring** (EmbeddingProvider /
ConsensusScorer injection — the CLI auto-registers deterministic enrichers only), (b) an offline
NLI model. Code change + model. Tracked in #1497.

## Priority 5 — Prevention (#1494 / #1497) — unbuilt
Deploy-time **staleness guard** (index schema + edge-vocabulary), a **reprocess-GI-from-existing-
transcripts** flag, incremental/embed-free indexing. Prevents the drift recurring.

## Deferred — DGX-dependent (box is DOWN)
- Validate the DGX diarization guardrail (`flock` cross-process single-flight) against a **live
  DGX** — unit-tested only.
- Reconcile `production` branch (in `-infra`) — it still holds the 2 original DGX commits now on
  main; rebase/clean when convenient.

## Related issues
- #1494 — scalable + incremental search indexing.
- #1497 — MCP relational tools empty on prod (edge-vocabulary drift + diarization root cause).
