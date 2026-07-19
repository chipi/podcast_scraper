# Open threads — evals, tests, and GH issues left behind (2026-07-16)

Full-context companion to [`HANDOVER-2026-07-16-fuse-gi-cap-nextpr.md`](HANDOVER-2026-07-16-fuse-gi-cap-nextpr.md).
That handover covers the 8 unpushed commits; **this** captures everything else still open around the
prod-v3 bake-off / fuse / GI work so nothing is silently dropped across the session reset.

---

## 1. Deferred EVAL work

Context: all 9 bake-off arms in `data/eval/runs/bakeoff_*_18ep/` are **`grounded_insights` task only**
(no prose-summary scoring). Report: [`BAKEOFF-18EP-RESULTS.md`](BAKEOFF-18EP-RESULTS.md).

| # | What | State | Notes |
| --- | --- | --- | --- |
| E1 | **grok-4.3 judge surf/ep re-score of the 3 re-runs** (`deepseek_medium_18ep_postfix`, `anthropic_sonnet45`, `deepseek_chat`) | **The one remaining step.** Runs finished; timings + call-counts final; distinct-knowledge (surf/ep) re-scoring is not done. | `deepseek-chat (V3.2, non-reasoning)` is the standout (1.1 min/ep, ~$0.05); its knowledge score vs deepseek-v4-flash's 17.1 is the open question the judge pass answers. |
| E2 | **Summary-quality scoring — the A/B/C decision** | Deferred, never started. | The arms produce grounded insights, not prose summaries. Scoring summary quality needs a **summarization pass** + a grok-judge build. Decide A/B/C (which arms, which reference) before running. |
| E3 | **sonnet-5 (`claude-sonnet-4-5`) full re-run** | Partial — crashed at episode 11 when the Anthropic account hit its **spend cap**. | The bundled-empty-content fix WAS verified (0 empty-content failures) before the cap hit, so the fix is sound. Re-runnable in one command once the cap resets. |
| E4 | **Re-baseline the bake-off for the NEW dynamic GI cap** | NEW — created by commit `a01377c3`. | The duration-scaled insight ceiling changes long-episode insight counts vs the flat-50 baseline every arm was measured under. `avg_insight_nodes` and surf/ep shift on >1h episodes. Re-baseline before trusting long-form comparisons. |
| E5 | qwen-on-DGX augmentation of the report | DONE (1.8 min/ep, $0) — recorded. | No action; listed for completeness. |

---

## 2. Deferred TEST / measurement work

| # | What | Why it matters |
| --- | --- | --- |
| T1 | **Full `make ci`** against the fuse/cap changes | Never run since those commits (unit + GI green locally, but integration + e2e + coverage-enforce + docker stack-test unverified). **First task next session.** |
| T2 | **Per-episode `llm_calls` counter in the metrics** | The fuse limits (1500/40000) and the staged-mode risk are **modeled, not measured**. `_EpisodeFuse.calls` already holds the real per-episode count — log it at episode end (e.g. when `install_episode` overwrites the prior one) so the next real run measures the true call distribution and the ~6-candidates/insight assumption is replaced with data. Ties to GH #1180. |
| T3 | **Staged-mode fuse headroom decision** | `gil_evidence_nli_mode` default is `staged` (1 call/pair). A staged 4h/200-insight episode ≈ 1,650 calls > the 1,500 per-episode fuse → could false-trip. Prod/eval use `bundled` (~200), so only the naive default is exposed. Resolve with T2's data (raise to ~2,000, or make staged bundle by default). |
| T4 | **e2e resource-gated skips remain (by design)** | `test_deepgram_provider_e2e` (needs `USE_REAL_DEEPGRAM_API=1`) and `test_diarization_e2e` (needs provisioned pyannote models offline) still skip. These are **e2e**, not unit — the "no unit skips" rule (done) did not target them. Flag only if you want them wired to fixtures. |

---

## 3. Open GH issues in this work's orbit

Filed this session:

- **#1191** — GI route-and-tag (no pipeline truncation); the *proper* fix behind the interim duration-scaled cap. Source: `GI_WHAT_TO_SURFACE.md`.

Directly adjacent (touched or implied by this work):

- **#1173** — Transcript segment timestamps drift from audio. *Note:* the bake-off/fuse/cost commits were prefixed `#1173`, but that issue is narrowly about timestamp drift — the commit scope is broader (bake-off umbrella). Reconcile the epic reference when opening the next PR.
- **#1169** — Speaker resolution (name voices, host/guest roles). The `download_args` 9-tuple that caused the summarization crash came from here (`b4efeab6`).
- **#1180** — Document pipeline parallelism + add overlap/saturation metrics. **Directly relevant** to the fuse/parallel-worker review and the T2 per-episode call counter.
- **#1182** — GI silently produces ZERO quotes on transcripts with no line structure. GI-path robustness.
- **#1181** — GI: qwen evidence is thin, not missing — re-assess the quote tail. GI-quality.
- **#1188** — Pattern cleaning misses host-read cross-promos (Athletic ad survives 10/10). Cleaning-stage.
- **#1189** — Golden fixtures v4 (one per show, real diarization + hand-labelled truth). Pairs with `CORPUS-V4-FIXTURE-LADDER.md` (edited this session, KG-segments audit closed).
- **#1002** — Fine-tune response-shape guardrail thresholds against firing-rate data. Resilience-adjacent (ADR-100/113 family).
- **#1179 / #1178 / #1177 / #1174** — transcription/diarization deathmatch epic + MOSS provider/eval + large-v3-turbo. Eval-heavy; adjacent to the corpus this bake-off used.
- **#972** — Real-podcast full sweep (multi-episode × 4-way transcription × 3-backend summary). The broader eval sweep this bake-off is a slice of.
- **#1167** — Unresolved `Speaker NN` ids leak into the KG as Person entities. Speaker/KG.

Epics / umbrellas: **#1062 / #911** (Learning Platform), **#1160** (infra security before public edge) — not in this work's path, listed so the orbit is complete.

> Section 3 is the raw orbit; **§5 below records the triage actually performed on these issues.**

---

## 5. GH issue triage performed this session (2026-07-16)

**Root cause of the whole triage:** PR #1190 merged with an **empty `closingIssuesReferences`** — it
auto-closed **zero** issues despite delivering ~11 issues' worth of work, because its body carried no
`Closes #N` lines (and squash erased the commit-level mentions). The PR body even contained an
unactioned *"Verify the `Closes #N` list before merge"* reminder. This is the #1111 failure mode
repeated. Everything below is the **manual** recovery.

**Closed (delivered — with a closing comment referencing the merge):**

- **#1169** — speaker resolution. All 3 title goals shipped (name voices 36.6%→47%, host/guest roles,
  cameo/commercial labels). Its one leftover — the ~113 un-introduced-panel recall lever — split to
  **#1192**.
- **#1167** — `Speaker NN` leaking into the KG as Person. Fixed via the "exclude from surfaces"
  direction (read-side guard `is_unresolved_speaker_placeholder`), and importantly **by #1127
  (RFC-088), not #1190**. Two residuals (write-side nodes still written; MCP surface still leaks)
  split to **#1193**.

**Created this session:**

- **#1191** — GI route-and-tag (no pipeline truncation); the durable fix behind the interim GI cap.
- **#1192** — speaker recall lever for un-introduced panel guests (the ~113 tail; precision/recall
  trade).
- **#1193** — KG garbage nodes: model unnamed voices as non-Person (write-side) + close the MCP
  read-side leak + sweep other placeholder nodes (`podcast:unknown`, empty-name Persons).

**Kept open with a "delivered X / remaining Y" status comment (verified NOT closeable):**

- **#1188** — cleaning cross-promos. Delivered: the ad-readers-as-hosts *consequence* is mitigated by
  metadata-first host detection (ADR-110). Remaining (the core): the Athletic cross-promo still
  survives `PatternBasedCleaner` — no cross-promo detector built; issue is explicitly parked.
- **#1173** — timestamp drift. Delivered: the root-caused **fix** (`b293f3d9`, word-level segment-time
  refinement). Remaining: the acceptance ACs — the 46-fixture RTTM drift harness (AC1), the
  p95≤300ms/max≤1000ms verification (AC2), prod-safety check (AC3), regression test (AC4). Fix in,
  proof not.
- **#1179** — transcription/diarization deathmatch. Delivered: the *summary/GI* bake-off (a different
  eval). Remaining: the 5-stack ASR/diarization table — blocked on children **#1174/#1177/#1178**.
- **#1177** — MOSS provider. Partial/gated (operator-confirmed); stays open.

**Method note for the next PR:** its body **must** carry explicit `Closes #N` for each issue it
completes — do not repeat the #1190/#1111 miss.

---

## 4. Where the authoritative detail lives

- Fuse + GI-cap numbers, model, rebase-conflict files → `HANDOVER-2026-07-16-fuse-gi-cap-nextpr.md`.
- Bake-off results + the re-run/judge state → `BAKEOFF-18EP-RESULTS.md`.
- GI truncation design (the #1191 target) → `GI_WHAT_TO_SURFACE.md`.
- v2↔v3 corpus comparison → `CORPUS_COMPARE_V2_V3_PILOT.md`.
- Fixture ladder + the KG-segments audit → `CORPUS-V4-FIXTURE-LADDER.md`.
