# Anomaly log — the 2026-09 batch ingest (Batch A top-up to 40 + Batch B onboarding)

Running record of everything observed during the unattended ingest, with root-cause analysis as
each is researched. Written so the review afterwards is against evidence rather than memory.

**Status key:** `OPEN` needs research · `RCA` cause established, fix not made · `FIXED` hotfixed
to main (lands on next deploy) · `TRACKED` has a GitHub issue · `NOT-A-DEFECT` investigated and
dismissed, kept so it is not re-investigated.

**Context.** Prod ran `prod_dgx_full` (pinned corpus-wide in `viewer_operator.yaml` on 2026-09-04)
against DGX vLLM `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`. Corpus 1,066 → 1,145+ episodes over the
run. DGX averaged 88% GPU utilisation throughout, which is the backdrop for several entries here.

---

## A1 — One episode produced insights with ZERO quotes

**Status:** `TRACKED` — [#1970](https://github.com/chipi/podcast_scraper/issues/1970)
**Observed:** 2026-09-04T17:01:03, one episode of ~77.

```
grounding produced NOTHING: 6 insights, 0 quotes. Either the grounder is disconnected
(the 513-insights-zero-quotes signature — check the evidence-provider align: model_copy
skips validators) OR the grounding calls failed transiently (e.g. APIConnectionError on
quote extraction)
→ intent gate: 1 uncited acceptance — Invariant violation
```

**Why it matters.** Insights with no supporting quotes are ungrounded assertions. The invariant
caught it and said so — this is the no-silent-fail contract working — but the episode shipped with
6 insights nothing corroborates.

**What is NOT yet known.** Which of the two branches fired. The message names both and they need
different fixes: a disconnected grounder is a code defect (`model_copy` skipping validators), a
transient `APIConnectionError` is a retry/resilience gap. One occurrence in ~77 episodes points at
transient, but that is an inference from frequency, not evidence.

**Next step.** Identify the episode (search `bridge_partition.gi_only > 0` across the run's
episodes), then check whether its GI artifact shows quote-extraction errors in the same window.

---

## A2 — The value gate self-grades on every DGX episode

**Status:** `RCA`
**Observed:** 8+ times, every episode of the run.

```
value gate: SELF-GRADING — vllm is rating its own output with
'NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4'. This is lenient (~10% of insights dropped vs
~25% for a distinct rater); counts are not comparable with a curated-rater run.
```

**Root cause — DELIBERATE, not an oversight.** `providers/ml/model_registry.py`
(`_PREFERRED_GATE_MODEL`) states it outright:

> `# qwen / groq: ungoverned routes with no settled stronger sibling yet. They self-grade and say`
> `# so in the log until someone measures which sibling is worth the spend.`

The cloud providers have curated raters (`deepseek` → `deepseek-v4-pro`, `openai` → `gpt-5.4`,
…); vLLM/qwen has none because nobody has measured which sibling is worth the spend. Omitting the
gate model on an uncurated provider yields `(True, None)` — "gate on, no distinct rater" — which
is the documented contract, and the log warning exists precisely so this is visible rather than
silent. So this is an OPEN MEASUREMENT, not a missed config line, and a blind pin would be
guessing at exactly the question the comment says to measure.

An initial draft of this entry said `prod_dgx_full` "was not part of that pass", implying an
oversight. That was wrong and is corrected here rather than deleted, because the wrong reading is
the tempting one.

**Consequence, quantified by the log itself.** ~10% of insights dropped versus ~25% with a
distinct rater — the DGX corpus is graded **~2.5× more permissively** than the cloud corpus. Any
comparison of insight counts between DGX-ingested and cloud-ingested episodes is invalid until
this is closed.

**Why this is not urgent-stop.** It is lenient, not wrong: it keeps insights a stricter rater
would drop. Nothing is lost, quality of the retained set is unchanged.

---

## A3 — `METADATA_SEC_PER_1K_TRANSCRIPT_WORDS` is DGX-measured but governs cloud too

**Status:** `RCA` (documented in-code; no profile closes it)

`utils/timeout_config.py:216` sets `150.0` sec per 1k transcript words. Its own docstring:

> "As of 2026-09-03 **no shipped profile sets that key** (grep `config/` — zero hits), so this
> DGX-measured 150.0 currently governs every profile, cloud included. The commit that added the
> override advertised 'stop a DGX-measured rate governing cloud'; what it actually did was make
> stopping it possible… the mechanism is not the fix."

**Direction is benign** — the deadline is `max(flat, scaled)`, so an unmeasured cloud rate can
only lengthen a budget, never shorten one. Closing it requires *measuring* a cloud rate, which is
work, not a config edit.

---

## A4 — Metadata deadline overruns at the 1200s floor

**Status:** `NOT-A-DEFECT` (saturation signal)
**Observed:** 3 × `DEADLINE EXCEEDED … longer than 1200s and is STILL RUNNING`.

Initially flagged as "#1920 may have been closed without being fixed". **That was wrong.** The fix
is deployed: `get_metadata_generation_timeout()` returns `max(flat, scaled)` with
`scaled = (words/1000) × 150`. Because of the `max`, any episode under ~8,000 words keeps a 1200s
deadline — so a message citing 1200s is the *expected* shape for a short episode, not evidence the
flat deadline survived.

**What it actually indicates:** short episodes exceeding 20 minutes of summary+GI+KG, with the DGX
at 88% average utilisation. Contention, not a mis-sized budget. Recorded so it is not
re-investigated as a deadline bug.

---

## A5 — Incremental indexing never removes rows from superseded runs

**Status:** `TRACKED` — [#1969](https://github.com/chipi/podcast_scraper/issues/1969) — the highest-value finding here
**Observed:** the full rebuild on 2026-09-04 dropped 3,149 derived vectors while episode coverage
stayed identical.

```
episode_title / episode_description / summary_short   1,067 → 1,067   +0
insight −1,024 · quote −1,699 · kg_topic −297 · kg_entity −102 · summary −24 · transcript −3
total 157,995 → 154,846
```

**Root cause.** The old index was built incrementally over months. Doc types split exactly along
ID stability:

* **per-episode docs** (`episode_title`, …) have stable episode-keyed IDs → overwritten on
  reindex → unchanged
* **derived docs** (`insight`, `quote`, `kg_topic`, `kg_entity`) are keyed by content/source → a
  reprocessed episode yields *different* IDs → the old run's rows are never deleted

`dedupe_metadata_paths_newest_run_per_episode` picks the newest run at *discovery*, so a full
rebuild indexes only current artifacts — but incremental indexing has no delete pass.

**Arithmetic corroborates:** 3,149 ÷ ~104 derived vectors/episode ≈ **30 episodes** with a
superseded run. Independently, one feed's ingest logged `32 metadata files, 31 distinct GUIDs` —
one such episode in a single feed.

**Consequence.** Between full rebuilds, search returns insights and quotes from pipeline runs that
no longer exist in the corpus. Silent: counts look larger, not smaller, so nothing alerts.

**Corpus integrity is unaffected** — `/api/ops/corpus/integrity` → PASS, 1,067 scanned, 1,067
healthy_gi, 0 missing/unreadable. This is an index-hygiene defect, not a data-loss one.

---

## A6 — `/api/index/stats` cannot see in-flight indexing

**Status:** `RCA` (ops-visibility)

During the 1h47m rebuild the endpoint sat frozen at `12,663 / 74 episodes` for over 20 minutes
while the job was healthy and burning a full core. It reads the last *committed* LanceDB table
state; the indexer commits in bulk. This produced a false "the reindex is stuck" read and a bogus
5-hour ETA extrapolated from a counter that does not track progress.

**Signals that DO track progress**, for whoever watches the next one:

* `search` dir size via `/api/ops/corpus/usage` (`by_directory`) — grew steadily 2.3–2.8 MB/min
* container CPU via cadvisor (`compose-api-run-*`) — pinned at ~1 core, dropped to 0 at completion

---

## A7 — Recurring LiteLLM auth failures (self-resolved / partly self-inflicted)

**Status:** `NOT-A-DEFECT` for the observed window, but worth a note

40 `auth_exception_handler … user_api_key_auth()` events over 72h to 2026-09-04, in bursts of
identical-second retries. **Zero in the 13h since.** Separately, a monitor polling
`/api/ops/gateway/auth` every 60s produced a 429 that looked exactly like a gateway outage — that
endpoint makes a live upstream call, so polling it frequently rate-limits it. At least some of the
historical bursts are plausibly the same shape (a poller, not an attack), but that is unproven.

---

## A8 — `topic_consensus` and `topic_similarity` disabled and stale

**Status:** `TRACKED` (#1921 for topic_consensus)

Both `enabled: false`; `topic_consensus` last ran 2026-08-24 with `last_status: timeout`,
`topic_similarity` 2026-08-24. Ten-plus days stale through the whole batch, so the 160 new
episodes get no consensus or similarity signal. `topic_consensus` is the enricher that most needs
multiple speakers per topic — the exact value the Batch B geographic expansion is meant to create.

---

## A9 — Insight overgeneration

**Status:** `TRACKED` (#1891)

149 occurrences in 13h. The model returns 32/25, 31/25 per chunked pass (and 174/50 unchunked on a
laptop run). `a624e773` deliberately made this *cheap and visible* rather than fixed —
`GI_INSIGHT_TOKENS_EACH` 150→50 caps a runaway, salvage recovers intact lines, counters make it
countable. The real fix (making the model respect the count) is #1891, still open. Not a
regression; recorded so the warning volume is not mistaken for one.
