# Handover to the triage-fleet agent — signal-quality feedback (2026-08-27)

**From:** the v2.7 cleanup pass (processed ~60 open issues, deep-triaged 30 milestoned to v2.7).
**Why:** the auto-triage is genuinely useful and every issue was traceable — but a few systematic patterns turned ~27 real signals into ~50+ issues and mixed real bugs with working-as-designed noise. Fixing the top three would roughly halve issue volume and make the rest trustworthy. Each item below is grounded in the specific issues it came from, with the actual data, so you can verify and reproduce.

---

## 1. Your dedup key is too GRANULAR — normalize the signal before hashing (highest leverage)

I first assumed you "have a stable `fp` but ignore it." Wrong — I checked the fingerprints across each cluster and **they are all different**, because the fingerprint embeds volatile per-occurrence tokens (episode hashes, byte sizes, run IDs, paths). So GlitchTip groups every occurrence separately, and you dutifully file a new issue for each.

**Evidence — one bug, one fingerprint per occurrence:**

Audio-eviction "cold vs local size mismatch" (**10 issues**, #1840–1849):
```
#1840  fp: glitchtip:PODCAST-PIPELINE-3F
#1841  fp: glitchtip:PODCAST-PIPELINE-3E
#1842  fp: glitchtip:PODCAST-PIPELINE-3D
#1849  fp: glitchtip:PODCAST-PIPELINE-36
```
…because the messages differ only in volatile data:
```
#1845  "Audio eviction size mismatch: cold (69782850) != local (65937149) for …"
#1843  "cold size != local size (~72KB …)"
#1844  "KEEP /app/output/feeds/rss_feeds.npr.org_7… "
```

ADR-148 summary re-roll (**9 issues**, #1556…#1866) — even worse, spanning TWO fingerprint schemes:
```
#1556  fp: glitchtip:PODCAST-BS          (old scheme)
#1576  fp: glitchtip:PODCAST-D1          (old scheme)
#1820  fp: glitchtip:PODCAST-PIPELINE-31 (new scheme)
#1861  fp: glitchtip:PODCAST-PIPELINE-8  (new scheme)
```
The message embeds the episode hash: `…for episode 6a319f2260728bbcda06b463`. Different episode → different message → different group → new issue.

OpenAIProvider-not-initialized (**5 issues**, #1570, 1578–1581): `PODCAST-CH, D7, D5, D4, D6` — five groups, one bug.

**Do:** don't trust GlitchTip's per-event fingerprint for issue-dedup. Compute your OWN stable key by **normalizing** first: strip volatile tokens before hashing —
- episode/work IDs (`6a319f2260728bbcda06b463`, `6eb845ef-3075-4858-9dbf-4dda8eca28fd`),
- byte counts / dollar amounts (`69782850`, `$8.9874`),
- paths (`/app/output/feeds/rss_…`), run IDs (`run-20260826T092912.861891Z`), timestamps.

Then key on `(area, exception_type, normalized_message_skeleton, top_app_stack_frame)`. Dedup on THAT. Before filing, search open issues for the same normalized key; if found, increment an occurrence counter + append the raw instance as a comment. **This one change removes ~25 of the ~60 issues.** (I had to reconstruct these clusters by hand-reading titles; a normalized key would have collapsed them at source.)

## 2. Severity must track RECOVERABILITY, not "it logged at ERROR"

You treat every ERROR log as a bug. But the pipeline deliberately logs some ERRORs for *handled/recoverable* conditions.

**Worked example — #1556.** Your own body text says:
> *"Trace expired but error is recoverable, suggesting pipeline handled it gracefully but still logged as error."*

You noticed it was recoverable and filed a `code-invariant` bug anyway, with acceptance criteria *"_generate_and_validate_summary robustly handles the schema … or the schema is corrected."* I traced it: the handling is already robust and deliberate — `parse_summary_output` (3 strategies + repair) → ADR-148 in-place re-roll (`providers/guardrails/reroll.py`) → ADR-100 fallover → then ONE loud error, which is an explicit operator directive (`workflow/metadata_generation.py:3646-3669`, Marko 2026-08-20: "never acceptable to have an episode without a summary"). The remedy is **reprocessing the episode**, not a code fix. So the correct signal was: *recoverable degradation, WARN-level, low priority, ops-action = reprocess.*

**Two more:** #1855 — the repair **succeeded** (`repaired 18/19 · 1 NOT FOUND`), i.e. the system *working*, filed as a data-integrity bug. #1409 — a guardrail firing on `finish_reason_length` is the guardrail *doing its job*.

**Do:** this repo emits recoverability markers you can read directly — `RecoverableSummarizationError`, `record_stage_outcome(stage, idx, "degraded"|"failed")`. If the exception/outcome is `recoverable`/`degraded`, weight it WARN + low-priority (or roll it up, don't file). Reserve ERROR-filed issues for `failed`/terminal/unrecoverable.

## 3. Separate EXTERNAL / ENVIRONMENT / TRANSIENT from CODE BUG

Several signals are outside the app's control, yet each got a `code-invariant` acceptance criterion.

| Issue | Actual signal | True class | Right acceptance criterion |
| --- | --- | --- | --- |
| #1529 | `Deepgram transcription failed: The write operation timed out` (count=3) | external-transient | "retry w/ backoff; on exhaustion WARN + degrade" — not "fix invariant" |
| #1480 | `BadRequestError 400 data_inspection_failed` (OpenAI safety reject) | external | "catch → skip/fallback that episode + WARN" |
| #1854 | `Multi-feed run finished with one or more feed failures … chat podcast-flash-0731` | transient/config | improve message first (see #5), then ops |
| #1546 | `LanceError(IO): Permission denied … /root/.cargo/registry/…` | environment | container/image fix (CARGO_HOME/perms) — not code |
| #1345 | single unsymbolicated iOS `SIGABRT: Signal 6` | low-signal | hold until symbolicated / recurs (see #4) |

**Do:** tag each new signal with a class — `code-bug` / `graceful-handling-needed` / `external-transient` / `environment` / `low-signal` — and template the acceptance criteria off the class. An `environment` signal is not a code invariant.

## 4. Occurrence-count + affected-users gating before filing

You filed single-occurrence, unsymbolicated, "no users affected yet" events as full issues.

**Worked example — #1345:** one event, `iPhone18,3`, stack unsymbolicated, filed as a bug. Un-actionable as-is. Contrast #1570 (12 occurrences in 52s) or #1529 (count=3) — those cross a bar. **Do:** hold a signal until ≥N occurrences OR ≥1 user affected OR an actionable/symbolicated stack; aggregate sub-threshold events onto a rollup issue.

## 5. Capture the underlying `__cause__`, not the wrapper message

**Worked example — #1854:** the entire signal is *"Multi-feed run finished with one or more feed failures. Culprit feed: chat podcast-flash-0731."* — it never says *why* the feed failed. Untriageable without re-running. The real error is one layer down (the per-feed exception `service.py` caught). **Do:** capture the innermost exception + its top app-stack frame, not just the outermost handler's summary. Same issue on several summary/init signals — the actionable exception was 1–2 layers below the reported message.

## 6. Stamp the code version / commit at signal time

Stack line numbers had drifted badly by the time I read them: #1570 cited `clean_transcript` at `openai_provider.py:2774` (actually ~2893 now); #1556 cited `_generate_and_validate_summary:3390` (actually 3543). I had to re-locate every frame. **Do:** include the `code_version` / commit sha (already exposed at `/api/health` as `code_version`) so frames resolve against the revision that fired.

## 7. Cluster by area/stack-signature and cross-link at file time

Even without perfect normalization (#1), a cheap grouping would have caught the clusters: the 10 audio-eviction issues all carry `Area: podcast-pipeline … eviction` and near-identical top frames; the 9 ADR-148 issues all cite `_generate_and_validate_summary` / `RecoverableSummarizationError`. **Do:** on file, search recent open issues by `Area:` + top frame and either fold or add "possibly related to #N."

## 8. Auto-assign area label + a triage milestone

184 of ~200 open issues had **no milestone** and most had no area label — invisible to release planning until this manual pass moved 23 bugs into v2.7 by hand. **Do:** map your `Area:` field to a repo label and drop new signals into a default `triage` milestone.

---

## Net
**#1 (normalize the dedup key)** is the single highest-leverage change — it removes ~40% of the volume outright, and the evidence above shows exactly why the current per-event fingerprint can't do it. **#2 and #3** (recoverability + external/transient classes) stop you asking humans to "fix" code that is already correct (#1556, #1855, #1409, #1546). The rest sharpen the signal. Nothing here is about the *quality* of what you file — it's traceable and the acceptance criteria are a good draft — it's purely **"one occurrence ≠ one issue"** and **"ERROR-logged ≠ code bug."**

Companion artifacts from this pass: `docs/wip/V2.7-BUG-TRIAGE-2026-08-27.md` (per-issue classification of all 30) and `docs/wip/ISSUE-CLEANUP-2026-08-27.md` (the consolidation record: which dups were closed into which canonical).
