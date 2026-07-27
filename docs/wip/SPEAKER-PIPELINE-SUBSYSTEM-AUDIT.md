# Speaker / Diarization / Cleaning Subsystem Audit

**Date:** 2026-07-25
**Scope:** The whole accumulated speaker-and-transcript subsystem — diarization, speaker
naming/labeling, and "identifying bullshit" (ads / cross-promos / filler / boilerplate) — not
just recent changes. Produced by four parallel read-only audit agents (diarization, naming,
ad-cleaning, and a branch-diff closure pass), then the load-bearing findings re-verified by hand.

**Verification legend:** `[V-hand]` re-verified in the code by the synthesiser; `[V-agent]`
reproduced end-to-end by the auditing agent; `[reported]` agent-read, not independently re-run.

---

## Cross-cutting themes (the shape of the debt)

1. **The DGX / primary path silently diverges from the local path.** Tuning knobs that fix
   diarization quality are applied only to local pyannote and are dead on the DGX tier that
   production actually uses; the diarization cache key can't tell two providers apart; retry
   posture differs 5 ways across backends. A "quality" knob whose effect depends on *which tier
   answered* is a config-vs-behaviour contradiction. **This is why the community-1 boundary
   purity that dropped Kara's name can't be tuned away on the production path.**
2. **Built-tested-and-dead capabilities.** Diarization-aware cleaning (cross-promo excision +
   sponsor diarization signals) is fully built and unit-tested but unreachable on the default
   `hybrid` summarization path. Its tests assert the *dead* state as correct.
3. **Silent degradation feeding downstream as ground truth.** Ad detection swallows all
   exceptions and returns `[]`; the recurring-text index freezes stale; LLM resolution returns
   `{}` on error — and the roster consumes all of these as fact with no way to see they degraded.
4. **Accumulated heuristic sprawl.** Multiple overlapping regex families and cleanup functions
   answer the same question with different guards/thresholds — a footgun for the next call site.

---

## DIARIZATION (`providers/ml/diarization/*`, `providers/tailnet_dgx/*`)

- **D1 — HIGH `[V-hand]` — pyannote tuning knobs are dead on the DGX (primary) tier.**
  `diarization_clustering_threshold` / `diarization_min_cluster_size` / `diarization_min_segment_ms`
  are applied only inside `PyAnnoteDiarizationProvider` (local). The DGX client
  (`tailnet_dgx/diarization_provider.py`) sends only `num_speakers`/`min_speakers`/`max_speakers`;
  the DGX `/v1/diarize` server has no clustering/squelch params. So the #1170 phantom-speaker /
  over-segmentation fix has **zero effect on the DGX-healthy path**, yet
  `diarization_config_fingerprint` still hashes these knobs (so changing them busts the cache and
  re-runs DGX for nothing). **Directly implicated in the v2.2 host-turn-merge problem.**
- **D2 — HIGH `[V-hand]` — the diarization cache key omits the provider backend.**
  `diarization_config_fingerprint` hashes `diarization_model` + speaker bounds only — never
  `diarization_provider` / `dgx_diarize_model` / `moss_model` / `deepgram_/gemini_` model. Reprocess
  the same audio under `local` then `tailnet_dgx` (or deepgram) without touching those fields and
  the second run silently serves the **first provider's** diarization under the second's name.
- **D3 — MEDIUM `[reported]` — `_feed_recurring_text` freezes at first-call state for the process
  lifetime.** Module-level dict keyed by `output_dir`, never refreshed. On a fresh feed's first
  full pass (the common case), episodes 2..N run the mid-roll-ad heuristic against the near-empty
  evidence that existed when episode 1 started.
- **D4 — MEDIUM `[reported]` — HOLD strategy yields zero-retry for 3 of 5 backends.** The factory
  refuses to build the fallback chain under HOLD, trusting the provider's own hold-and-probe — but
  only `tailnet_dgx`/`moss` have it. `local`/`gemini`/`deepgram` then fail an episode outright on
  one transient blip, strictly worse than failover.
- **D5 — LOW `[reported]` — Deepgram silently ignores speaker-count hints** (accepts the params,
  never sends them; undocumented, unlike moss which documents the same no-op).
- **D6 — LOW `[reported]` — squelch can erase a whole episode's diarization; the WARNING the
  operator sees doesn't say why** (the "dropped N phantom speakers" reason is DEBUG-only).

## SPEAKER NAMING (`speaker_detectors/*`, `diarization/roster.py`)

- **N1 — HIGH `[V-agent]` — `_canonicalize_to_known_host` swaps a real guest's identity with the
  host's.** It runs over EVERY voice's self-intro and snaps any name whose surname is within
  edit-distance-3 / soundex / stem of a known host onto that host. Reproduced end-to-end: a guest
  self-introducing "I'm Kevin Ross" (known host "Kevin Roose") →
  `guest voice → name="Kevin Roose", role=host` and `host voice → name="Kevin Ross", role=guest`.
  A full identity swap — the #876 "wrong name on a voice" failure. Fix direction: gate
  canonicalization on host-likelihood (`host_hint_voices`/opener), not every voice.
- **N2 — HIGH `[V-hand]` — `guests_introduced_by_the_host` is ungated and feeds the forced
  one-name-one-voice assignment.** It scans ALL voices' text (not just hosts) for greeting /
  introduction patterns and unions the names into the guest pool. A guest quoting *"Sarah Chen,
  thanks so much for coming to my defense…"* adds "Sarah Chen"; if exactly one spare name + one
  unassigned voice remain, that unrelated voice is force-named "Sarah Chen." The `_GUEST_GREETED`
  pattern (added in the v2.1 work) widened this surface, since greeting phrasing is common in
  ordinary speech. Directly contradicts the deliberately-removed "anchor" rule documented right
  below it (roster.py:1105-1124).
- **N3 — HIGH but currently DEAD `[reported]` — `detect_hosts_from_transcript_intro` loses its
  capitalisation signal to `re.IGNORECASE`** ("I'm going to explain…" → captured as a host name).
  Unreachable today (no live call site passes `transcript_text`), but exported + untested and would
  reactivate the bug if rewired.
- **N4 — MEDIUM `[reported]` — leftover-voice role (guest vs unknown) decided by unrelated
  evidence.** `has_guest_intro = any(vid not in assigned for vid in voice_intro)` is true if ANY
  voice anywhere self-introduced, so a no-evidence cameo/tape voice can be labelled `guest`.
  Doesn't misname, but corrupts role-based aggregates (talk-share-by-role, host/guest counts).
- **N5 — MEDIUM `[V-hand]` (from the branch fix) — `host_hint_voices` opener fallback trusts the
  first non-ad speaker before roles are known.** Computed before `conv_guests`; if a guest opens
  (cold-open clip), that guest is trusted as a host hint → greeting reclamation can move onto the
  guest cluster and the weaker intro forms can misname. Edge case in the just-committed #1290 fix.
- **Drift:** two self-intro regex families (one network/person-guarded, one not); three
  independent name-cleanup implementations (`normalization._sanitize_person_name`,
  `hosts._clean_stated_name`, roster's own wrappers); three org/mononym filters differing only in
  which guard is included (swap-by-accident footgun).

## CLEANING / "IDENTIFYING BULLSHIT" (`cleaning/*`, `gi/ad_regions.py`, `adfree_transcript.py`)

- **C1 — HIGH `[V-hand]` — the default `hybrid` cleaner drops diarization context, so cross-promo
  excision + sponsor diarization signals are DEAD for summaries.** `transcript_cleaning_strategy`
  defaults to `hybrid`; `HybridCleaner.clean(text, provider, metrics)` has no diarization param and
  calls `pattern_cleaner.clean(text)` with text only; `metadata_generation.py` forwards
  `**pattern_clean_kwargs` on the PatternBasedCleaner branch but NOT the HybridCleaner branch. So
  an opening cross-promo is correctly stripped from the GI/KG/search `.adfree.txt` base (separate,
  correctly-wired path) but survives into the **summary**. The existing tests assert
  `diarization_segments=None` as *expected*, baking in the gap.
- **C2 — MEDIUM-HIGH `[reported]` — `excise_ad_regions` runs 3× on 3 different text derivations**
  (raw whisper segments in `pipeline._ad_intervals`; fully-formatted screenplay in
  `adfree_transcript`; post-`clean_for_summarization` text in `PatternBasedCleaner`). Same
  thresholds, different strings → the `ad_intervals` the roster uses to type "commercial" voices
  can disagree with the boundaries actually excised into `.adfree.txt`. No test asserts they agree.
- **C3 — MEDIUM `[reported]` — ad-free write failures swallowed at DEBUG.**
  `save_adfree_artifacts` catches `OSError`, logs DEBUG, returns None; downstream silently falls
  back to the raw ad-laden transcript with no operator-visible signal.
- **C4 — MEDIUM `[reported]` — the LLM cleaner has no content-fidelity guard** beyond an
  output/input length ratio, even though the codebase already knows (ad_regions docstring) that LLM
  rewriting disguises ads.
- **Drift / dead code:** four independently-maintained ad/sponsor pattern lists (`SPONSOR_PATTERNS`,
  `gi.filters._AD_PATTERNS` — a private symbol imported cross-package, `DEFAULT_PROMO_CUE_PATTERNS`,
  `preprocessing` credit/outro/garbage lists) with overlapping vocabulary; threshold drift on the
  same `_AD_PATTERNS` (2-hit insight filter vs 3-hit excision); `PatternBasedCleaner.remove_sponsors()`
  / `remove_outros()` have no production caller (tested dead code).

## RECENT-BRANCH CLOSURE (the 7 commits `feat/1188-cleaning-crosspromos`)

- **B1 — MEDIUM `[reported]` — relabel_only / rediarize_only pick the newest-mtime transcript
  across ALL `run_*` dirs in the feed** with no run-tag correlation and no listing of skipped
  alternates. The documented reprocess workflow routinely leaves multiple run dirs on disk.
- **B2 — hygiene** — branch is behind `origin/main`; rebase before any push (rule #2).
- **B3 — docs** — `CORPUS_REPROCESSING.md` has no row for `relabel_only`/`rediarize_only`
  (tracked open in `docs/wip/RELABEL-ONLY-OPTION-PROPER-JOB.md`).
- **Verified clean:** secrets scan (0 hits); obs (`dev_push`/`otel_init`) truly inert when env
  unset — no prod/Docker leak; RSS `follow_redirects` correctly wired; embedding-device never-MPS
  clean; deleted backfill scripts leave no dangling references; config/CLI pipeline_stage plumbing
  correct.

---

## Proposed fix order (before v2.2 run)

**Tier 1 — correctness, fix before any scale run (these can put wrong names on voices):**
- N1 canonicalization identity swap (gate on host-likelihood) — repro test first.
- N2 ungated `guests_introduced_by_the_host` → forced hallucination (host-gate the greeting feed).
- N5 `host_hint_voices` opener-trusts-guest edge (fold the conv_guest signal in / abstain).
- C1 hybrid cleaner drops diarization (wire `pattern_clean_kwargs` through HybridCleaner).

**Tier 2 — silent-divergence / data-integrity (wrong results, not wrong names):**
- D2 cache key omits provider backend (add provider fields to the fingerprint).
- D1 DGX tuning knobs dead (either forward them to the DGX server — infra change — or make the
  client refuse/warn when they're set against a provider that ignores them).
- C2 three `excise_ad_regions` derivations diverge (single source of truth for ad boundaries).

**Tier 3 — robustness / observability:**
- D3 recurring-text staleness, D4 HOLD zero-retry, C3/C4 swallowed failures + LLM fidelity,
  D5/D6 silent no-ops → surface them (log/raise), B1 reprocess run-dir selection.

**Tier 4 — hygiene / dead code / drift:** N3 dead IGNORECASE fn (remove or fix), N4 role leakage,
consolidate the name-cleanup + ad-pattern families, remove dead wrappers, B3 docs, golden-fixture
diversity.

**Then:** Fable-5 advisor architectural review → decide v2.2.
