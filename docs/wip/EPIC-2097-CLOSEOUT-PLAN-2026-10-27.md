# Epic #2097 close-out plan (2026-10-27)

What the read-only pass of 2026-10-27 (naming plan item 15) left open, and how each item gets
confirmed and closed. Starts after the naming release's own steps are finished (0034/0035/0036,
the 0030 re-run, verify, ownership, sidecar refresh, incremental index).

**Rules that bind every step:** prod corpus fixes are text-file fixes (`CLAUDE.md`; fix ladder:
read-time rule → deterministic migration over the artifacts → targeted index-row edits → re-run
deterministic non-LLM steps for the affected episodes). Nothing here relabels, re-diarizes,
re-ingests or calls an LLM / the DGX. Whatever text on disk genuinely cannot fix goes to
`docs/wip/DGX-DEFERRED.md` with the evidence, and stops there. A backfill only follows a pipeline
that already handles the case for new episodes. Every corpus write goes through
`docs/guides/CORPUS_CHANGE_METHOD.md` (outcome table, fresh reviewer, the operator's go).
Closing a GitHub issue needs the operator's go.

## Baseline (read-only, prod at 2.7.30, 2026-10-27)

| Item | Finding | How measured |
|---|---|---|
| #2082 transcript pairing | 23 of 3,544 metadata files point at another episode's transcript: 15 credit the other episode's people, 8 unclassified; **all 23 are superseded runs, 0 served** (`select_served_artifacts`). The audit's own route split (14 / 9) is not a plan. Feeds: Pragmatic Engineer 5, No Priors 4, Latent Space 4, Invest Like the Best 2, NVIDIA AI 2, Planet Money 2, The Journal 2, Hard Fork 1, The Daily 1 | `scripts/audit/transcript_pairing_audit.py --corpus-dir /app/output` (2914e56), exit 1 |
| #2100 speakerless publisher episodes | Served (`select_served_artifacts`, each metadata's own `transcript_file_path`): Explaining Brazil 47/47, The Flip 42/42, Korea Deconstructed 13/13 with speakers, all Whisper + diarized (Korea `a353187e` is a solo lecture, one voice correct). The 0-speaker publisher versions exist only in superseded runs. (A first scan derived transcript names from the metadata stem and reported "Explaining Brazil 9": wrong.) | served-corpus scan, 2026-10-27 |
| The 51 text-prefixed publisher transcripts | 0 of 3,006 served episodes with ≥3 unlabelled `Speaker N:` prefixes (via each `transcript_file_path`) | served-corpus scan |
| #2094 D4 / D5 / D6 | D6 0 (fixed); D5 still swapped (Ground Truths Couric/Topol); D4 43 candidates (`d4_candidates.json` on the mini). D4 + D5 with the naming session | served-corpus scan |
| Epic activities 6-7 (migrations run; #2075 measurements re-derived) | migrations: corpus at 2.7.30, 0 pending, 0032 held (#2380). Measurements: not re-derived | `upgrade status` |

## 1. #2082: the 23 metadata files that point at another episode's transcript

1. **Is the pipeline still producing them?** For each of the 23, the run date and the code path
   that wrote the pointer (the reprocess idx collision fixed under #2082?). Confirm today's code
   cannot produce a new one (a test or a replay on a fresh run). If it can, the fix goes to the
   pipeline first and the backfill waits for it.
2. **Classify the 8 unclassified** by opening both transcripts and the roster for each (names
   in neither / both). Output: confirmed misattribution, stale-pointer-but-right, or harmless.
3. **For each confirmed episode, what text on disk supports:**
   - its own transcript exists on disk (another run, the audio-archive provenance, the publisher
     transcript) → a deterministic migration repoints `content.transcript_file_path` and rebuilds
     the derived surfaces from that transcript (speakers, GI/KG/bridge people, context, sidecars,
     the episode's index rows);
   - its own transcript exists nowhere on disk → the wrong people come off its served surfaces
     (a text fix: no credit beats a wrong credit), and the episode goes to `DGX-DEFERRED.md` with
     the evidence that its text is missing.
4. One migration (or a read-time rule) under CORPUS_CHANGE_METHOD; dry run, outcome table, fresh
   reviewer, the operator's go; verify; ownership; derived surfaces.
5. **Done when:** the audit exits 0, or every remaining hit is listed in `DGX-DEFERRED.md` with
   evidence and shows no wrong person on any served surface.

## 2. #2100: the 51 publisher episodes listed as speakerless

1. Reconcile `DGX-DEFERRED.md`'s 51 (by episode id from the 2026-10-01 census) with what is served
   now: for each id, the served run, its speaker count, and what gave it speakers (which run, which
   stage). (Done 2026-10-27: all 102 served episodes in the three feeds have speakers; the row is
   removed from `DGX-DEFERRED.md`.)
2. Any id that still has no speakers: confirm the text-on-disk evidence again (publisher original
   has no voice tags / `Name:` cues) and keep it on `DGX-DEFERRED.md`; every id that now has
   speakers comes off the list with the evidence.
3. **Done when:** `DGX-DEFERRED.md`'s #2100 row matches the served corpus id by id, and any
   episode that dropped out of serving is explained or restored.

## 3. The 51 text-prefixed publisher transcripts

1. The scan found 0 on served runs. Confirm by id against the 2026-09-26 list (epic comment):
   for each, the served run's segments carry `speaker` and no embedded `Speaker N:` prefix, and the
   roster names the guest the title names.
2. **Done when:** all 51 ids check out, or each that does not is fixed (text: the cue prefix
   becomes `seg["speaker"]`, the parser fix `d6c0de341` replayed over the stored file) under
   CORPUS_CHANGE_METHOD.

## 4. #2094 D4 / D5 / D6 against today's artifacts

1. D5 (Ground Truths Topol / Couric swap) and D6 (Tyler placeholder speaker): read the current
   served artifacts for those episodes. Fixed → evidence on the issue. Not fixed → the pipeline
   cause first (does today's code still produce it on a new run?), then a text backfill.
2. D4 (a 2-4% ad-read cluster takes the guest's name; 29 episodes): count on today's corpus with
   the talk floor (0034) applied — the floor removes names under 5% of talk, which is D4's shape.
   What remains is a pipeline defect (#2096) first, then a backfill.
3. **Done when:** each of D4 / D5 / D6 has a measured count on today's corpus and either 0 or a
   fix in place.

## 5. Epic activities 6-7: re-derive the #2075 measurements

1. After 1-4, re-run the measurements quoted in #2093, #2095 and the #2075 PR (the coherence check
   and the role-transition tables) on the prod corpus, restricted to episodes the pairing audit
   clears, and post the new numbers next to the old ones.
2. **Done when:** every number the epic marks as needing re-derivation has a post-repair value.

## Closing

When 1-5 are done: a final read-only pass (audit, scan, D4-D6 counts), the evidence posted on
#2082, #2094, #2100 and #2097, and the operator decides which issues close.

## Not covered by this plan

- #2096 (the attribution-quality epic) beyond D4's pipeline fix.
- The DGX-deferred items themselves: they stay on the list until the operator decides.
- The order between items 1 and 4 is not fixed: D4's count may change after #2082's repointing.
