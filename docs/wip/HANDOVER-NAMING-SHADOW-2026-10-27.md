# Handover: what runs in shadow on prod, and how to evaluate it (2026-10-27)

For the next agent picking up the naming arc after the 2026-10-27 deploy (`sha-2914e56`, main
`2914e562e`). **Updated at the end of 2026-10-27**: see "What waits for this evaluation" below; the
next deploy is the operator's first test of a new deploy method, so nothing new reaches prod before
it. Prod is the deploy; this file is what to measure from the new episodes it ingests.
Everything here is from the repo at `2290fa91a`; numbers marked *step 5* are from the pre-deploy
measurement recorded in `docs/wip/NAMING-RELEASE-POST-DEPLOY-PLAN-2026-10-25.md` (items 19-21).

## What is in shadow

Set in `config/profiles/prod_dgx_full.yaml` (comment above the keys says why):

| Key | Mode | What it is | Where its answer is written |
|---|---|---|---|
| `transcript_naming` | `shadow` | An LLM reads the whole diarized transcript and names every voice (`providers/ml/diarization/transcript_naming.py`). | speaker-diagnostics sidecar, key `transcript_naming` |
| `voice_identity_rung` | `shadow` | Last naming rung: a voice that sounds like a person this feed's other episodes name (fingerprint groups named by vote, same feed only) takes that name; a name leaves a voice that clearly is not that person. Threshold `voice_identity_threshold` 0.7. Design: `docs/wip/VOICE-IDENTITY-DESIGN-2026-10-12.md`. | speaker-diagnostics sidecar, key `voice_identity` |

In shadow the **ladder publishes**; each side namer's answer is only recorded beside it. `publish`
would change the served roster. Field docs: `src/podcast_scraper/config.py` (`transcript_naming`,
`voice_identity_rung`).

Two caveats from the config docs:
- In the bundled LLM modes (`llm_pipeline_mode` bundled / mega_bundled / extraction_bundled) the
  bundled call answers instead of the transcript-naming call, and its answer is recorded only:
  `publish` would publish nothing more than `shadow` there until the bundled write-back exists.
- The sidecar file is `<base>.speakers.diagnostics.json` under the run's `transcripts/`.

## What is published (not shadow) but still needs a post-deploy check

Also in the prod profile, live on new episodes since the deploy:
- `speaker_name_min_talk_share: 0.05`: the talk floor (a name on a voice under 5% of the talk
  comes off). Served episodes got the same through migration 0034.
- `ladder_rescue_rules: true`: the E1d and S30s rescues.
- `merge_split_voices: true` at `merge_split_voices_threshold: 0.8`: two voices of one episode
  whose fingerprints are 0.8 similar are merged as one person. The profile comment says 0.7 would
  merge more but needs a post-deploy merge audit first (plan item 19).
- `speaker_detection_temperature: 0.0`.

## How to evaluate

1. **The naming shadow report** (read only, sidecars only):
   `python scripts/measure/naming_shadow_report.py --corpus /app/output [--json out.json] [--limit N]`
   (module `src/podcast_scraper/quality/naming_shadow.py`, whose docstring defines every column).
   Three tables:
   - **Side namers:** per namer, episodes, calls that answered, and per voice agree / adds /
     removes / changes against the ladder's published name. Every non-agree voice is listed. An
     episode where the namer did not answer counts under `not_answered` by reason; a voice it never
     judged is `not_judged` (the LLM pass skips voices under 40 words; the rung skips voices without
     a fingerprint or under `voice_identity.MIN_VOICE_TALK_S`). Episodes written before these keys
     existed count as `without`, never as agreement.
   - **Talk floor:** every name the floor took off, by the role the LLM pass gave that voice (a
     proxy: an `ad` / `promo` voice losing a name is the point; a `host` losing one is a loss to
     read).
   - **Rescues:** every voice E1d or S30s published, with the LLM's name for it beside it.
   Each row carries the episode's insight count on that voice: read the rows with the most
   insights first.
2. **Counts do not say who is right.** Agree / changes only compare namers. To judge right or
   wrong, read the listed voices against the transcript (blind reads, as for 0034 / 0035), or score
   against gold:
   `python scripts/measure/gold_insights.py ... --namer llm|voice|ladder` scores a shadow namer's
   names in place of the GI's stored ones (S7 / S8: insight-weighted right / wrong / unnamed).
   `--namer ladder` scores the roster the run served.
3. **The promotion bar** is the operator's decision, from a table of right / wrong per namer on
   new episodes. Rank a wrong name as worse than a hidden insight (operator's rule).

## Where things stand (step 5, pre-deploy)

- **Voice rung**, filling only unnamed voices: 33 right / 5 bad, and 5 wrong names repeated within
  a feed (the error-spreading check: one wrong group name spreads to every episode of the feed).
- **Transcript naming**, filling unnamed: 103 right / 43 bad; publishing everywhere 437 right / 75
  bad, against the ladder's 360 / 51.
- The operator's ruling: **transcript naming stays in shadow**; its data accumulates.

## What waits for this evaluation

- **0032's 91 uncovered voices** (48 additions, 29 renames, 14 unnamings; 0032 itself is retired,
  `docs/wip/0032-ASSESSMENT-2026-10-27.md`). Operator, 2026-10-27: their reviewed list comes AFTER
  this evaluation, because what the rungs and the LLM pass say on new episodes decides whether
  re-deriving old episodes is worth it. Same machinery as 0037 (frozen list, two blind readers with
  names hidden, verdicts bound to the list); the 29 renames first.
- **The rungs' reach on old episodes:** archived audio covers 2,950 of 2,980 episodes (plan item 22,
  closed on prod 2026-10-26; 18 size mismatches, 30 none).
- **First shadow data exists:** prod's `naming_shadow_report` JSON from night 1
  (`~markodragoljevic/prod-naming-20261027/naming_shadow_20261027.json` on the mini, plan item 12).
  Night 1's blind read (21 episodes, private eval repo `data/eval/naming-night1-20261027/`) found
  the ladder's served names wrong on 4 voices in 3 episodes, now fixed by 0037 (not yet applied).
- **Reviews in this evaluation follow method v0.9 section 4a**: blind reads are evidence, not a
  review; a review reads for at most 5 minutes and produces no numbers.

## Next steps for the evaluating agent

1. After a week of nightlies, run the shadow report on prod (read only, the prod session runs it)
   and keep the JSON.
2. For each side namer, blind-read a seeded sample of the non-agree voices, weighted by insights.
   Report right / wrong / undecidable per namer, plus the error-spreading count for the rung.
3. Audit the split-voice merges at 0.8 on new episodes (plan item 19) before 0.7 is considered.
4. Archived audio is known (2,950 of 2,980): use it to bound the rung's reach on old episodes.
5. Bring the table to the operator; promotion (`publish`) is his call, then a profile change and its
   own deploy.

## Monitoring and evaluation: decided after the rollout to the new infrastructure

Operator, 2026-10-28: nothing below starts before the move to the new infrastructure and the other
pending changes are done; then he decides each one. The fixes list is in
`HANDOVER-NAMING-DEBT-2026-10-27.md`, section F.

| # | What | When / trigger | Status |
|---|---|---|---|
| S1 | **Shadow evaluation** of the transcript-naming LLM pass and the voice identity rung (this file's "How to evaluate") | after about a week of nightlies on the deployed code | not started |
| S2 | **0032's 91 uncovered voices** (48 additions, 29 renames, 14 unnamings) as a reviewed list | after S1: what the rungs say on new episodes decides whether re-deriving old ones is worth it | not started |
| S3 | **Corpus-scope enrichers after the 2026-10-28 nightly** (open item 7): none stale for `corpus_artifacts_newer` after 0037's rewrite | the prod session reads `corpus_status` after the nightly | prod's check |
| S4 | **The known wrong-name causes recurring on new episodes**: a feed host forced on the interviewer's voice (D5) or on a clip voice (`d99b4720`); the LLM's pair swap (`216929530`); a guest name on a clip voice (`1732482`) | a blind read of a fresh sample of new episodes, as night 1 did (21 episodes, private eval repo `data/eval/naming-night1-20261027/`); count each shape | not started |
| S5 | **The talk floor's host exemption** (a seated host the LLM also calls host keeps its name under 5%): 2 of D4's 43 candidates kept a name this way, 1 of them undecidable | a count of `kept_host` traces on new episodes and a read of those | not started |

Reviews in this work follow method v0.9 section 4a; blind reads are evidence, not a review.

## NOT covered by this handover

- No shadow data from after the deploy has been read yet: the numbers above are pre-deploy.
- The shadow report's reading of the bundled modes was not checked here.
- Whether the prod session can run `naming_shadow_report.py` from the image (no mounted source:
  compose-api-1 runs the image's site-packages, per the prod session 2026-10-27) was not tried.
- The gold sets' coverage of new feeds was not checked.
