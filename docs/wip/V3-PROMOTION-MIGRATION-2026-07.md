# v3-promotion migration — make v3 the active fixture corpus (#1148)

Operator decision (2026-07-06): **promote v3 to the active fixture version, "make it work for
all, no way back."** The enricher content (6 scripted episodes, structures, gold, corpus_meta)
lives in v3; the app/viewer corpus historically built from **v2**. This migration flips the
whole substrate to v3.

## Done

- `tests/fixtures/FIXTURES_VERSION`: **v2 → v3**.
- `build_app_validation_corpus`: wired to **6 shows × 4 eps** (added p04/p01/p07 to `APP_SHOWS`,
  bumped `--max-feeds 6 --max-episodes-per-feed 4`, added risk-management/systems-thinking
  umbrellas to `CROSS_CUTTING_TOPICS`).
- App corpus rebuilt from v3 → `tests/fixtures/app-validation-corpus/v3/` — **24 episodes, 6
  shows** (panel + all enricher content now in the built corpus).
- v3 generator: scripted-dialogue path + 6 natural episodes + corpus_meta (3 seeded users,
  shared_topics, contradiction_pairs, corpus gold). Deterministic, lint + mypy clean.
- Additive-preservation verified: regenerating v3 changed **only the manifest** among existing
  files; every existing episode's transcript/ground-truth is byte-identical (detection targets
  intact).

## Remaining

1. **Reconcile hardcoded-v2 references** (the real ones — most `/v2` grep hits are API/OAuth
   version false positives):
   - `tests/fixtures/app-validation-corpus/v2/` — orphaned; retire once tests read v3.
   - App/viewer e2e specs asserting specific v2 titles/entities/counts:
     `app/e2e/smoke.spec.ts`, `app/playwright.config.ts`,
     `web/gi-kg-viewer/e2e/validation/real-corpus.spec.ts`, `web/gi-kg-viewer/TESTING.md`.
   - `tests/fixtures/baselines/v2-metrics.json`, `tests/fixtures/app-validation-corpus/README.md`.
   - Measure breakage by running the suites; reconcile assertions to v3 content.
2. **Rebuild the viewer-validation corpus from v3** (`tests/fixtures/viewer-validation-corpus/v2/`
   → v3). NB `scripts/dev/upgrade_viewer_validation_corpus_to_v3.py` is a *GI-schema* v3 upgrade
   (MENTIONS/ABOUT edges, #1075) — a naming collision, NOT the transcript-version rebuild.
3. **Run the enricher loop** on the v3 corpus: enrichers (embeddings for topic_similarity,
   DeBERTa for nli — local, not CI) → scorers → `write_gate_metrics` → gate reads real numbers.
   Reconcile the authored gold (`expected_enrichment`) against measured output.
4. **Player + viewer e2e** over the v3 corpus; full CI green.
5. **Regenerate TTS audio** for the v3 episodes (operator: after scripts settle) — the scripted
   episodes carry `#fixture-v3: voice=` hints; realize them via ElevenLabs (#993).
6. **Retire orphaned v2 artifacts** once everything reads v3.

## Enricher-loop findings (2026-07-06) — the fixture↔enricher reconciliation

Ran the deterministic enrichers over `app-validation-corpus/v3` via the enrichment CLI.
They execute (status=ok) but **every corpus-scope enricher emits 0** — the fixture GI/KG is
built for the consumer read surfaces, not shaped for the enrichers. Three concrete gaps in
`build_app_validation_corpus`, each fixable:

1. **Persons are `person:speaker-NN`** (raw diarization ids), not canonical. `guest_coappearance`
   filters speaker-NN by design → 0 pairs; `grounding_rate` finds 0 persons for the same reason.
   *Fix:* canonicalize speakers to stable cross-episode person ids (map the diarization roster to
   the known guest names the transcript header already carries).
2. **Publish dates hardcoded** to `2026-01-{day}` — the builder ignores the authored
   `publish_offset_days`. All episodes land in one month → `temporal_velocity` flat.
   *Fix:* read `publish_offset_days` from ground-truth and stamp `CORPUS_EPOCH + offset`.
3. **Topic cooccurrence 0** — topic id/edge scheme in the fixture KG doesn't feed the
   corpus-scope topic enrichers; needs alignment.

Also: `topic_similarity` needs an embedding run (`--with-ml`, sentence-transformers local);
`nli` needs DeBERTa. Once (1)–(3) land, the reference scorers (guest_coappearance / grounding /
topic_similarity) can grade real output → real `gate_metrics`. Until then the loop is wired but
reads empty. Reference scorers registered; per-enricher gold authored.

## Risk / blast radius

~29 files reference the active/v2 corpus; many are false positives. The genuine reconciliation
is the app + viewer e2e specs (content-specific assertions) and the two committed corpus dirs.
Staged: app corpus (done) → tests/e2e reconcile → viewer corpus → loop → audio.
