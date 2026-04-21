# Next Session Plan (2.6 remaining)

## Order of work

1. **Whisper API cost optimization (#577)** — ~half day
   - Bitrate sweep (5 eps × 5 bitrates)
   - Local vs API breakeven analysis
   - Update profile presets with optimal settings

2. **Mega-bundle experiment (#632)** — ~1 hour
   - Single LLM call for summary + GI + KG
   - Score per-field quality vs standalone baselines
   - If promising: test on gemini + anthropic

3. **Post re-ingestion validation** — ~3 hours
   - Depends on user providing new production corpus
   - Validate all 5 explore expansion CLI commands
   - Insight clustering quality + bridge merge rate
   - See `POST_REINGESTION_PLAN.md`

4. **Viewer UI follow-up (#609)** — last, 2.6 close-out
   - Insight cluster integration in viewer
   - Cluster browse, quote search, topic-insight matrix in UI
   - Separate TypeScript/Vue work

## Follow-ups noted during #646 / Phase 3C validation (not yet scheduled)

- **Real-episode end-to-end validation for single-feed corpus layout
  (#644).** 4 unit tests in
  `tests/unit/podcast_scraper/test_service_single_feed_corpus_layout.py`
  confirm `single_feed_uses_corpus_layout=True` wraps `output_dir` with
  `feeds/<slug>/run_*/`, but no real scraper run has exercised it. Same
  class of gap as #643 Phase 3C. Work: scrape one RSS feed with and
  without the flag, diff the on-disk trees, confirm single-feed output
  matches multi-feed shape exactly. Add an integration test that runs the
  scraper in dry-run against a mocked RSS to assert directory layout.
  Blast radius: silent divergence between single- and multi-feed corpora
  (the original bug this flag was supposed to fix).
- **Real-episode end-to-end validation for audio preprocessing profile
  wiring (#634/#642).** Research itself was validated against real fixtures
  (`data/eval/runs/_bitrate_sweep/`, `_silence_sweep/`) and profile
  resolution has 8 unit tests, but no real scraped episode was ever run
  end-to-end with `speech_optimal_v1` to confirm the downstream ffmpeg
  call actually uses 32 kbps + -30dB/0.5s. Same class of gap as #643
  Phase 3C (wiring-not-validated). Work: hook logging on preprocessing
  stage, run one real episode through `cloud_balanced`, confirm ffmpeg
  args + file-size reduction + WER within 1 % of reference, add an
  integration test to prevent regression. Blast radius of ignoring:
  silent cost regression, no crash.
