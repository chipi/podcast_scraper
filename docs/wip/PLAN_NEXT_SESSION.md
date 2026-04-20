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
