# Post-deploy list — fixes of 2026-10-01 / 2026-10-02

Everything that has to happen, or be checked, once the local commits on `wt-kg` (on top of
`fcdae675d`) are pushed and deployed. Nothing here has run live yet. Every prod WRITE below needs
per-instance operator approval; every cleanup is dry-run first, then applied, then verified.

Companion: `ADVISOR-REVIEW-2026-10-02.md` (what was fixed, evidence, open questions).

## A. Before / during deploy

| # | Step | Why |
|---|---|---|
| A1 | Push the local commits to main (operator approval) | ~25 commits, none pushed |
| A2 | Deploy ALL prod surfaces at one sha (operator approval) | the player `api` needs `df2369773` (compose passes `APP_MCP_RESOURCE_URLS`) — a config change, so the player surface must be redeployed, not only images |
| A3 | Read the deploy log for the obs MCP line | `4b1232b84`: it must say "authorize accepts the resource". A WARN with `authorize-accepts-resource=no` means the env did not reach the api |
| A4 | Force-build the ad signatures once, right after deploy: `write_for_corpus(Path("/app/output"), force=True)` in the pipeline container (operator approval — writes only `search/ad_signatures.json`, deterministic, no LLM) | otherwise the file first appears at the END of the next multi-feed batch, so that nightly runs with no ad opinion, B2/B3 slip a day, and C3 has no source (advisor review) |

## B. Live verification (read-only)

| # | Check | How | Expected | Fix commit |
|---|---|---|---|---|
| B1 | Obs MCP connector | claude.ai → reconnect "Close Listening O11Y" at `https://obs.closelistening.app/mcp`; then call a tool | login completes; a tool returns prod data | `df2369773` |
| B2 | Ad signatures built | after the first corpus finalize: `search/ad_signatures.json` exists; log line `ad signatures: N recurring passages, ad languages ['de']` | ~2,400 passages, `de` learned, `es`/`pt` not | `40d8d9844` |
| B3 | German / house ads not named | first new episode of an affected show (Past Present Future, In Our Time, The Daily) | ad voices typed `commercial`, no host name on them | `40d8d9844` |
| B4 | Quote re-anchor | after the first SINGLE-feed Jobs-API job (whole-corpus finalize): `add_spoken_by_edges` warnings | from 36 episodes / 2,021 quotes per run to ~24 quotes (20 duplicated, 4 absent) + an INFO "re-anchored" line | `6661b186a` |
| B5 | Bundled-quote loop | VictoriaLogs: `extract_quotes_bundled parse FAILED` with `DOCUMENT_ENDED_EARLY`; `decoding loop` warnings | parse failures from loops ~0; any loop logs "kept N closed quote(s) … not retrying" | `57c3ec09d` |
| B6 | Deadline overruns | `DEADLINE EXCEEDED` on long episodes (Latent Space) | fewer; when present, followed by "OVERRAN … COMPLETED" | `57c3ec09d` |
| B7 | Episode spans | VictoriaTraces: `episode.transcribe`, `episode.metadata` spans with `run_id`/`episode_id` | present; a failed transcription is an ERROR span | `47fc7a77f` |
| B8 | Run summary | nightly log: one `Multi-feed run summary: feeds= ok= failed= not_started= episodes_processed=` line | present at the end of every run | `47fc7a77f` |
| B9 | Topic clusters | nightly log | `topic-clusters + ad signatures: deferred to the multi-feed batch finalize` instead of a WARN per feed | `848bee921` |
| B10 | Filename length | re-run the Design Meets Business smoke episode that failed with `[Errno 36] File name too long` | the episode is written | `dd8835954` |
| B11 | Naming warnings | VictoriaLogs counts of "TALKS ABOUT" discards and "a name WAS available" | the self-intro forms ("I am your host, X", "I am Rob X") no longer discarded | `e50b5879f` |
| B12 | Name gate | new episodes: no "Host", "OK", committee / job-title names in `content.speakers`; real names with role-word parts still published (Christopher Guest class) | none refused wrongly | `c069bc827`, `0651c7b2e`, `5b4ebc091` |
| B13 | Named-voice rate | per nightly: named voices / substantive voices, vs the prior 7 nights | no drop — an over-rejecting gate shows here first (advisor) | `c069bc827`, `5b4ebc091` |
| B14 | Commercial voices per episode | before vs after deploy, per feed | a rise on the German-ad / house-ad shows only; a rise elsewhere = signature over-reach (cross-posts, syndicated clips) | `40d8d9844` |

## C. Text-only cleanups of what is already on prod

Allowed by the operator rule because the pipeline handles each case correctly once deployed (we only
backfill what today's pipeline gets right). Each is a deterministic migration over the artifacts —
no LLM, no DGX, no relabel, no rebuild — on the m0012 model: a FROZEN set decided at apply time and
written to receipts; every file backed up under `.podcast_scraper/upgrade-backups/<id>/`; `undo`
restores a file only while it is still exactly what the migration wrote; `verify` against the frozen
set. Order: dry-run → operator approval → apply → verify.

| # | Cleanup | Evidence (prod, 2026-10-02) | Surfaces (all five, or none — as m0012) |
|---|---|---|---|
| C1 | **Junk published names** — a new migration (`m0015`) that removes every published speaker name `is_publishable_speaker_name` now refuses: "Host" ×20, "OK" ×3, "Thank", "Right", "GE", committees and job titles ("House Select Committee", "PC Alexander Committee", "Alexander Committee", "Treasury Foreign Exchange", "Meter Redwood Research", "Commodity Context", "Roblox CEO"), unplaced org hosts ("The China-Global South Project" ×10) | 45 of 4,638 published names | metadata `content.speakers` + `detected_*`; segments + adfree segments `speaker_label` (voice stays, `voice_type: unknown`); kg Person node + edges; gi Person node, `SPOKEN_BY`, quote `speaker_id`, insight `speaker` (+ route/tag recompute); bridge identity row |
| C2 | **Prefix repairs** (person_web drops the old id's bio on its next run and fetches the 3 new ids under budget — no separate enrichment step needed) — 3 names keep the person, lose the job / show: "Your Host Luisa Leni" → Luisa Leni, "Deputy Editor Eilish Hart" → Eilish Hart, "Planet Money's Kenny Malone" → Kenny Malone | 3 | a RENAME across the same surfaces, using the m0010 canonical-name path (ids derive from names, so node ids and edges move together) |
| C3 | **Ad readers named as people** — the frozen set records the signature file's `built_at` + SHA-256 in the m0015 receipt (the file is rebuilt every 6h; a later rebuild must not change what was applied); voices hit by the LANGUAGE rule alone (not recurrence) are hand-checked before applying — they are the ones that can be content — names on voices the corpus ad signatures classify as ads: Daniel Atkinson (Wirecutter) on dozens of episodes, Jonathan Knight (NYT Games), Shannon Maldonado (Shopify), "Gemini"/"Claude" (AI promo), Vox's promo voice, host names on German ads | replay: 90 names removed; the frozen set must be re-derived from the BUILT `ad_signatures.json` (B2), never from a replay | same five surfaces as C1; `voice_type: commercial` instead of `unknown` |
| — | **C1 voice type** | the pipeline types an unnamed voice `cameo` (<20s), `unidentified` (nobody could name it) or `unknown` (we failed to); m0012 wrote `unknown` for all. C1 computes the type the pipeline would, or records the divergence in the receipt (advisor) | — |
| — | **C1 runs only on the gate as of `5b4ebc091`** | the earlier gate refused real names whose parts are role words (Christopher Guest); deriving C1's set from it would remove real people from prod | — |
| C4 | **Odd Lots quote offsets** | 36 episodes, 2,021 quotes | self-heals on the first single-feed job's whole-corpus `enrich-edges` (B4). If none runs soon: an `enrich-edges` over the corpus is a deterministic, non-LLM step (ladder step 4) — operator approval |

NOT in C (would need a roster re-run, i.e. a relabel — out of bounds unless the operator raises it):
the 39 Tyler Cowen seats, David Runciman's own-voice seats, and every other "gained" name in the
replays. They are correct going forward for NEW episodes only.

After C1–C3, every derived surface built from what changed must agree: the bridge (done inside the
migration), the search-index rows that carry speaker names (targeted in-place edit of the affected
rows, ladder step 3 — measure how many first), and the person enrichers (`person_web` etc.) for the
affected names only (ladder step 4).

## D. Operations

| # | Item | State (2026-10-02 11:22 UTC) |
|---|---|---|
| D1 | Nightly `ed9ab40e` | 12 of 55 feeds, 42 episodes in 8h20m (~5 episodes/h); ~30h left at this pace — overlaps tonight's 03:00 slot. Operator decides ~midnight whether to delay the nightly. |
| D2 | DEEPEN (18 shows → 10 episodes) | scheduler alive, waiting for the nightly to finish; at this pace starts tomorrow |
| D3 | Design Meets Business | re-run after deploy (B10); freshness gate ~2026-10-11 |

## E. Held — not post-deploy work

Advisor questions 1–8 (seat logic, third-person guard dominance, description hosts, spelling
variants, deadline alarm level); DGX list (`DGX-DEFERRED.md`); housekeeping (commit the replay
harness to `scripts/`, delete scratch files holding real episode text, remove the `wt-kg` worktree).
