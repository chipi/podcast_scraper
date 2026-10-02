# Advisor review pack — naming + pipeline fixes (2026-10-01 → 2026-10-02)

Running record of what was fixed, what was measured, and what is deliberately left open. The
"Questions for the advisor" section is the point of the review: each one is a decision we held
back because the evidence did not settle it.

All measurements are on prod (`sha-0ee4580`, ~2,050 served episodes) unless stated. "Replay" means
the faithful roster replay: production's roster code re-run over each stored episode, fed the LLM
answers the pipeline stored in `.speakers.diagnostics.json` (no new LLM calls).

## Commits in scope (local `wt-kg`, not yet pushed unless marked)

Pushed to main earlier (`fcdae675d` and before), not deployed:

| Commit | What |
|---|---|
| `3325f9ea7` | person_web prune |
| `7b4222b70` | host seat: whitespace / hypothetical host statements |
| `ddb41645c` | "my name's", "for today", reported speech in self-intro detection |
| `9e63c53b4` | two-voice show: the host-introduced stated guest is named |
| `dd8835954` | transcript filename bounded to 255 bytes + write-failure recorded |
| `fcdae675d` | KG reply that is not JSON logs WHERE at WARNING |

Local, not pushed:

| Commit | What | Evidence |
|---|---|---|
| `c53565843` | roster: a vacant host seat is not filled by the guest who owns the talk (≥50% with a guest present) nor by a voice under 5% | replay 2,051 eps: +46 names (39 Tyler Cowen), −41 (ads, clips, org names, guests with a host's name), 7 renamed. Known regressions: 2 correct names lost, 1 wrong gain. "Stop at the dominant voice" variant measured and rejected (−5 real co-hosts) |
| `df2369773` | compose: pass `APP_MCP_RESOURCE_URLS` to the player api | obs connector authorize returned 400 `invalid_target` (07:46/07:47Z); the env value was staged on 09-05 but never reached the container |
| `6661b186a` | GI: a quote whose transcript was relabelled is re-anchored onto its text | 936 warnings/night = the same 36 Odd Lots episodes (relabelled 09-28) re-warned by every whole-corpus enrich pass; 1,997 of 2,021 quotes occur exactly once → recovered on the next run |
| `57c3ec09d` | bundled quotes: bounded JSON schema on vLLM + a decoding loop is salvaged, not bisected | 13/13 captured failures were a filler loop ("like, you know, …") at 5120/5120 with presence_penalty 1.5 on. Replay of a looping batch ×3: json_object looped 1/3 (all 24 quotes lost), schema 0/3 (24/24 resolved). The 1200s "hangs" (#2040/#1983) all completed — they were this loop |
| `e50b5879f` | third-person guard: "I am your host, X" and "I am Rob X" count as self-introductions | 28 discards hand-reviewed, ~17 removed a correct name |
| (item 4) | spans: the span helper no longer replaces the block's exception with `RuntimeError: generator didn't stop after throw()`; `episode.transcribe` + `episode.metadata` spans; a handled failure marks its span; one `Multi-feed run summary` line per run | reproduced the RuntimeError locally; 0 hits in 30 days of prod logs (latent). `episode.process` covers only the download (max 11s) |

## Questions for the advisor

1. **Third-person guard vs a guest's own surname.** About 12 of the 17 wrong discards are the guest
   on the dominant voice (54–84% of talk) uttering their own surname or a relative's (quoting a
   headline about themselves, "my mother", "my first husband's name"), or the host's closing
   "Name, thank you" merged into the guest's cluster. Last night two of them hid 61 and 56
   insights. Candidate rule: a metadata-stated guest name on the voice with ≥50% of the words is
   not refuted by mentions. Risk: the guard exists for the Unhedged case where a co-host (~50% on
   a two-host show) was named after the person discussed (Jay Powell). Is dominance + "stated
   guest" + "voice does not self-introduce as someone else" enough, or is there a better signal?

2. **Junk "names" in the candidate list.** A case-sensitive surname match (to stop "a black
   t-shirt" refuting Sue Black) was replayed: +2 correct names, +6 junk title fragments ("AI White
   House", "Claude Code", "Commodity Context", "Russian Spring", "Treasury Foreign Exchange",
   "How Football Shirts") placed on large voices. The case-blind match was rejecting them by
   accident. Where should non-person candidates be stopped — at detection, or at placement?

3. **Show hosts vs episode hosts (#2092).** The Daily states three hosts; on most episodes one is
   present. The seat guard only fires when substantial voices outnumber stated hosts, so a
   reporter-guest at 61% is seated as host and goes unnamed (34 insights hidden on one episode).
   Should the seat count come from the episode (who self-introduces) rather than the feed?

4. **Seat guard known regressions** (`c53565843`): a correct guest name reached through
   elimination is lost when the dominant voice is no longer seated, and an archive clip can take
   the vacant seat. Acceptable trade for the gains, or is there a cleaner formulation?

5. **Feed-description host parser** (held patch): finds 18 correct hosts in 10 feeds, but fed to
   the roster before the seat guard it put names on guests and ads. Re-test now that the seat
   guard exists, or keep hosts-from-description out of placement entirely?

6. **The in-flight deadline alarm.** `timeout_context` logs ERROR "DEADLINE EXCEEDED … STILL
   RUNNING" when the deadline passes, while the work continues. All three on 2026-10-02 then
   completed (the decoding loop), and the ERROR level is what files #2040/#1983 repeatedly. The
   comment keeps it at ERROR deliberately: it was the only signal during a real 4h15m wedge, and
   alerting keys on it. Candidate: WARNING at the deadline, ERROR only at a multiple of it (the
   2026-10-02 overruns finished at ≤2.6×; the wedge was ~13×). Changes an alerting contract, so
   held.

7. **Holistic:** where is naming thin overall — seats, missing names, misspellings (Bernard
   Liang/Leong, Kittrow-F), duplicates — and are the last ~10 commits sound (due diligence)?

## Observability item 5 (config / noise) — decided, not changed

- **HF_TOKEN unset:** one Hub metadata ping per model load against a cached model. `pipeline-llm`
  is built with `PRELOAD_ML_MODELS=false`; MiniLM + nli-deberta live in the `hf_cache` volume,
  downloaded on first use. Both load offline in the real image (verified), but `HF_HUB_OFFLINE=1`
  would break a fresh volume. Options for the operator: an `HF_TOKEN` secret, or bake the models
  into the image (+640 MB) and then go offline.
- **SELF-GRADING:** the value gate's rater is the generator because the DGX serves one model.
  Added to `DGX-DEFERRED.md` (#1895).
- **insight_salvage over ceiling (84/night):** the model returns ~30 for a ceiling of 25 and the
  salvage keeps 25 spread across the episode. A schema `maxItems` would cut the tail of each
  transcript slice instead, which is worse. Left as is.
- **topic-clusters skipped:** fixed (`848bee921`) — a multi-feed feed no longer looks for an index
  in its run directory.

## Corrections to earlier claims

- `6661b186a`'s message says the 36 Odd Lots episodes heal "on the first pipeline run after
  deploy". Wrong for the nightly: a multi-feed feed finalizes only its own run directory
  (`enrich-edges: episodes=4`). The whole-corpus pass is the finalize of a SINGLE-feed Jobs-API
  run (`path=/app/output`) — that is what re-warned 36 episodes 26 times — so they heal on the
  first single-feed job after deploy (e.g. the DEEPEN jobs).

## Not done / not verified

- Kennedy Center (34 insights) traced only to a hypothesis (question 3); Planet Money (4) not looked at.
- The re-anchor and loop fixes are verified by tests and prod replays, not yet by a live run.
- Observability item 4 is fixed in code (`47fc7a77f`) but not yet seen in VictoriaTraces.
