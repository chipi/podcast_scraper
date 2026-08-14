# Stabilization phase — 2026-08-14

**Status:** OPEN — ingestion deliberately paused.

A hold on corpus growth to fix what we found while growing it. Ingestion resumes only
when the exit criteria below are met.

The rationale, stated precisely: **enrichment is corpus-wide and re-runnable, so its
correctness does not depend on when it runs. Per-episode artifacts are not.** Summaries,
insights and KG nodes are minted at ingest time by whatever the pipeline is that day. If
a per-episode defect exists, every episode ingested before it is found carries it. So the
pause protects the per-episode layer, not the enrichment layer.

---

## Entry state (frozen 2026-08-14 ~07:00Z)

| | |
| --- | --- |
| Corpus | **662 episodes**, 14 feeds |
| Running | `1ebba1af` — The Pragmatic Engineer, off=17 n=16 |
| Queued | `0260ddba` — The Pragmatic Engineer, off=33 n=16 |
| Queued | `6d9d9c6f` — `corpus_enrichment`, `--only <7 deterministic>` (diagnostic) |
| Cancelled | six new-feed windows — see "Resume" below |

The two Pragmatic Engineer windows are the tail of the previous batch (17+16=33, 33+16=49,
landing the show at its 50 target). They were left to run: cancelling mid-batch would have
left the show at a ragged offset for no benefit.

### Cancelled, to be re-queued verbatim on resume

Cancelled via `POST /api/jobs/{id}/cancel` while `queued` — a clean dequeue, no SIGTERM.
Re-queueing costs nothing: `skip_existing` is GUID-keyed corpus-wide, so a re-issued job
is idempotent against whatever already landed.

| Show | Feed URL | offset | max_episodes |
| --- | --- | --- | --- |
| Odd Lots | `https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/8a94442e-5a74-4fa2-8b8d-ae27003a8d6b/982f5071-765c-403d-969d-ae27003a8d83/podcast.rss` | 0 | 25 |
| Odd Lots | *(same)* | 25 | 25 |
| Conversations with Tyler | `https://rss.libsyn.com/shows/137081/destinations/850607.xml` | 0 | 25 |
| Conversations with Tyler | *(same)* | 25 | 25 |
| In Our Time | `https://podcasts.files.bbci.co.uk/b006qykl.rss` | 0 | 15 |
| In Our Time | *(same)* | 15 | 15 |

All with `skip_existing=true`, `episode_order=newest`. Window sizes follow the audio-minutes
rule from **G-opt** (`window ≈ 1400 / median_minutes`): Odd Lots 50m→25, CwT 55m→25,
In Our Time 54m→15.

---

## Exit criteria — what "stable" means

Ingestion resumes when **all** of these hold. Each must be evidenced by a command and its
output, not by assertion.

1. **Enrichment produces data on the real corpus.** A named, non-empty enricher set runs,
   and `GET /api/corpus/enrichments` shows artifacts for it. (H2)
2. **An empty enricher set cannot silently succeed.** The no-op path is loud — the failure
   mode that hid this for the corpus's entire life is closed, with a test.
3. **The per-episode layer has been audited** on a sample spanning several feeds: summary
   substance, insight/KG density, GI↔KG bridging, and ad/sponsor contamination. Findings
   filed in the homework doc.
4. **No unexplained job failure** in the current queue. `failed_total` is understood
   line-by-line, not just counted.

Explicitly **not** exit criteria: H1 (audio archive) is blocked on operator-supplied
secrets and must not gate the phase; H4 is recorded-only by operator decision.

---

## Work items

### S1 — fix enrichment the royal route — `[IN PROGRESS]`

Homework **H2**. Advisor engaged on the design. Working hypothesis of the causal chain:

| # | Location | What happens |
| --- | --- | --- |
| 1 | `config.py:4097` | `profile_name = data.pop("profile", None)` — name is known here |
| 2 | `config.py:4173` | `profile_dict.pop("profile", None)` — discarded, because `Config` is `extra="forbid"` with no `profile` field |
| 3 | `orchestration.py:1791` | `profile = getattr(cfg, "profile", None) or block.get("profile")` → both empty |
| 4 | `profile_sets.py:159` | `enricher_set_for_profile(None)` → empty set → zero enrichers, reported as success |

Constraints on the fix:
- `profile_sets.py` stays the source of truth. The `enrichment:` YAML block is advisory
  documentation mirroring it, **not** an input to resolution.
- The profile must survive **both** invocation paths, one of which crosses a **subprocess
  boundary** (`_maybe_spawn_enrichment_after_pipeline` spawns a child). An in-memory
  attribute alone does not survive that.
- Royal route only — no shim, no hardcoded profile.

### S2 — verify enrichment on the real corpus — `[BLOCKED on S1]`

The queued `6d9d9c6f` (`--only <7 deterministic>`) is a *diagnostic*, not the fix: it
bypasses profile resolution by naming enrichers explicitly. Its value is proving the
executor and the enrichers themselves work. Do not read a green result from it as evidence
that S1 is fixed — those are different code paths.

### S3 — per-episode quality audit — `[TODO]`

Sample across feeds and measure what the pipeline actually produces. Prior tooling for
this shape of question exists from the onboarding calibration work. Attention on ad/sponsor
copy leaking into summaries, since several of the 14 feeds are ad-supported.

### S4 — homework items worth doing in this window — `[TODO]`

| Item | Why now |
| --- | --- |
| **D6** — every budget signal measures spend, none measures headroom | Cap was just raised to $10 and ~130 episodes are about to be queued against it |
| **H1** — audio archive built, documented, switched off | **Blocked on operator**: two GitHub secrets. Every episode ingested meanwhile discards audio, the one artifact we cannot regenerate |
| **H3** — post-ingest passes architecturally inconsistent | Directly adjacent to S1; whatever S1 touches informs it |

### S5 — reviews — `[TODO]`

Advisor / reviewer passes on whatever S1 changes before it merges.

---

## Log

- **07:00Z** — six new-feed windows cancelled; phase opened; advisor engaged on S1.
