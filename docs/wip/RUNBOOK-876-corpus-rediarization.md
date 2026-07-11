# Runbook — #876 corpus re-diarization (100 episodes, DGX)

**Status:** Ready to execute once PR #944 merges. Drafted 2026-06-09.

Re-process the ~100-episode `whisper_transcription` corpus through **DGX Whisper
(large-v3) + DGX pyannote (speaker-diarization-community-1; 3.1 fallback)** so episodes gain real
multi-speaker diarization + named screenplays, and the relational layer
(corpus-wide `SPOKEN_BY`) is re-derived.

## Mechanism

> ⚠️ **Use `make migrate-diarization`, NOT `make redo-diarization`.** The two targets
> differ by a single flag with a huge consequence:
>
> | Target | Has `--reprocess-existing-only`? | What it actually processes |
> | --- | --- | --- |
> | `make migrate-diarization` | **yes** | the corpus's existing on-disk GUIDs (the whisper episodes) — correct |
> | `make redo-diarization` | **no** | scrapes the **live feed** and processes the newest items (583-episode feed → wrong episodes) |
>
> Verified 2026-06-11: invoking the raw `cli` re-diar command *without*
> `--reprocess-existing-only` logged `Episodes to process: 2 of 583` and started
> downloading brand-new feed episodes instead of re-diarizing the corpus. Always
> confirm the log line reads `Existing-only re-diarization (#876): kept N, dropped …`.

`make migrate-diarization CORPUS_DIR=… PROFILE=…` runs two steps:

1. `cli --config <PROFILE> --feeds-spec <CORPUS_DIR>/feeds.spec.yaml
   --output-dir <CORPUS_DIR> --skip-existing --reprocess-source whisper_transcription
   --reprocess-existing-only`
   — `--reprocess-existing-only` scopes to the corpus's on-disk GUIDs;
   `--reprocess-source whisper_transcription` force-reprocesses **only** those whose
   `transcript_source` is `whisper_transcription` (direct-download episodes untouched,
   #925). Each reprocessed episode re-runs the **full per-episode cascade**:
   transcribe → diarize → screenplay → GI → KG → bridge → search index.
2. `cli enrich-edges --output-dir <CORPUS_DIR>` — re-derives the corpus-wide
   `SPOKEN_BY` relational edges (#876/#909).

`PROFILE = config/profiles/cloud_with_dgx_primary.yaml` (DGX Whisper primary,
`diarization_provider: tailnet_dgx`, openai transcription fallback; in-process
pyannote fallback on DGX health failure).

> ⚠️ **Clear `.cache/transcripts` before re-diarizing** (`rm -rf .cache/transcripts`). The
> transcript cache is keyed by audio hash and stores the **already-formatted screenplay**
> (post-diarization). A warm cache short-circuits transcribe→diarize→format, so the run
> reuses the *old* diarization/naming and re-diarization becomes a silent no-op. Verified
> 2026-06-12: with the cache warm the run logged `Transcript cache hit … transcribe_sec=0.0`
> and the screenplay was unchanged. (`transcript_cache_enabled: false` in the profile does
> *not* reliably take effect via the CLI config merge — clear the dir instead.)
>
> ⚠️ **Keep the laptop awake for the whole run** (`caffeinate -i …`, mains power). If this
> machine sleeps mid-run the tailnet drops and every DGX diarize POST fails with
> `Connection reset by peer` → slow in-process fallback. Verified 2026-06-12: a sleep during
> an unattended run produced exactly this; the busy-vs-down health check (#956) correctly
> reported the box `busy`/`ready`, so the failure was local, not the DGX box.

## Prerequisites (status as of 2026-06-09)

| Dependency | Status |
| --- | --- |
| DGX Whisper (Speaches, `:8000`, `Systran/faster-whisper-large-v3`) | ✅ healthy |
| DGX pyannote (`:8001`, `pyannote/speaker-diarization-community-1`) | ✅ healthy |
| Tailnet reachability to `your-dgx.tailnet.ts.net` | ✅ active/direct |
| `cloud_with_dgx_primary` profile on `main` | ✅ (#941) |
| `--reprocess-source` / `--reprocess-existing-only` / `make migrate-diarization` tooling | ✅ (#944 merged) |

## Step 1 — Health gate (abort if either service is down)

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "src")
from podcast_scraper.providers.tailnet_dgx import health
HOST = "your-dgx.tailnet.ts.net"
w = health.check_faster_whisper_health(HOST, port=8000,
        require_model_substring="faster-whisper-large-v3")
d = health.check_pyannote_diarize_health(HOST, port=8001)
assert w, "DGX Whisper (:8000) not healthy — abort"
assert d, "DGX pyannote (:8001) not healthy — abort"
print("DGX health OK: Whisper large-v3 + pyannote 3.1")
PY
```

## Step 2 — Backup the corpus (reversibility)

The reprocess **overwrites** transcripts, diarization, screenplays, GI/KG/bridge,
and the search index for every Whisper episode. Snapshot first:

```bash
CORPUS_DIR=/path/to/corpus
tar -czf "$HOME/corpus_backup_$(date +%Y%m%d-%H%M%S).tar.gz" -C "$(dirname "$CORPUS_DIR")" "$(basename "$CORPUS_DIR")"
```

Record the current transcript-source split so you know how many episodes the
reprocess will touch:

```bash
grep -rl "whisper_transcription" "$CORPUS_DIR"/*/metadata/*.json 2>/dev/null | wc -l
```

## Step 3 — Pilot (2–3 representative episodes, on a COPY)

Validate quality before committing 100 episodes. Run on a copy so the live
corpus is untouched until the pilot passes.

> ⚠️ **`--max-episodes` is IGNORED under `--reprocess-existing-only`** (verified
> 2026-06-11: it processes *all* on-disk GUIDs regardless). To pilot a subset you must
> **trim the corpus copy** to the episodes you want, then re-diar existing-only — the
> scan picks up exactly the GUIDs that remain on disk.

```bash
PILOT_DIR=/tmp/rediar_pilot
rm -rf "$PILOT_DIR" && cp -R "$CORPUS_DIR" "$PILOT_DIR"

# Keep only the first 2–3 episodes' metadata so existing-only scopes to them.
# (metadata/*.metadata.json is what the GUID scan reads; trim transcripts to match.)
MD=$(echo "$PILOT_DIR"/feeds/*/run_*/metadata); TR=$(echo "$PILOT_DIR"/feeds/*/run_*/transcripts)
for d in "$MD" "$TR"; do for f in "$d"/*; do
  case "$(basename "$f")" in 0001*|0002*) : ;; *) rm -f "$f";; esac
done; done

# NOTE: --reprocess-existing-only is REQUIRED — without it the run scrapes the live
# feed instead of the corpus (see Mechanism). --max-episodes would be ignored here.
.venv/bin/python -m podcast_scraper.cli \
  --config config/profiles/cloud_with_dgx_primary.yaml \
  --feeds-spec "$PILOT_DIR/feeds.spec.yaml" \
  --output-dir "$PILOT_DIR" \
  --skip-existing --reprocess-source whisper_transcription \
  --reprocess-existing-only
.venv/bin/python -m podcast_scraper.cli enrich-edges --output-dir "$PILOT_DIR"
# Confirm the scoping log line: "Existing-only re-diarization (#876): kept 2, dropped …"
```

> 💡 If the audio cache (`.cache/audio`, keyed by `sha256(guid)`) was cleared, the pilot
> re-downloads from the feed. To diarize off-cache, pre-seed the cache from any stored
> raw episode audio: `cp <ep>.mp3 .cache/audio/sha256/<h[:2]>/<h[2:4]>/<h>.mp3` where
> `h = sha256(guid)`. The run then logs `audio cache HIT … (no feed fetch)`.

Pilot acceptance (pick a 2-speaker, a panel, and a single-host episode if possible):

- [ ] Screenplay has ≥2 distinct **named** `Name:` markers for multi-speaker episodes.
- [ ] `diarization_num_speakers` in the enriched result matches the known cast.
- [ ] GI `Quote` nodes carry `timestamp_start_ms`; KG has the speaker `Entity` nodes.
- [ ] `SPOKEN_BY` edges present (Quote → Person) after `enrich-edges`.
- [ ] No "char_start not aligned" warnings (Bug3 guard) — see Step 4.

## Step 4 — Cascade / offset verification (Bug3)

> **Post-#974:** GI now indexes the saved **ad-free base** (`*.adfree.txt`, produced at
> transcript-save time) whose segments carry exact `char_start`/`char_end`. Re-diarization
> regenerates that base, so quote offsets re-derive **exactly** to the new transcript and the
> screenplay-marker drift (#545) can no longer occur. `verify-gil-offsets-strict` still confirms
> alignment, but the manual offset knob (`cil_lift_overrides.transcript_char_shift`) is obsolete
> for ad-free corpora.

Re-diarization shifts transcript char offsets; GI quote `char_start` must index
the **new** transcript. The reprocess rebuilds GI per episode so they stay
aligned, but verify explicitly:

```bash
make verify-gil-offsets-strict CORPUS_DIR="$PILOT_DIR"
# Also scan logs for the offset-guard warning (should be absent):
#   "add_spoken_by_edges: N quote(s) have char_start not aligned with the transcript"
```

If offsets mismatch: GI was not rebuilt against the new transcript — do **not**
proceed; investigate before the full run.

## Step 5 — Full run (all 100, after pilot passes + backup exists)

```bash
make migrate-diarization \
  CORPUS_DIR="$CORPUS_DIR" \
  PROFILE=config/profiles/cloud_with_dgx_primary.yaml
```

Monitor: tail the run log; **confirm the scoping line** `Existing-only re-diarization
(#876): kept N, dropped … new feed item(s)` (if you instead see episodes being
downloaded by title from the feed, you ran `redo-diarization` by mistake — stop and
use `migrate-diarization`). Re-run the **Step 1 health gate** periodically (the provider
falls back to in-process pyannote on DGX failure — watch for fallback breadcrumbs so you
know whether episodes actually used DGX). Expect this to take a while (large-v3
transcription ≈ 6–7 min/episode + pyannote per episode, ~100 episodes).

## Step 6 — Post-run validation

- [ ] `make verify-gil-offsets-strict CORPUS_DIR="$CORPUS_DIR"` clean.
- [ ] Spot-check ~5 random episodes: named screenplay + `SPOKEN_BY` + KG entities.
- [ ] Corpus episode count unchanged; no episodes dropped/errored in the run summary.
- [ ] Rebuild/refresh the search index is included in the per-episode cascade;
      confirm a vector search returns sensible results.

## Rollback

```bash
rm -rf "$CORPUS_DIR"
tar -xzf "$HOME/corpus_backup_<stamp>.tar.gz" -C "$(dirname "$CORPUS_DIR")"
```

## Notes

- This is a **data migration**, not a code change — run it from `main` after #944
  merges, as a tracked #876 operation (not bundled into a PR).
- Track quality-vs-baseline if the corpus feeds eval (#903 v2 eval-track).
- Natural-audio fixture work (Gemini TTS) is unrelated and tracked separately in #934.
