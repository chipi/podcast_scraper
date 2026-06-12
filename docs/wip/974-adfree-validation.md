# #974 ad-free transcript model — validation report

Two-artifact transcript model (raw canonical + ad-free processing base) fixing the
#545 offset drift at its source. See GH issue #974.

## Empirical result on the real 10-ep corpus

Measured on `.test_outputs/manual/rediar-batch-iltb` (run_20260612-130837), the real
Invest-Like-the-Best re-diarized corpus (Brian Chesky, Paul Tudor Jones, …), Gemini GI.

For each existing `gi.json` quote: re-locate its verbatim text in the ad-free base and
run the new exact segment→speaker mapping.

| Metric | Current gi.json | Ad-free base |
|---|---|---|
| Quotes with `speaker_id` | 54% (100/185) | **100% (185/185)** |
| Episodes fully unattributed | 4/10 (0002,0006,0007,0010) | **0/10** |
| Quotes findable in transcript | — | 100% |
| Per-episode ad chars removed | — | 1,773–3,548 |
| Old `char_start` drift vs real position | ~1 KB | exact (slice-invariant holds) |

The four episodes that previously dropped ALL attribution (34/8/36/7 quotes) attribute
100% through the ad-free base. Confirms:
- **Fault A** — old `char_start` indexed the unsaved ad-excised space (R4); ~1KB off the
  raw `.txt` everyone read. Ad-free base is saved → offsets exact.
- **Fault B** — screenplay `Name:` markers tripped the cumulative-length guard, dropping
  segment speaker mapping. Explicit per-segment offsets map exactly, no guard.

## Test pyramid (all four layers)

- **unit**: formatter offsets, ad-excision-with-offsets, producer/resolver,
  `_segment_char_spans`/mapping, `_maybe_produce_adfree` gate, `_transcript_path`
  ad-free preference, viewer sidecar derivation.
- **integration**: `tests/integration/gi/test_adfree_chain.py` — producer → resolver →
  GI grounded build → enrich-edges; Faults A/B + SPOKEN_BY.
- **e2e**: `tests/e2e/test_diarization_e2e.py` — real whisper+pyannote screenplay drives
  the ad-free base; sidecars written, GI char_start exact, speaker_id from diarization.
- **Playwright**: `transcript-viewer-dialog.spec.ts` — a `.adfree.txt`-referenced quote
  loads the ad-free base + `.adfree.segments.json` sidecar and highlights exactly.

## Validation chain on the reprojected corpus (production tools)

`.test_outputs/manual/rediar-974-reprojected` — corpus copy, reprojected via
`scripts/migrate/reproject_gi_to_adfree.py` (backfill ad-free + re-locate each quote in
the ad-free base, recompute char_start/speaker_id/timestamps, transcript_ref→adfree),
then production `enrich-edges` + `index`.

| Validator | Result |
|---|---|
| `diarization-quality` | ✓ all thresholds pass |
| quote_attribution_rate | 0.54 → **1.0** |
| episodes_unattributed | 4 → **0** |
| spoken_by_coverage | 0.54 → **1.05** |
| episodes_with_network_speaker | **0** |
| `verify-gil-chunk-offsets --strict` | **verdict: aligned, overlap_rate 1.0, 185/185 quotes overlap, exit 0** |

The full chain (GI char_start ∈ ad-free → FAISS chunks ad-free → 100% overlap) is green.
`episodes_missing_num_speakers: 10` is a pre-existing, opt-in (non-enforced) metadata
propagation gap, not a #974 regression.

## Remaining for full sign-off

- Produce a corrected serveable corpus for Tier-3 graph walk (reproject existing quotes
  onto ad-free offsets + enrich-edges + reindex, OR a faithful GI re-run via Gemini).
- Tier-3: serve corrected corpus, walk viewer tabs, confirm highlight alignment +
  SPOKEN_BY in the graph.
- Run `verify-gil-offsets --strict` + `make diarization-quality` on the corrected corpus.
