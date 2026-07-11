# Fixture transcripts — v3 (CURRENT — use these)

`v3` is the current fixture set (`tests/fixtures/FIXTURES_VERSION` = `v3`). **v1 and v2 are
deprecated — do not use them** (see their READMEs). New fixture work and all diarization / speaker
eval runs use `v3`.

## Ground-truth sidecars

Each `<name>.txt` has a co-located `<name>.groundtruth.json` declaring **exactly what is inside
the episode** — the source of truth for the speaker / diarization eval (so nothing is parsed or
guessed at eval time):

| field | meaning |
| ----- | ------- |
| `speakers` | distinct **human** speaker labels (excludes the synthetic `Ad` voice) |
| `num_human_speakers` | `len(speakers)` |
| `has_commercial` / `num_ad_voices` | whether a mid-roll `Ad:` (sponsor) voice is present |
| `expected_diarized_voices` | humans + ad voices — what a correct diarizer should **detect** |
| `type` | `monologue` (1) · `interview` (2) · `panel` (≥3) |
| `failure_modes` | from the transcript's `#fixture-v3: failure_modes=…` annotation |

`expected_diarized_voices` is the diarization tuning target. `type` + `has_commercial` +
`failure_modes` let an eval slice results — e.g. keep the **panel** (`p05_e04`) and the
**commercial** cases (30 fixtures with an ad voice) honest so tuning can't collapse to a
degenerate "always 2 speakers".

## Regenerating

The sidecars are derived from the transcripts — never hand-edit; edit the transcript then:

```bash
python tests/fixtures/scripts/make_groundtruth.py          # regenerate all v3 sidecars
python tests/fixtures/scripts/make_groundtruth.py --check  # CI: verify they're up to date
```

Audio is generated from the transcripts via `tests/fixtures/scripts/transcripts_to_mp3.py`.
