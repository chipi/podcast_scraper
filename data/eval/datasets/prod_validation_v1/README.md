# prod_validation_v1 — frozen real-prod validation tier (#933)

## Why

`docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` documents that every closed
autoresearch child (#853 / #594 / #904 / #905 / #906 / #816) reached for
ad-hoc prod backup data to validate findings because synthetic v2 smoke
can't represent the failure modes that actually appear in production.

Each ticket picked its own ad-hoc 3-5 episode prod subset. **There was no
shared frozen ground truth.** This dataset is that ground truth — a small
(15 episode), frozen subset hand-curated to span the failure-mode taxonomy.

## Source

Local prod backup at
`.test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/`
(pulled 2026-04-21, 54 episodes from 10 RSS feeds). The 15 picks under
`episodes/` are symlinks into that backup — the dataset is portable as
long as the backup root stays in place.

## Taxonomy coverage (v1)

| Tag | Episodes |
| --- | --- |
| `native_ad_heavy` | 12 |
| `cross_feed_topic_cluster` | 11 |
| `long_interview` | 3 |
| `sponsor_shaped_real_content` | 2 |
| `asr_garble` | 1 |
| `low_grounding` | 0 — needs more diverse backup |
| `ner_zero_hosts` | 0 — needs NPR-shape episodes |
| `multi_accent` | 0 — needs audio probing |
| `sustained_burst` | 0 — needs ≥3h continuous telemetry |
| `dialogue_insight_offender` | 0 — needs GI runtime output |
| `nickname_alias` | 0 — needs KG canon pair output |

6 of 11 tags are uncovered because the source backup is FT-Unhedged-
dominant (one feed accounts for ~50 of the 54 episodes). Two paths to
covering them, both deferred to v2:

1. Pull a more diverse backup spanning NPR, omnycontent, and other feeds
   with the failure modes we want to test.
2. Probe the uncovered tags at runtime per episode and write the results
   to a sidecar file — the frozen `manifest.yaml` stays unchanged; only
   the sidecar grows.

## Freeze guarantee

Per the #933 design: **v1 does not churn after commit.** If a bug is
discovered in episode metadata, it goes in a sidecar errata file. If
new failure modes worth adding emerge, we open `prod_validation_v2/` —
never overwrite v1. Reproducibility across consumers (issues 921, 927-931,
932, 923) depends on this invariant.

## Used by

- `scripts/eval/validate_prod_set.py` — runs a configurable pipeline
  step against the 15-episode subset and emits a summary report
- **#921 v3 fixtures rebuild**: "does autoresearch on v3 reproduce
  findings here?" — fidelity check
- **#932 finale tier**: top-2 sanity check after G-Eval picks finale
  winner; if the winner flips on this set, that's a finding we'd have
  shipped wrong
- **#923 prod_dgx_full_with_fallback**: final reality check before any
  prod LLM / Whisper / diarize swap

## Layout

```text
data/eval/datasets/prod_validation_v1/
├── README.md             (this file)
├── manifest.yaml         (the frozen tag + source spec)
└── episodes/
    ├── ep_0001.txt -> ../../.../transcripts/0001 - Boing...txt
    ├── ep_0001.segments.json -> ../../.../transcripts/0001 - ....segments.json
    ├── ep_0013.txt -> ...
    └── ... (15 episodes total, each with its segments.json sibling)
```

Stable episode IDs (`ep_0001` … `ep_0056`) decouple downstream consumers
from filename churn in the source backup. The backup's original filenames
encode a date prefix; the symlink names are stable.

## Reproduction

The dataset is curated; to recreate the symlinks against the same source
backup:

```bash
.venv/bin/python <<'PY'
import os
from pathlib import Path
src = Path('.test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts')
dst = Path('data/eval/datasets/prod_validation_v1/episodes')
dst.mkdir(parents=True, exist_ok=True)
selected = ['0001','0013','0014','0021','0024','0029','0033','0034','0036',
            '0038','0040','0046','0049','0050','0056']
for sel in selected:
    txt = next(src.glob(f'{sel} - *.txt'))
    seg = txt.with_name(txt.stem + '.segments.json')
    for s, name in [(txt, f'ep_{sel}.txt'), (seg, f'ep_{sel}.segments.json')]:
        target = dst / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(os.path.relpath(s, dst))
PY
```
