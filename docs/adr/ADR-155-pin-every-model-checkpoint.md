# ADR-155: Pin every model checkpoint; prefer safetensors where the canonical repo has it

- **Status**: Accepted
- **Date**: 2026-09-24
- **Authors**: Marko Dragoljevic
- **Issues**: [#2144](https://github.com/chipi/podcast_scraper/issues/2144)
- **Supersedes**: —
- **See Also**: [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md),
  [ADR-068](ADR-068-bart-led-as-ml-production-baseline.md),
  [ADR-154](ADR-154-hybrid-map-reduce-retirement.md),
  [ADR-106](ADR-106-transformers-v5-ml-backend-unification.md)

## Context & Problem Statement

`make ci` stopped completing on x86_64 macOS with what looked like a platform complaint:

```text
Failed to preload allenai/led-base-16384: Due to a serious vulnerability issue in
`torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at
least v2.6 ... This version restriction does not apply when loading files with safetensors.
```

`transformers >= 4.56` refuses `torch.load` below torch 2.6, citing **PYSEC-2025-41** — an RCE
that `weights_only=True` does not close. The message names its own way out: safetensors.
`allenai/led-base-16384` has none. It ships `pytorch_model.bin` and nothing else, so loading it
*requires* unpickling.

**transformers is right to refuse.** It is declining to unpickle an archive on a torch with a
known `torch.load` RCE. The first instinct — pin an older torch so the guard stops firing — was
measured and is strictly worse: `pip-audit` against the last torch with an x86_64 macOS wheel
reports `30 known vulnerabilities` across `torch 2.2.2` (22) and `transformers 4.57.6` (8), and
`PYSEC-2025-41` is among them. Downgrading to load the pickle means accepting the exact RCE the
guard exists to prevent.

The platform symptom is incidental. The real finding is what the model set looks like when you
cross two properties that were never looked at together:

| model | safetensors | pinned revision |
| ----- | :---------: | :-------------: |
| `google/flan-t5-base` | yes | yes |
| `sentence-transformers/all-MiniLM-L6-v2` | yes | yes |
| `facebook/bart-base` | yes | no |
| `facebook/bart-large-cnn` | yes | no |
| `google/long-t5-tglobal-base` | no | yes |
| `google/long-t5-tglobal-large` | no | yes |
| `google/pegasus-large` | **no** | **no** |
| `google/pegasus-cnn_dailymail` | **no** | **no** |
| `google/pegasus-xsum` | **no** | **no** |
| `allenai/led-base-16384` | **no** | **no** |
| `allenai/led-large-16384` | **no** | **no** |
| `sshleifer/distilbart-cnn-12-6` | **no** | **no** |

**Six checkpoints are pickle-only AND unpinned**, and only four of twelve are pinned at all.
The pickle format is the part that fires the guard; the missing pin is the part that makes it
dangerous. A pinned pickle is a known quantity — the bytes cannot change under you — and
`get_pinned_revision_for_model` already exists precisely because of that, with the reasoning
recorded at its definition (*"the PINNED FLAN-T5 matched its frozen baseline exactly while the
unpinned models were where expectations wobbled"*). An **unpinned** pickle is a moving remote
archive that `torch.load` executes at load time, fetched in CI and baked into the pipeline image.

Eight of the twelve float on `main`, including four that ship safetensors and were never at risk
from the guard. So this is not really a safetensors problem that happens to involve pinning — it
is a pinning problem that the safetensors guard made visible.

`allenai/led-base-16384` is in that set and is also `TEST_DEFAULT_SUMMARY_REDUCE_MODEL`
(`config_constants.py:266`) and the `summary_reduce_model` of `airgapped_thin.yaml:78`.

## Decision

**Every checkpoint is pinned to a 40-hex SHA revision in `get_pinned_revision_for_model`. No
exceptions.** Pinning is cheap, it is already the mechanism, and it is the only property that
holds regardless of file format — a pinned pickle cannot change under you, and a pinned
safetensors file cannot drift either. Today 4 of 12 are pinned; the other 8 float on `main`.

**Where the canonical repository offers safetensors, the loader prefers it.** That is what lifts
the `torch.load` guard and removes the unpickling step entirely. It is a second, independent
requirement — not an alternative to pinning.

**A community re-upload is not an acceptable way to satisfy the safetensors requirement.**
Checked on 2026-09-24: the `distilbart-cnn-12-6` copies on the Hub are themselves pickle-only,
and `allenai/led-base-16384-ms2` — the one nearby repo that does ship safetensors — is a
fine-tune on MS2, i.e. different weights. Swapping a canonical checkpoint for an individual's
re-upload trades a format problem for a provenance problem, which is the worse of the two.

## Consequences

- **Eight models get pinned.** `bart-base`, `bart-large-cnn`, `pegasus-large`,
  `pegasus-cnn_dailymail`, `pegasus-xsum`, `led-base-16384`, `led-large-16384`,
  `distilbart-cnn-12-6`. The first two already pass the safetensors requirement and are pinned
  for drift, not for safety; the other six are the pickle-only set and pinning is what makes them
  admissible at all.
- **The pickle-only six stay pickle-only.** No safetensors version of those *weights* exists.
  Converting and hosting them ourselves is the clean end state and is out of scope here; until
  then the pin is what bounds the risk.
- **Retired models should be dropped rather than pinned.** `pegasus-*` and `distilbart` are
  reachable from no profile, and ADR-067 already retired Pegasus/LED for podcast content — the
  held-out v2 eval measured the BART+LED baseline at judge-mean **0.23**
  (`podcast-scraper-eval-data:docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md` §6, *"no rubric
  change recovers that"*). Deleting an entry removes the pickle instead of managing it, so the
  audit is per model: drop what nothing reaches, pin what remains.
- **The test profile stops borrowing the airgapped constraint.** `test_default.yaml:41` and
  `airgapped_thin.yaml:78` both set `summary_reduce_model: long-fast`, but for unrelated reasons:
  airgapped genuinely needs a long-context LOCAL reduce, while tests need something small and
  fast that loads. Tests inheriting a 16k-context pickle is an accident of both wanting "the
  small one". `test_default` moves to a safetensors checkpoint; `airgapped_thin` keeps LED, pinned.
- **The invariant needs a test.** A rule nothing enforces is a comment. A test that walks the
  model set and fails on any unpinned entry — and separately flags any entry whose canonical repo
  offers safetensors that we are not using — is the same shape as the fixture-content gates, and
  is the only thing that stops the next added model from reintroducing this.
- **x86_64 macOS still cannot run the local-transformers stages.** This decision does not fix
  that and is not trying to: torch is capped at 2.2.2 there, the guard is correct to fire, and
  the answer for that host is the torch-free embedding path (#2142), not an older torch.

## Alternatives considered

- **Pin `torch < 2.6` so the guard stops firing.** Rejected on measurement: 30 known
  vulnerabilities, `PYSEC-2025-41` included. It trades a load-time failure for the
  vulnerability that failure is protecting against.
- **Set `use_safetensors=False` / suppress the guard.** Same objection, moved into our code
  where it is less visible.
- **Ban pickles outright.** Rejected: it would remove `airgapped_thin`'s only long-context
  REDUCE option, and a pinned pickle is not the risk being managed. The risk is a pickle that
  can change.
- **Convert the checkpoints to safetensors and host them.** The cleanest end state, and out of
  scope here — it means owning a conversion and a hosting location. Clause 2 is the bridge.
