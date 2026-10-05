# Pipeline check — runbook

`make pipeline-check` runs the pipeline under a set of **variants** (a locale today; a profile or
any config override through the same mechanism), records **every decision** the pipeline makes,
and compares runs. It is the standard check before merging a change to the pipeline's
language-, naming-, ad-, transcript- or search-routing code (#2287).

It answers two questions with one machinery:

| Question | How | Example |
| --- | --- | --- |
| Did this change alter the pipeline's behaviour? | **code vs code** — the same variants on `BASE` and on the candidate | `BASE=main` |
| How does the pipeline treat these variants differently? | **variant vs variant** — several variants on the same code | `LOCALES="pt-BR pt-PT"` |

## Quick start

```bash
make pipeline-check                    # this checkout, every check in expectations.yaml
git fetch origin                        # BASE resolves through git: use origin/main, not a stale local main
make pipeline-check BASE=origin/main   # this checkout vs main, every check
make pipeline-check CHECK=english BASE=origin/main
make pipeline-check CHECK=english LOCALES="en en-US en_GB"   # ad hoc variant list
make pipeline-check CANDIDATE=feat/x BASE=origin/main CORPUS=$PWD/tests/fixtures/app-validation-corpus/v3
```

The last line prints `PIPELINE_CHECK_EXIT=0` (every check passed) or `1`. The one-page report
is `.test_outputs/pipeline-check/report.md`; the raw data (every decision, per side) sits next to
it as `<check>.candidate.json` / `<check>.base.json`, with `findings.json`.

Fixture mode needs no DGX and no network, and takes about a minute per check.

## Parameters

All optional.

| Variable | Meaning | Default |
| --- | --- | --- |
| `CHECK` | which checks from `expectations.yaml` (space-separated) | all |
| `BASE` | git ref to compare against (use `origin/main` after a fetch — a local `main` may be stale; the report prints the SHA each ref resolved to); checked out into its own worktree under `.test_outputs/pipeline-check/worktrees/` | none: single-ref check |
| `CANDIDATE` | git ref to check | this working tree |
| `LOCALES` | replace a check's locale list; `-` means "the feed declares nothing", `override:en` means "declares nothing, an operator override says `en`" | the check's own |
| `FEEDS` | replace a check's fixture feeds (`p01,p02`) | the check's own |
| `PROFILE`, `OVERRIDES` | further variant dimensions (`OVERRIDES='{"key": value}'`) | none |
| `CORPUS` | the fixture corpus both sides run on | `tests/fixtures/app-validation-corpus/v3` of this checkout |
| `REAL`, `FEED`, `MAX_EPISODES` | real-episode mode (below) | off |

**Both sides always read the same input.** The corpus comes from one place (`CORPUS`, else the
checkout the tool runs from), never from each ref's own tree. When the candidate adds fixture
episodes the base does not have (as a multilingual branch does), pass `CORPUS` pointing at the
candidate's corpus.

## What it does

1. **Recorder** (`scripts/validate/pipeline_check/recorder.py`) runs inside the pipeline's
   process and *discovers* the decision points — nothing is listed by hand, so a map added next
   month is traced without editing the tool:
    - every module-level dict keyed by language codes is replaced by an identical dict that logs
      each lookup (the key asked for, and whether the row was there);
    - every function with a `language` / `lang` / `feed_language` / `source_language` / `tag`
      parameter is wrapped to log what arrived and what it returned.
2. **Drivers** (`drivers.py`) run the deterministic stages over the fixture episodes, once per
   variant, calling the real pipeline functions of the checkout under test: language resolution,
   the refusal gate, host detection, speaker naming, ad removal, transcript selection, the
   translation/analysis gate, search routing. LLM calls are not made: what is checked is what the
   pipeline **chooses**, not what a model answers.
3. **Comparator** (`compare.py`) applies one check from `expectations.yaml`.
4. **Report** (`report.py`) writes one page.

A base ref without some stage (`main` before the multilingual work has no language gate) records
it as absent; the comparator never treats "absent" as "identical".

## Reading the report

The verdict line comes first. Then one section per check:

| Column | Meaning |
| --- | --- |
| Decisions | keyed choices the stage made, across every episode and variant |
| Holes | choices that did not resolve as the check requires |
| Outputs | the stage's records against the base (`identical to base (N)`) or against pinned values (`pinned values hold (N)`) |

Below the table, **only the problems** are listed, each with the file and line of the pipeline
code that made the choice. Passing items are counted, not listed.

**Coverage** says how many of the discovered decision points the run exercised. The ones never
hit are named in `findings.json`; a hole cannot hide in a point that was never exercised, so a
change to such a point needs its own test. **Import-time copies** are values copied out of a map
when the module loaded (`_X = _X_BY_LANGUAGE["en"]`) — chosen before the recorder existed, so
checked statically by `tests/unit/podcast_scraper/test_english_patterns_are_mains.py`, not traced.

### What a hole looks like

Run against the commit before the `en_US` host fix (c7a2bcc50), the `english` check reports:

```text
Host detection — `…hosts._HOST_PHRASES_BY_LANGUAGE (called from …/hosts.py:851)`: looked up 'en_gb' under 'en_GB'
```

The host-statement map was asked for `en_gb`, a key it does not hold, so an English feed tagged
`en_GB` got no host patterns. The *output* comparison for that stage still said "identical to
base" — the fixture feeds happen not to need those patterns — which is why decisions are traced
rather than only outputs compared.

## Expectations (`scripts/validate/pipeline_check/expectations.yaml`)

One named check per question:

| Field | Meaning |
| --- | --- |
| `feeds`, `locales` | the input episodes and the variants to run |
| `decisions_resolve_to` | every keyed lookup asks for exactly this key and finds it; every language argument normalises to it (or is unset) |
| `variants_identical` (+ `variants_identical_ignore`) | every variant must produce the same stage records and the same decisions |
| `base_identical_stages` | these stages must equal `BASE` exactly, episode by episode |
| `expect_values` | pinned `<stage>.<field>` values, for stages the base may not have |
| `report_only` | nothing is expected; the differences between variants are the answer |

**An intended behaviour change is an edit to this file**, and the edit is what review sees. Do
not edit it to make a red check green without saying why in the commit.

Shipped checks:

- `english` — 40 English fixture episodes under `en`, `en-US`, `en_GB`, `English`, `eng` and
  `override:en`: every choice resolves to `en`, all six behave identically, and hosts / naming /
  ad removal / transcript selection equal the base.
- `no-language` — a feed that declares nothing is refused before download (#2283).
- `portuguese-variants` — `pt`, `pt-BR`, `pt-PT` on the Portuguese fixture feed, report only.

## Real-episode mode (DGX)

Fixture mode proves the choices. Real mode runs the **full pipeline** on real episodes — ASR,
diarization, naming, summaries, insights, KG — so the same decisions are traced on a real run and
the LLM's variation can be measured:

```bash
export VLLM_API_KEY=EMPTY   # the DGX vLLM endpoints' key (infra/vllm/*/docker-compose.yml)
make pipeline-check REAL=1 BASE=origin/main FEED=<rss url> PROFILE=prod_dgx_full MAX_EPISODES=1
```

- The base runs **twice**, the candidate once. The two base runs define the noise band.
- All runs share one transcript cache, so Whisper runs once per episode and every side reads the
  same transcript: ASR non-determinism is removed, not measured.
- `vector_search` is off in real mode (local embeddings need an ML stack not every machine has).
- **Deterministic artifacts** (file set, `metadata.json` minus the LLM's own fields, the processing
  manifest) must equal the base apart from `allowed_artifact_differences`.
- **LLM output** — insights, grounded share, quotes, summary bullets, and word/name overlap of the
  summary and KG — is compared against the band: `max(base-vs-base spread, 10% of the value,
  floor)`; overlaps must be within 0.1 of the base's own overlap. Outside the band is flagged
  `LOOK AT THIS`; it does not fail the verdict by itself.

**Only run real mode when the DGX is quiet** — it shares the GPU with production, and a busy GPU
makes a 30-minute episode take an hour in the Whisper queue. Check the GPU first:

```bash
curl -s http://homelab:8428/api/v1/query --data-urlencode \
  'query=avg_over_time(DCGM_FI_DEV_GPU_UTIL[10m])' | grep -o '"value":\[[^]]*\]'
```

Cost per run: ASR once per episode (shared), plus three full LLM passes (two base, one candidate).

## Limits

- **LLM output quality** is not judged; only its variation is measured, and only in real mode.
- **Search quality on an existing corpus** is not covered (needs the ML stack or a prod snapshot).
- **Decision points the driven stages never reach** are counted and named, not checked.
- **Two spellings of one language are compared on the same audio.** `pt-BR` vs `pt-PT` on
  fixtures shows whether the pipeline treats the tags differently; comparing two real accents
  needs real episodes of each (real mode).

## Extending it

- **A new stage:** add a driver to `drivers.py` (`STAGES` and `run_stages`) that calls the real
  pipeline function and returns a JSON-able record; tolerate its absence on older refs.
- **A new dimension** (profile, override): pass it through `PROFILE` / `OVERRIDES` today; for a new
  kind of keyed decision, add its parameter names to `recorder.LOCALE_PARAMS`' sibling list and a
  discovery rule in `recorder._is_keyed_map`.
- **A new question:** add a check to `expectations.yaml`.

Tests: `tests/unit/scripts/validate/test_pipeline_check.py` and
`test_pipeline_check_artifacts.py` build each failure the tool must catch and assert it does.
