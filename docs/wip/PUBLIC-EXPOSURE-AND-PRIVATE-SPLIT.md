# Public exposure: what leaks, what does not, and the private-split option

**Status:** analysis only — no action taken, no repo created, nothing moved.
**Written:** 2026-07-14, while the gemini-vs-qwen 18-episode head-to-head was running.
**Trigger:** operator: *"Nothing that touches prod content goes to GitHub in any way, since it is a
public project and I don't want to share how I fine-tune things."*

`chipi/podcast_scraper` is a **public** repo. This note measures what that actually exposes today,
separates the two very different boundaries that got conflated in the question, and records the
private-split options for when the operator wants to revisit.

---

## The two boundaries are not the same

| | What it is | Status today |
| --- | --- | --- |
| **Prod content** | The transcripts themselves, the diarized segments, the GI/KG artifacts built from them | **Never public. Boundary holds.** |
| **Method** | *How* the corpus gets tuned: eval configs, per-model params, judge design, scorecards, failure taxonomies | **Fully public.** Operator has accepted this **for now** (2026-07-14) and will revisit. |

The operator's concern was prod content. That one is safe. The methodology surface is much larger
than it looks, so it is measured below — the decision to accept it should be an informed one.

---

## Prod content — verified safe

Four paths could carry it. All four are closed:

| Path | Carries | Why it cannot be committed |
| --- | --- | --- |
| `data/eval/materialized/*` | full transcripts + `.segments.json` | `.gitignore:189` |
| `data/eval/runs/*` | `predictions.jsonl` — GI artifacts with **verbatim quote text** | `.gitignore:196` |
| `.test_outputs/*` | the prod-v2 / prod-v3 corpora, audio, ad-hoc eval configs | gitignored |
| `data/eval/datasets/prod_v3_*.json` | **tracked** — but manifest only | 8 KB / 18 episodes: `episode_id`, `title`, `show`, `chars`, `transcript_hash`, `transcript_path`. No transcript text. |

`data/eval/datasets/prod_validation_v1/episodes/ep_*.txt` **looks** like 30 committed prod episodes.
It is not: they are **symlinks** into the gitignored `.test_outputs/` (30 files, 4,872 bytes total —
each blob is just a relative path). Anyone cloning the public repo gets dangling links, not content.
This is a good pattern and worth keeping.

What *is* public: episode **titles and show names**, in the manifests and symlink names. These are
published RSS metadata, so the content is not secret — but the *selection* does reveal which
episodes are used for evaluation. Judged acceptable; noted because it is the one residual.

### The gap: the boundary is enforced by `.gitignore` alone

There is **no check that would stop a prod transcript from being staged**. `.github/hooks/pre-commit`
runs black / isort / mypy and nothing else; `scripts/check/` holds only `no-secrets-in-image.sh`.

This session came within one directory of proving the point. `materialize_dataset.py` was taught to
copy `.segments.json` (full transcript text) alongside the materialized transcripts. It landed in an
**ignored** directory — by location, not by design. A tool that writes prod content one level to the
left commits it silently.

### A guard is viable (measured, not built)

Detect by **shape**: a screenplay transcript is a file with many `^Speaker Name: text` lines. Against
the current tracked tree that fires on **203 files** — every one of them a synthetic fixture
(`tests/fixtures/transcripts/**`, `data/eval/sources/curated_5feeds_*/**`). So shape alone is
unusable, but shape **plus an allowlist of the known-synthetic roots** has zero false positives today
and would refuse any *new* screenplay file appearing outside them.

Proposed (NOT implemented — needs operator go-ahead):

- refuse staged `*.segments.json`, `*predictions.jsonl`
- refuse any staged file with >= 20 screenplay turns outside `tests/fixtures/transcripts/**` and
  `data/eval/sources/**`
- wire into `.github/hooks/pre-commit` next to the existing checks

---

## Method — public today, accepted for now

This is the part the operator flagged as *"how I fine-tune things"*. All of it is on public `main`:

| Surface | Size | What it gives away |
| --- | --- | --- |
| `autoresearch/MODEL_PLAYBOOK.md` | 1,279 lines | per-model behaviour, what worked and what did not |
| `autoresearch/JUDGING.md` | 651 lines | judge design, the cross-vendor bias rule (#939) |
| `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` | 245 lines | the canonical per-model vLLM flag table for GB10 |
| `data/eval/configs/**` | 456 files | model params, prompts, sampling, judge choices |
| `docs/guides/eval-reports/**` | 49 files | scorecards — which model won, by how much, why |
| `docs/wip/**` | 148 files | failure taxonomies, fixture ladders, tuning plans |

**Operator decision 2026-07-14: fine for now.** Recorded so the acceptance is explicit rather than
accidental.

---

## If the split happens later

### The hard constraint: history is permanent

Moving files to a private repo protects everything **from that point forward**. It does not
un-publish what is already pushed. Deleted content stays reachable in clones, in forks, and via
GitHub's archive; even after a history rewrite GitHub serves orphaned blobs by SHA until asked to GC
them, and existing forks keep them regardless. **Forward-only and retroactive are different decisions
with wildly different costs.** Do not let a "just move it" framing hide that.

### Scope options

| Option | Cost | Buys |
| --- | --- | --- |
| **Forward-only** (recommended if the split happens) | low — no rewrite, no broken clones | everything new is private; accepts that the current playbook is already out |
| **Forward + retroactive purge** | high — `filter-repo` + force-push, every clone/fork breaks, GH support ticket to GC, *still* cannot clean existing forks | only worth it if the current playbook is genuinely competitive IP |
| **Make the whole repo private** | loses the public-project / shared-edge story | simplest possible; no split, no path plumbing |

### Mechanism options — **parked, operator will decide later**

| Mechanism | Trade-off |
| --- | --- |
| **Sibling clone + symlinks** | private repo cloned next to this one; symlink `data/eval/{configs,datasets}`, `autoresearch/`, `docs/guides/eval-reports/` into place; public `.gitignore` covers the links. **Paths stay identical, so Makefile targets and tests keep working.** No submodule auth pain in CI. |
| **Private git submodule** | clean provenance and versioning, but the mount point must be a single directory, and a clone without access hard-fails rather than degrading gracefully. |

The symlink pattern is already proven in-repo — `prod_validation_v1/episodes/` does exactly this,
which is why 30 "committed episodes" turned out to be 4.8 KB of paths.

---

## Open questions for the operator

1. Build the pre-commit prod-content guard? (Measured above; zero false positives with the allowlist.)
   The boundary currently rests on `.gitignore` and on nobody writing a tool that drops a transcript
   in the wrong place.
2. Does the methodology acceptance hold once prod-v3 ships, or does the split happen then?
3. If it happens: forward-only, or retroactive? (See "history is permanent".)
