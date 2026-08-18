# Agent-navigable codebases

> **Document structure:**
>
> - **This document** — why the repo is documented and governed the way it is, and how to apply it elsewhere
> - [Testing Strategy](TESTING_STRATEGY.md) — the test pyramid and decision criteria
> - [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md) — how the tiers and fixtures fit together
> - `scripts/tools/check_doc_structure.py` — the enforcement this document argues for

## The problem this solves

A growing share of the work in this repo is done by contributors who arrive with **no memory of
yesterday**. Agents triaging an issue, agents fixing a bug, a new engineer, or you in six months.
They are handed a symptom and a repository, and everything they do next depends on which file they
open first.

Traditional documentation strategy optimises the wrong variable for that reader. It asks *is this
document good?* The question that actually decides the outcome is *will this document be reached?*

**A document's value is its quality multiplied by the probability it is found.** We had documents
scoring high on the first term and near zero on the second, and the product is what the reader
experiences.

## The evidence

On 2026-08-13 an agent was asked to remove a mock: the e2e suite substituted synthetic audio because
the test corpus shipped an unplayable placeholder. It searched `tests/fixtures/app-validation-corpus/v3`,
found no `.mp3`, and concluded the repository had no fixture audio. It then checked for `ffmpeg` in
two places, found none, concluded no encoder existed, and **hand-wrote MPEG-2 Layer III frames**,
probing five header variants in a real browser to find one Chromium would decode. It rewrote 36
fixture files and grew the committed corpus by 2 MB before the operator stopped it.

`tests/fixtures/audio/v3/` contained real audio for all 36 episodes — one directory up. `pip install
imageio-ffmpeg` provides ffmpeg in one command.

The instructive part is not the mistake. It is that **the answer was already written, three times**:

| Document | What it said |
| --- | --- |
| `tests/fixtures/README.md` | Titled *"Offline Podcast Fixtures (RSS + Transcripts + Audio)"*; lines 5–8 say `audio/` is versioned and to check `FIXTURES_VERSION` first |
| `docs/guides/E2E_TESTING_GUIDE.md` | `make serve-e2e-mock`, port 18765, *"Audio files: tests/fixtures/audio/*.mp3"* |
| `docker/mock-feeds/README.md` | The nginx sidecar that serves exactly those files |

None was thin. `tests/fixtures/README.md` is 342 well-written lines and answers the question in its
title. It failed because **nothing in the tree the agent was working in pointed at it** — every
document with the answer lived in the Python half of the repo, every document the agent touched
lived in `web/`, and there were no links across.

Two further findings from the same audit:

- The app's own `README.md` had **six dead spec links** — PRD-035, PRD-038, PRD-039, RFC-099,
  UXS-011, PLATFORM_API — in the file a new contributor opens first. Every one had been broken for
  months. Nothing failed, so nobody knew.
- A comment in the corpus generator asserted *"the player never decodes it in e2e."* That had become
  false. It read as justification and **actively corroborated the wrong premise**. It cost more than
  silence would have.

## Principles

### 1. Route, don't restate

Measure before writing: ~88% of this app's source files already carry a substantial top docblock.
*What is this file* was solved. What was missing was *which file do I start from*, *what contract
spans these files*, and *what enforces it*.

So directory documentation is a **routing table**: what each file owns, when you would reach for it,
the conventions that no single file's docblock can own, and the guard that enforces them. It does not
re-explain code.

This is not only about redundancy. Restated code is the **fastest-rotting** content in any document,
and prose cannot be mechanically verified. Writing about contracts and reasons — which change slowly
— is what keeps the unverifiable half true for longer.

### 2. Put the pointer where the reader is standing

The fix for the incident was not to write documentation. It was to add pointers **at the five places
the agent actually stood**: the helper it read, the surface map it edited five times, the corpus
directory it worked in, the generator it consulted.

A signpost of 15 lines in the right place beats 342 excellent lines one directory away. Rank
candidate locations by *where does someone land with this problem*, not by *where does this
information belong*.

### 3. A stale pointer is worse than a missing one

A missing document produces a search. A wrong document produces **confident action**. The false
comment in the corpus generator was worse than no comment, because it was trusted.

This is the argument for governance rather than good intentions: the only sustainable way to keep
pointers honest is to make the build fail when they rot.

### 4. Govern exactly what is mechanically checkable — and say what is not

We enforce three properties: required READMEs exist, they are not stubs, and **every relative
markdown link in the repo resolves**. We cannot enforce that prose is true, and the check's own
docstring says so plainly. A guard that overstates its coverage recreates the problem it was built
to solve.

This is also why principle 1 is load-bearing rather than stylistic: the unverifiable half must be
written about slow-moving things.

### 5. Measure before you gate

A gate that fails on day one gets disabled on day two. Before wiring the link check into CI we
measured: **19 broken links across 936 markdown files**, in 13 files. Small enough to fix outright,
so the check ships with **no allowlist and no exemptions** — which means every future failure is
genuinely new information.

Had it been 500, the correct move would have been a scoped check plus a visible backlog, not a
weakened one.

### 6. A guard that has never been red is not evidence

Every check here is **mutation-tested**: break the thing it claims to catch, watch it fail with a
useful message, restore, watch it pass. Two guards written during this work looked correct and
caught nothing until they were mutated — one probed only an element's centre, another compared a
UI bar against itself. Both were green, plausible and worthless.

Applies to test suites generally, and doubly to meta-checks, which nobody exercises by accident.

### 7. Name what is not covered

Every README here ends with known gaps: which store has no test, which composable is untested. A
coverage claim without a gap section reads as *no gaps*, and silence about a gap is a claim about it.

### 8. Documentation cannot fix a behavioural failure

The answer was in three documents the agent never opened. Better documents would not have changed
that. This work was paired with two operator rules — *a zero-result search is evidence about the
search, re-run it at repo root* and *a workaround's weirdness is evidence against your premise* —
because the failure was half informational and half behavioural, and only fixing one half fixes
nothing.

If you roll this out, roll out both halves.

## The mechanics

**Tree-root READMEs.** One at each root where someone lands and needs orientation — not every
directory. Here: the test tiers, the fixture trees, the two web apps, each app's e2e directory, and
the store/composable layers. Roughly 40–60 lines each:

1. A one-line purpose.
2. A table of the files: what each owns, and **when you would reach for it** — this column maps a
   symptom to a file, and is what a fresh reader uses.
3. The conventions that span the directory, with the reason attached.
4. Which guard enforces what.
5. Related documents, pointing outward.
6. Known gaps.

**Enforcement.** `scripts/tools/check_doc_structure.py`, wired into `_ci_body`, `ci-fast` and
`docs-check`. It runs in ~1.7 s across 936 files, which is what justifies its place in the
pre-commit path rather than a nightly. Required roots are an explicit list, so adding one is a
decision and the set cannot sprawl.

**Division of labour with existing linting.** `make lint-markdown` checks style and formatting.
This check verifies that the pointers are *true*. Neither subsumes the other.

## Applying this to another repository

1. **Find the boundary knowledge does not cross.** Usually language, package or app boundaries —
   ours was Python fixtures versus a TypeScript front-end. That boundary is where dead ends form.
2. **Reconstruct one real failure.** Take an issue that went badly and list what the person read, in
   order. The missing pointers are at those stops, not where you would have guessed.
3. **Inventory before writing.** How many files already have docblocks? What documentation exists
   but is unreachable? Ours failed by unreachability, not absence — writing more would have missed.
4. **Add pointers at the stops**, then READMEs at tree roots, then the check.
5. **Measure broken links before gating.** Fix if small; scope the check and publish a backlog if
   large.
6. **Mutation-test the check.**
7. **Add the behavioural rules** to whatever your agents read as standing instructions.

Expect roughly a day for a repository this size, and treat the link check as the durable artifact —
the READMEs will be rewritten many times; the guarantee that their pointers are live is what persists.

## Costs and limits

- **It is an assumption, not a measurement.** This app has no consumers yet, so we cannot A/B whether
  a fresh agent resolves issues faster. The bet is that a fresh reader lands in a directory more often
  than it lands on an issue with good pointers. If your triage step can emit entry points into the
  issue itself, that is cheaper and more targeted — **do that first**, and treat READMEs as the
  backstop.
- **Prose still rots.** Only pointers are guarded.
- **More documents is not the goal.** A README at every directory would dilute the signal that a
  README means *this is a place you land*.
- **Coverage is not uniform.** `.markdownlintignore` excludes `tests/fixtures/**` and `docs/wip/**`
  from style linting, so some governed files are structure-checked but not style-checked.

## Related

- [Testing Strategy](TESTING_STRATEGY.md) · [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md)
- [`tests/README.md`](https://github.com/chipi/podcast_scraper/blob/main/tests/README.md), [`web/README.md`](https://github.com/chipi/podcast_scraper/blob/main/web/README.md) — the pattern applied
- [`tests/fixtures/audio/README.md`](https://github.com/chipi/podcast_scraper/blob/main/tests/fixtures/audio/README.md) — the incident's subject, documented
