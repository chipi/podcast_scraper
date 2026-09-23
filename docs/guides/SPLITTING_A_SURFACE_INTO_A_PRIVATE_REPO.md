# Splitting a surface out into a private repo

**Who this is for:** you are about to move a whole surface — infra, ops, a
corpus, a research area — out of this repo into a private one, and you want the
decisions already paid for rather than rediscovering them.

**Where this comes from:** the eval research split (arc 1 + arc 2, PR #2134 and
its follow-ups). 2,146 files, −198,669 lines, plus a week of consequences. Every
rule below is here because breaking it cost something real, and the cost is
named so you can judge whether it applies to you.

The worked example is `chipi/podcast-scraper-eval-data`; see
[Mounting the private eval research repo](EVAL_RESEARCH_MOUNT.md) for what the
result looks like from a user's seat.

---

## The one rule that decides everything

> **Tradecraft moves. Runtime stays.**

For eval that was: *research* (datasets, reports, tuned constants, harness)
moves; the *application* stays. For infra it will be some other pair — probably
secrets, topology and live state versus the code that consumes them. Write your
version of that sentence down before you move a single file, because every
argument later resolves to it.

Two corollaries that were not obvious:

- **Both axes of a discipline move together.** Eval had QUALITY (WER, ROUGE) and
  PERFORMANCE (wall time, peak RSS). Splitting them would have left half a
  discipline behind, so `data/profiles/`, `data/perf/` and the capture harness
  went with `data/eval/`.
- **The generator can stay while its output moves.** `scripts/build_v3_fixtures.py`
  stayed public because it produces the fixtures the public tests consume. Ask
  "who authors this?" not "who reads it?"

---

## Sequencing: two arcs, never one

**Arc 1 — copy everything in. Change nothing on the public side.**
**Arc 2 — delete from public, fix every calling site.**

Copying *in* does not authorise deleting *there*. Keeping them separate means
arc 1 can be wrong without breaking anything, and arc 2 is reviewable as a pure
deletion. It also gives you a window where both copies exist, which is when you
prove equivalence.

**While arc 2 is open, main keeps moving.** Before each rebase, diff what main
touched against what you delete:

- A **delete/modify conflict** git shows you. Fine.
- An **add on one side** git shows you *nothing*. This is the dangerous one.

Both happened. Main added `data/baselines/baseline-2026-W39.json` into a
directory arc 2 deletes — no conflict, silently lost. And a bug fix (#2075)
landed in a file arc 2 deletes; a clean-looking resolution would have dropped
it. **Port first, then rebase.**

---

## Finding what to move: three sweeps, and why the first two failed

Do not grep for a word. It fails in both directions, and it failed four times:

| sweep | basis | missed |
| --- | --- | --- |
| 1 | the word "eval" | every `autoresearch/` doc — the directory moved, its ADRs stayed |
| 2 | the word "autoresearch" | `RFC-015`, `RFC-041` and the five ADRs they were built on |
| 3 | **do the paths this doc cites still exist?** | nothing of that class |

Sweep 3 is the one to start with: extract every path literal in the repo and
test each against the pre-deletion tree. It narrows 200 candidates to ~30.

**Then read all thirty.** The shortlist is a filter, not a verdict. Four
documents matched every name-based signal and correctly stayed, each identified
only by opening it:

- `enrichment/eval/` — *application runtime* that happens to be called eval
- `RFC-090` — runtime search architecture; mentions "eval" 29 times
- `RFC-116` — its own header calls it the driver for a public fixture generator
- two ranking scorers — they grade a product surface

> A grep for a word is not a test of subject. Neither is a path heuristic. It
> only narrows what you have to read.

**Watch for audit false positives too.** Four "dangling" paths turned out to be
package-relative (`podcast_scraper/data/known_models.yaml`, resolvable at
runtime) and one was a `RELEASE_vX.Y.Z` template placeholder. Verify before
"fixing" working code.

---

## The mount: one folder, both contexts

The private repo clones into a folder inside the public one, so one agent sees
both. That is the whole trick, and it needs four changes before the first clone.

### 1. `.gitignore` — both forms, before the clone exists

```gitignore
/eval-data/
/eval-data
```

Both, deliberately. The trailing slash matches a **directory** (a plain clone);
the bare form matches a **symlink** (the sibling+symlink setup). Only the first
was there initially, and `git status` showed an untracked `?? eval-data` for the
symlink variant — exactly the leak the entry exists to prevent.

**It must exist before the clone does.** Git will not add a nested repo's
contents — it writes a gitlink — but that still records the private repo's
existence and commit SHA in public history. And if the nested copy ever loses
its `.git` (a file copy, an rsync, a Docker build context), every file in it
becomes addable by `git add -A`.

### 2. Tool exclusions — every tool that walks the tree

Each of these was found by a gate going red, not by foresight:

| tool | exclusion |
| --- | --- |
| black | `\| eval-data` in the `exclude` regex (`pyproject.toml`) |
| isort | `skip_glob = [..., "eval-data/**"]` |
| markdownlint | `--ignore "eval-data/**" --ignore eval-data` |
| mkdocs | not in `docs_dir`, so nothing needed — verify |

### 3. A setup guide

Someone will be told "check out the other repo too" and needs to know where and
why. One page, linked from the guides index.

### 4. No pointers back

If the private side lives on a private network, **nothing public may point at
it** — not a URL, not a hostname, not in a comment. The mkdocs nav carries an
explicit note saying the link is deliberately absent. 88 such links across 26
files had to be reverted once; write the rule down before that happens.

---

## Citations: the thing that breaks six months later

When code cites evidence that now lives elsewhere, the citation rots and nothing
notices. 19 of them rotted here, and exactly one test could see it.

**Three forms, and reject anything else:**

```text
private-repo-name:path/to/doc.md   a doc in the private repo
docs/path/to/doc.md                a public doc — MUST resolve locally
#1234                              an issue
```

The middle one is what catches the regression: a citation left as a bare local
path fails immediately once the file is gone.

**Do not point at the mount.** The first fix used `eval-data/…`, which reads like
a local directory that is present in maybe one working tree in five. The check
almost never ran and the value was misleading. Name the repo instead.

**Put the resolving check in the repo that has both halves.** The public repo has
the citation but not the evidence; a developer's checkout has the evidence only
if they mounted it. The private repo has both — the docs on disk, and the public
code via its pinned dependency — so the check runs unconditionally there, and is
impossible here. ~140 lines.

**Publish the claim, withhold the data.** Every option here carries a
`headline_metric` and a `measured_at` alongside its citation — the number, its
date, the corpus size, what it superseded. A reader can see the claim is
specific and dated without being handed the raw material. The gap worth closing
is not visibility; it is that nothing verifies the published claim against the
withheld evidence.

---

## Silent failure modes, all of which bit

The recurring shape: **a surface with no assertion over it**. Three separate
incidents in one week, same root cause.

| pattern | what it did |
| --- | --- |
| `pytest.skip` when a file is missing | two tests went vacuous the moment the file moved; suite stayed green |
| `if script.exists():` around a feature | a CLI flag silently became a no-op |
| an empty leftover directory | `import pkg.sub` still **succeeded** — Python treats it as a namespace package, so a missed caller passes locally and fails in CI |
| `cmd && lint \|\| echo "not installed"` | swallowed the linter's own exit code — 21 findings printed, then "skipped", exit 0 |
| `;`-chained make recipe | printed "synced." after every `rsync` failed |

Rules that follow:

- **Never skip on a missing committed fixture.** Absence is a failure.
- **Delete empty directories after moving their contents**, or the import lies.
- **Guard, then run** — never `a && b || echo`.
- When a check "passes", ask *what would make it fail?* If nothing, it is not a
  check.

---

## Before deleting anything

- **Who WRITES here, not just reads?** `build_v3_fixtures.py` wrote its dataset
  into a directory arc 2 deleted — a live break, not a stale comment. Grep for
  the path as a write target.
- **Prove byte-equality for anything generated.** When a generated artifact
  moves, regenerate at the new location and hash it against the old. All three
  files matched, so the move changed location only.
- **Check whether the tooling you plan to build already exists.** The big
  registry refactor was scoped around "nothing gates drift between these two
  copies." `make profiles-check` already did, and already ran in `ci-fast`. The
  premise was false and most of the work was unnecessary. Look before pitching.

---

## Verify by breaking

Every assertion added during this split was mutation-tested. Several were
vacuous until broken:

- the citation check passed on 42 refs it never resolved
- the rationale check missed citations written without a `.md` suffix
- a "3 copies" assertion looked green because the `sed` that was meant to break
  it silently failed

The discipline: **change the thing the test is about, watch it go red, restore.**
If you cannot make it fail, you have not written a test. A one-off script that
breaks each surface in turn and reports CAUGHT/MISSED is worth the ten minutes.

---

## CI mechanics that will surprise you

- **Pushing cancels runs.** `cancel-in-progress: true` on a workflow means a new
  push kills the in-flight run; `false` means it *queues*, and GitHub keeps only
  one pending run per group, so a third push cancels the second. Four of this
  week's commits never completed a full run because of rapid pushes. Batch, then
  push once.
- **Heavy lanes may not run on PRs at all.** The full integration and e2e lanes
  here gate on `event_name == 'push'` and a main ref. A green PR is not a green
  branch — check which lanes actually ran before believing it.
- **Tier-3 / nightly suites are where stale expectations hide.** A spec that runs
  only nightly sat red for four nights after a rename because no PR ever
  exercised it.

---

## Checklist

**Arc 1**

- [ ] Write the tradecraft/runtime sentence for your surface
- [ ] Sweep by paths, not names; read every candidate
- [ ] Copy in; nothing on the public side changes
- [ ] Bring the tests and gates, not just the content — they are what make it run
- [ ] Private gate green (`make ci` or equivalent)

**Arc 2**

- [ ] `.gitignore` both forms, before any clone
- [ ] Tool exclusions: black, isort, markdownlint, doc build
- [ ] Setup guide written and linked
- [ ] Port anything main added to a directory you delete
- [ ] Delete; fix every calling site, including writers
- [ ] Remove emptied directories (namespace-package trap)
- [ ] Citations converted to the three-form rule
- [ ] Resolving check added in the repo that has both halves
- [ ] Every new assertion mutation-tested
- [ ] Full gate chain, then one push
