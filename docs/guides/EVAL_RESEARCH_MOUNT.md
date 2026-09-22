# Mounting the private eval research repo

**Who this is for:** you have been told "check out the eval repo too" and want
both the application and the research in one tree, so one agent — or one person —
can see both at once.

**Why it exists:** most eval work is not separable from application work. To make
a stage measurable you usually have to refactor it first; to fix a bug you often
want a run on the side to prove the fix. Two windows and two contexts make that
awkward, so the private repo mounts inside this one.

The research lives in **`chipi/podcast-scraper-eval-data`** (private). It holds
real episode transcripts, human-authored gold references, promoted baselines and
run results — publisher copyright, retained for internal research. That is why it
is a separate repo and why it is private.

## Setup

```bash
# from the root of THIS repo
git clone git@github.com:chipi/podcast-scraper-eval-data.git eval-data

# then read its README — it stands alone and explains the rest
less eval-data/README.md
```

That is the whole setup. `eval-data/` is already in `.gitignore`, and this repo's
linters and formatters already exclude it (see below), so nothing else is needed.

If you do not have access to that repo, everything in this one still works. The
eval harness is simply absent; no target here depends on it.

## Why `eval-data/` and not somewhere else

It maps one-to-one onto the repo name, so a reader who sees the directory can
guess what it is. It collides with nothing (`data/`, `docs/`, `config/`,
`scripts/`, `tools/`, `web/`, `infra/` are all taken). And it is deliberately
**not** dot-prefixed: a `.research/` directory looks safer and is not — black and
flake8 walk into a dot-directory exactly as readily, which was measured rather
than assumed.

## What was changed here to make the mount safe

| where | change | why |
| ----- | ------ | ---- |
| `.gitignore` | `/eval-data/` | so `git add -A` can never stage research |
| `.flake8` | `exclude = eval-data, …` | `make lint` runs `flake8 .` and would lint the nested tree |
| `pyproject.toml` | black `exclude` regex | `make format-check` runs `black --check .` |
| `pyproject.toml` | isort `skip_glob` | same reason |

`pytest` needed nothing: `testpaths = ["tests"]` already confines collection.
`mkdocs` needed nothing: `docs_dir` is `docs/`.

### The gitignore entry must exist before the clone

Git will not add a nested repository's *contents* — it writes a gitlink instead —
but that still records the private repo's existence and commit SHA in **public**
history. Worse, if the nested copy ever loses its `.git` (a plain file copy, an
rsync, a Docker build context that flattens it), every transcript inside becomes
addable by `git add -A`.

The entry is committed here already, so this is only a warning for anyone
tempted to mount it somewhere else.

## Alternative: siblings plus a symlink

If you would rather the private repo never sat inside this working tree at all:

```bash
git clone git@github.com:chipi/podcast-scraper-eval-data.git ~/projects/podcast-scraper-eval-data
ln -s ~/projects/podcast-scraper-eval-data eval-data   # still gitignored
```

Same single-tree convenience, and the research is physically outside this repo —
so the exclusions above become a convenience rather than the only thing between a
transcript and a public commit. Git does not follow a symlink into another repo.

## Two things the real mount taught us

Both were found by actually mounting the repo and running the gates, not by
reasoning about it. A probe with a dummy directory passed and missed both.

### A symlink is not a directory, to git

`.gitignore` carries **both** `/eval-data/` and `/eval-data`. The trailing-slash
pattern matches a directory — a plain `git clone` here. It does **not** match a
symlink, so with the sibling+symlink setup `git status` showed an untracked
`?? eval-data`, which is precisely the leak the entry exists to prevent. Both
forms are now present.

### `..` means different things in the two setups

With a **clone** at `eval-data/`, the parent directory is the public repo, so
this works from inside the mount:

```bash
make eval-against-local PATH_TO_CHECKOUT=..
```

With a **symlink**, `..` resolves to the symlink *target's* real parent
(wherever the private repo actually lives), not to the mount point. Use an
explicit path there:

```bash
make eval-against-local PATH_TO_CHECKOUT=~/projects/podcast_scraper
```

### Instruction files layer

Both repos carry their own `AGENTS.md` and `CLAUDE.md`. Working inside
`eval-data/` an agent sees the private repo's rules; at the public root it sees
this repo's. That is the intent — the two rule sets differ in ways that matter
(the private one permits committing audio, this one never should).

## Working across both

- **This repo is the system under test.** The eval repo depends on it as a pinned
  package and calls its assembly functions rather than re-wiring them. That
  constraint has its own decision record — `ADR-003` over there, which was
  `ADR-111` here before it moved.
- **Application gates stay here.** `scripts/eval/score/rank_discover_v1.py` and
  `rank_scenarios_v1.py` look like eval and are not: they import only
  `podcast_scraper.server.*` and read public fixtures. See that directory's
  README.
- **`src/podcast_scraper/enrichment/eval/` is runtime**, not research — it is the
  admission gate, and its metrics ship inside the wheel.
