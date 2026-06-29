# Handover — RFC-097 v3 GI migration on `podcast_scraper-ai-ml-improvements`

**For:** the agent operating in `podcast_scraper-ai-ml-improvements` worktree
**From:** the agent working main in `podcast_scraper-FUTURE` on 2026-06-29
**Why now:** the operator wants to run RFC-088 enrichment tests against the corpora there. Pre-enrichment, the GI envelopes need to be at RFC-097 v3 (typed `MENTIONS_PERSON`/`MENTIONS_ORG`) so the enrichers' person/org-aware logic has the right edges to read.

---

## Context — what's been verified

- Migration script `scripts/migrate_gi_to_v3.py` is **identical** on `main` and on `feat/consumer-remember` (verified with `diff -q`). Same code, same result, doesn't matter which branch performs the migration.
- Migration is **idempotent** (per script docstring + source reading): re-running on already-v3 files is a no-op.
- The migration script handles three things per file:
  1. Bumps `schema_version` `2.0` → `3.0`
  2. Rewrites legacy `MENTIONS` edges to typed `MENTIONS_PERSON` / `MENTIONS_ORG` based on the edge target's id prefix (`person:` / `org:`)
  3. Normalises `Insight.insight_type` vocab (legacy `fact`/`opinion` → v3 `claim`/`observation`; out-of-vocab → `unknown`)
- KG-side legacy `MENTIONS` edges (`Topic → Episode`) stay as-is **by design** — see `src/podcast_scraper/migrations/gil_kg_identity_migrations.py:205-211`. Not a gap.

## Current state — what we expect to find

All seven corpora under `.test_outputs/manual/` in this worktree are at GI **v2.0**, KG **v1.2**:

```
_rediar-iltb-quarantine-20260611-233057  gi=2.0  kg=1.2
my-manual-run-10                          gi=2.0  kg=1.2
prod-10                                   gi=2.0  kg=1.2
prod-pilot                                gi=2.0  kg=1.2
prod-v2                                   gi=2.0  kg=1.2
rediar-974-reprojected                    gi=2.0  kg=1.2
rediar-batch-iltb                         gi=2.0  kg=1.2
```

KG v1.2 is **the current state** — no migration needed there. Only GI needs `v2.0 → v3.0`.

**Dry-run from `podcast_scraper-FUTURE` (read-only) showed `prod-v2` would have:**

```
files scanned:                 209
files that would change:       209 (all)
schema bumps 2.0 → 3.0:        209
MENTIONS → MENTIONS_PERSON:    376
MENTIONS → MENTIONS_ORG:       718
insight_type vocab normalised: 0  (already current)
files errored on load:         0
```

Other corpora are smaller but same shape. Re-run the dry-run for each before applying (see below).

## Recommended scope — which corpora to migrate

In priority order:

| Corpus | Reason | Recommend |
|---|---|---|
| `prod-v2` | The operational corpus you'll run enrichment tests against | **YES (must)** |
| `prod-10`, `prod-pilot`, `my-manual-run-10` | Other prod-like fixtures the operator may swap to | YES (low-cost, idempotent) |
| `rediar-batch-iltb`, `rediar-974-reprojected` | Diarization rerun fixtures; only relevant if you'll re-run enrichment against them | only if asked |
| `_rediar-iltb-quarantine-20260611-233057` | Underscore prefix suggests quarantined / archived | skip unless asked |

Default safe play: migrate the top four. Skip the rest.

## Step-by-step

### 1. Dry-run each corpus first (read-only sanity check)

Save this as `scripts/_dry_run_v3.py` (local, don't commit) or run inline:

```python
# Adjust BASE to whichever corpus you're checking
BASE = '.test_outputs/manual/prod-v2'

import json, copy, glob
from podcast_scraper.migrations.gil_kg_identity_migrations import migrate_gi_document_v3

files = sorted(glob.glob(f'{BASE}/**/*.gi.json', recursive=True))
totals = {'scanned': len(files), 'would_change': 0, 'schema_bumps': 0,
          'mentions_person': 0, 'mentions_org': 0, 'errored': 0}

for f in files:
    try:
        before = json.load(open(f))
    except Exception:
        totals['errored'] += 1; continue
    after = migrate_gi_document_v3(copy.deepcopy(before))
    if before == after: continue
    totals['would_change'] += 1
    if before.get('schema_version') != '3.0' and after.get('schema_version') == '3.0':
        totals['schema_bumps'] += 1
    for be, ae in zip(before.get('edges') or [], after.get('edges') or []):
        if be.get('type') == 'MENTIONS' and ae.get('type') == 'MENTIONS_PERSON':
            totals['mentions_person'] += 1
        if be.get('type') == 'MENTIONS' and ae.get('type') == 'MENTIONS_ORG':
            totals['mentions_org'] += 1

print(f'{BASE}:', totals)
```

Run with `.venv/bin/python` (you are running from the ai-ml-improvements worktree, which has its own venv).

### 2. Backup before apply (cheap insurance)

```bash
cp -r .test_outputs/manual/prod-v2 .test_outputs/manual/prod-v2.pre-v3-migration
```

(Same for each corpus you migrate. They're not huge.)

### 3. Apply

```bash
.venv/bin/python scripts/migrate_gi_to_v3.py \
  --corpus .test_outputs/manual/prod-v2
```

Repeat for `prod-10`, `prod-pilot`, `my-manual-run-10`.

### 4. Verify post-apply

```bash
find .test_outputs/manual/prod-v2 -name '*.gi.json' | \
  head -5 | \
  xargs -I{} python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('schema_version'))" {}
# Expect: 3.0 3.0 3.0 3.0 3.0
```

Spot-check one file by hand — open in editor, confirm:
- `schema_version: "3.0"`
- At least one edge with `type: "MENTIONS_PERSON"` or `type: "MENTIONS_ORG"` (any file with person/org mentions)
- Insight node `insight_type` values are in `{claim, observation, unknown}` — no `fact` / `opinion` left

### 5. If anything looks wrong

Roll back from the backup:

```bash
rm -rf .test_outputs/manual/prod-v2
mv .test_outputs/manual/prod-v2.pre-v3-migration .test_outputs/manual/prod-v2
```

## What NOT to do

- **Don't touch `.kg.json` files.** KG envelopes legitimately stay at schema 1.2 with legacy `MENTIONS` edges (`Topic → Episode` discovery edges). The RFC-097 source code explicitly preserves them.
- **Don't commit the backup directories** — they're under `.test_outputs/` which is gitignored. Leave them as local rollback insurance.
- **Don't run the migration on a corpus that's still being written to** by a live pipeline job. Wait for the job to finish.
- **Don't generate dry-run from the FUTURE worktree against this worktree's corpus** — the venvs differ and Python imports can be tricky across worktrees. Run from THIS worktree's `.venv` against THIS worktree's corpus.

## After migration, the operator will run

The RFC-088 enrichment test plan, against (probably) `prod-v2`. The relevant enrichers as of 2026-06-29 main:

- `temporal_velocity` (deterministic — needs typed KG/GI; v3-migrated corpus is the prereq)
- `topic_similarity` (ml tier; `--with-ml` CLI flag)
- `nli_contradiction` (ml tier; `--with-ml` CLI flag)

Those should be the operator's call to make, not yours.

## References

- RFC-097: `docs/rfc/RFC-097-unified-kg-gi-ontology-v2.md`
- Migration script: `scripts/migrate_gi_to_v3.py`
- Migration module: `src/podcast_scraper/migrations/gil_kg_identity_migrations.py` (function `migrate_gi_document_v3`)
- Original RFC-088 implementation that needs this: PR #1127 (merged on `main` as `32ffb9b9`)
- Enrichment test plan: drafted verbally by main-worktree agent on 2026-06-29, ask operator if you need it as a doc
