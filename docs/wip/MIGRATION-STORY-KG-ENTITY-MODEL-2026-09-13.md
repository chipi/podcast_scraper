# Migration story: the KG entity-model changes

**Opened:** 2026-09-13. **Status:** plan. Nothing here has been run.

Four changes on `fix/llm-context-budgets-and-registry-plumbing` alter what the pipeline writes
into `*.kg.json` and `*.gi.json`. Exactly one of them has a migration, and that asymmetry is
deliberate — but it has to be written down, because "upgraded" will not mean "clean".

## What changed, and what each needs

| # | change | new artifacts | existing artifacts |
| --- | --- | --- | --- |
| #2055 | stray `)` stripped from display names | correct | keep the debris |
| #2057 | `Object` — third entity kind, schema 2.1 | correct | **m0008 stamps 2.1**; kinds unchanged |
| #2059 | role words episode-scoped (`person:host` → `person:speaker-{ep}-host`) | correct | keep the global id |
| #2060 | a known host/guest is not demoted to `mentioned` | correct | keep `mentioned` |

**Only #2057 has a migration, and it rewrites no nodes.** `m0008` sets `schema_version` to `2.1`
so readers accept `Object`, and touches nothing else.

## Why the other three cannot have one

Not "we did not get to it" — they are **not migratable**.

Each fixed a defect at EXTRACTION time, and the evidence the fix needs was destroyed before the
artifact was written:

* **#2057** — `_normalize_entity_kind` coerced the model's answer to `person` and wrote only the
  result. A corpus records `Person(Norman Conquest)` with no trace of the `event` the extractor
  actually reported. Recovering it means guessing from the name, which is inventing data.
* **#2059** — the id `person:host` records that *something* was called "Host"; it does not record
  which episode's host, and the ids from 54 episodes are byte-identical. They cannot be separated
  after the fact.
* **#2060** — `role="mentioned"` was written because the typed node was skipped. The host/guest
  fact lived in `detected_hosts`/`detected_guests` at build time, which the artifact does not
  carry.

A migration that guessed any of these would be worse than the pollution: pollution is visible and
countable, invented data is neither.

**So historical cleanup is a RE-DERIVE, not a migration.** `--pipeline-stage rederive_only` re-runs
the LLM stages against the stored transcript — no audio, no Whisper, no re-transcription.

## The mixed state after deploy, and why it is safe

A corpus that has run m0008 but not been re-derived is `schema_version: 2.1` containing pre-fix
nodes. Verified no reader mishandles it:

* `Object` nodes are *permitted* by 2.1, not *required* — a 2.0 artifact is a strict subset.
* Every unknown node type is skipped tolerantly by the readers (`kg/corpus.py:240` and siblings).
* The role-word filter now recognises the **legacy global** `person:host` as a placeholder, so the
  54-episode phantom disappears from all twelve consuming surfaces **immediately on deploy**,
  without any corpus pass. That is the one piece of historical cleanup that needs no GPU.

What stays visibly wrong until a re-derive: `Lukasz Kaiser)` labels, hosts rendered `mentioned`,
and non-people occupying Person nodes.

## Rollback

**There is none after m0008.** Pre-branch code rejects every artifact, because `kg/schema.py` on
`main` requires `schema_version == "2.0"` exactly. Rolling the image back without restoring a
corpus backup makes every episode unreadable.

Mitigation, in order:

1. Run `.github/workflows/drill-corpus-upgrade.yml` first — restore + migrate + verify against a
   **real prod backup**. The failure surfaces on a copy.
2. Confirm a fresh backup exists before running `upgrade run` on prod.
3. If a rollback is ever needed: restore the backup, then deploy the old image. Not the reverse.

## Sequencing

Migrations are **manual**. `deploy-prod.yml` only tracks the upgrade fixture baseline; nothing
auto-runs them, and nothing runs them at container start. The only runner is
`cli upgrade run --corpus-dir ...` (`make upgrade-corpus`).

```text

1. merge + deploy                        code only; corpus untouched, still 2.0
2. drill-corpus-upgrade.yml              restore + migrate + verify on a COPY
3. cli upgrade status --corpus-dir ...   expect 0008 pending
4. cli upgrade run --corpus-dir ...      stamps 2.1; seconds, no GPU
5. verify: top_people has no person:host phantom   <- expected to pass with NO re-derive
6. re-derive, targeted                   the GPU-expensive part, below
```

Steps 1-5 are cheap, reversible-by-backup, and fix the single most visible defect. Step 6 is a
separate decision.

## The re-derive, costed

```text
full corpus     1,930 eps x 440s measured GI+KG = 236 GPU-hours (~9.8 days)
queued backfill 1,452 eps                        ~7 GPU-days
```

Both want the same DGX. A blanket re-derive is the same order as the backfill itself, so
**targeting is the plan, not an optimisation**:

```text
  5% of episodes affected ->  96 eps =  11.7 GPU-hours
 10%                      -> 193 eps =  23.6
 20%                      -> 386 eps =  47.2
 50%                      -> 965 eps = 117.9
```

### Targeting without reintroducing guessing

A cheap scan picks **what to reprocess**; the LLM decides **the answer**. A false positive then
costs GPU time, never wrong data — no heuristic output enters the corpus.

Suspect episodes, all from evidence we own:

1. a `Person` node whose name matches a feed title in `feeds.spec.yaml` (#2057)
2. a `Person` node with an org marker — `Inc`, `LLC`, `University`, `Center`, `Foundation` (#2057)
3. any `person:host` / `person:guest` / role-word id (#2059)
4. a display name with an unbalanced paren (#2055)
5. a `Person` with `role="mentioned"` that also has a `HOSTS` or `GUESTS_ON` edge (#2060) — the
   dangling-role signature, and a precise one

**Step 0 is to run that scan read-only and count.** 5% versus 50% picks a different plan, and the
scan costs minutes.

### Recommended order

**After the backfill, targeted.** The backfill runs on corrected code, so those 1,452 episodes are
born clean and never need re-deriving; then re-derive only the suspects from the existing 1,930.
Lowest total GPU. Cost: the remaining pollution stays visible until then — but the worst of it
(`person:host`) is already gone at step 5.

## Acceptance

* [ ] Drill green against a real prod backup before prod `upgrade run`
* [ ] `upgrade status` shows 0008 pending, then applied
* [ ] `top_people` has no `person:host` entry **without** a re-derive
* [ ] Suspect scan run and counted before any GPU is committed
* [ ] Re-derives run on the DEPLOYED image, never a local build
* [ ] Scan re-run afterwards, returning a materially smaller set
* [ ] GPU cost compared against the estimate above rather than assumed to have matched

## Known gaps

* **The Norman Conquest may survive all of this.** `top_people` ranks on `SPOKEN_BY` quotes in
  `gi.json`, and #2057 fixes `entity_kind` in `kg.json`. Those are different layers — open
  question on #2057.
* **Guest attribution is unfixed** and is a separate arc. Until it lands, a re-derive reproduces
  today's attribution.
* `corpus_format_version` 1 → 2 is read only by `import_local_snapshot.sh`; nothing enforces it at
  runtime. The gate is advisory.
