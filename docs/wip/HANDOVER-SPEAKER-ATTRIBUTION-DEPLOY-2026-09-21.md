# Handover — speaker attribution, ready to validate (2026-09-21)

Short version: **five commits are on `main` and not pushed.** The code is clean under every local
gate; the thing that needs a human is the *validation*, and epic
[#2097](https://github.com/chipi/podcast_scraper/issues/2097) is that validation. Read this, then
work from the epic.

## What is sitting on main, unpushed

| commit | what |
| --- | --- |
| `e1ff39fb` | the screenplay guard said "named", the code meant "has turns" |
| `3ddcb3a4` | six holes in the transcript-speaker gate (plain text exempt, no sibling retry, refused bytes dropped, one-voice files accepted, …) |
| `74a3a4f9` | `canonical_person_name` corrupted a name containing a bracket pair |
| `e15038c4` | **new migration** `m0010_canonical_person_names` (#2130 step 1) |
| `c9d6ab0b` | `retranscript_only` handed the relabel names it then threw away |

**The next push is the last before deploy.** Nothing here has been pushed.

## The backup corpus is already available — you do not need prod to validate this

This is the part worth knowing before planning anything: a real production corpus snapshot is
restorable on demand, and the whole migrate-and-verify sequence already has a workflow around it.

* **Snapshots:** `chipi/podcast_scraper-backup` releases, tagged `snapshot-prod-YYYYMMDD`
  (`gh release list --repo chipi/podcast_scraper-backup | grep snapshot-prod-`).
* **Locally:** `make restore-corpus-prod` pulls the newest compatible snapshot into `corpus/`.
* **In CI, end to end:** `.github/workflows/drill-corpus-upgrade.yml` — "Drill — corpus upgrade
  (restore + migrate + verify)". It restores the latest prod backup into a throwaway GHA
  workspace, runs the *exact* `upgrade` command **in the published image**, and asserts: migrations
  pending → dry-run plan → apply → **per-migration verify** → no data loss → nothing still pending
  → smoke the upgraded corpus. It never touches prod.

**Run that drill against `main` before the deploy.** Its own header says to, for exactly this case:
"run BEFORE shipping any major version whose migration touches the corpus", and "the upgrade is
IN-PLACE and STOPS AT THE FIRST FAILING migration — so a bad migration can leave a partially
migrated corpus. This drill is how we find that out on a copy, not on prod." `m0010` is a new
migration that rewrites corpus artifacts, so it is precisely the case the drill exists for.

The drill's VERIFY step is also newly meaningful: `m0010` implements `verify()` rather than
defaulting to `"no verification defined"` — the failure mode already recorded in
`upgrade/state.py`, where the ledger claimed a version the data did not have and nothing in the
system could tell.

## What I already rehearsed, and on what

I restored `snapshot-prod-20260914` (2,257 episode artifacts, 11,945 JSON files) and applied
`m0010` to a **copy**:

```
2257 episodes scanned, 91 changed
  (39 ids remapped, 33 of them MERGING into an id already in that episode;
   1 duplicate node id folded; 162 published names canonicalised), 0 unparsable
2nd apply: 0 changed   ← idempotent        verify: ok=True
```

`person:peter-attia-md` 73 nodes → 1; edge-junk names 67 → 3 (all Organization nodes, correctly
untouched); duplicate node ids 1 → 0; dangling person edge endpoints 0 → 0; every artifact still
parses; no leftover `.tmp`.

That local copy lived in a session scratchpad and is **gone**. Re-create it with
`make restore-corpus-prod`, or just let the drill do it.

## What is NOT validated — this is the handover

Of the epic's runbook (`docs/wip/POST-DEPLOY-SPEAKER-ATTRIBUTION-2026-09-14.md`: steps 0a, 0, 1, 2,
3, 4 plus nine activities), **only Step 2's new migration has been rehearsed end to end.**

Not done: Step 0a on the live corpus, the 0007/0008/0009 applies (rehearsed 2026-09-14, not by me),
Step 3 against real episodes, Step 4's coherence check, and activities 2–5 (the #2082 repair of 147
mis-paired episodes).

Two caveats that change numbers rather than correctness:

1. **My counts were measured before the #2082 repair.** Epic activity 7 already says every
   #2075-era corpus measurement came off a corpus where 12.2% of metadata points at another
   episode's transcript. `m0010` reads those same fields, so its 39/162/91 will move. The migration
   is idempotent and rewrites whatever is there, so this is a sizing caveat, not a correctness one.
2. **Two censuses of the publisher-transcript episodes disagree.** The epic counts 133 across five
   feeds; my snapshot scan counted 250. Different populations — mine includes superseded runs, the
   same trap the epic already flags for its 1,956-vs-2,256 comparison. Not reconciled. Do not quote
   them against each other.

One more that is a genuine gap: `retranscript_only` has still **never fetched a live publisher
URL**. I verified its hand-off to the relabel on the exact bytes it writes — which is how the
name-discarding defect in `c9d6ab0b` was found — but the fetch half needs network access to the
feeds. The epic's blocking pre-check ("verify it on ONE before any batch") stands.

## Suggested order

1. `pytest tests/unit tests/integration` — it was still running when the session ended and **never
   reported**. Everything is clean under flake8 / black / isort / `mypy src` (680 files) and the
   targeted suites (1,779 workflow + diarization, 150 upgrade, 245 entity-identity), but the full
   suite has not answered against the final tree.
2. ~~`drill-corpus-upgrade.yml` against `main`~~ — **this cannot run before the push, and the line
   above was wrong.** The drill upgrades *with the published image* (`IMAGE_TAG`, default `main`),
   and `docker.yml:115` publishes only on a push to `main`, so against unpushed commits it would
   drill the OLD code and report green. The order is: push → the image publishes → *then* the
   drill. What replaces it beforehand is a local walkthrough on a restored snapshot
   (`make restore-corpus-prod`), which is what was done on 2026-09-21 — see the runbook.
3. Push, deploy.
4. Then the epic, starting at Step 0a. Nothing before that step produces a number worth quoting.
