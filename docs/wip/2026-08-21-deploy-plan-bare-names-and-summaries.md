# Deploy plan — #1685 / #1686 / #1799 (written 2026-08-21)

The one legitimate sequence for getting this change set into production. Written because the
existing docs do not cover it: `RELEASE_PLAYBOOK.md` is the general release process and
`PROD_RUNBOOK.md` is steady-state operations — neither describes a change that **rewrites
artifact ids in an existing corpus**, which is what makes this one order-sensitive.

**Status: nothing here has been executed.** Production is untouched.

---

## What is being deployed

| Issue | Change | Touches production data? |
|---|---|---|
| **#1686** | a lost summary is marked on the artifact, retried once, and requeueable | No — new episodes only |
| **#1799** | test-isolation fixes and guards | No — tests only |
| **#1685** | bare person names are episode-scoped instead of global | **Yes, via m0007** |
| **#1802** | `RcloneStorageBackend` keeps an absolute `base_path` absolute | No — prod uses `""` |

Only **#1685** requires a migration. Everything else ships with the image.

---

## Pre-flight (before touching anything)

**1. Back up the corpus.** m0007 rewrites `.gi.json` and `.kg.json` in place. Atomic per file
(tmp + `os.replace`), so a crash cannot truncate one — but a wrong verdict is still a wrong
verdict, and there is no un-migrate.

**2. Check for real mononyms.** This is the one case where the change destroys something
legitimate. `upgrade plan` counts but does not name; list them first:

```bash
# On the corpus: every single-token person id, with episode counts
podcast-scraper upgrade plan --corpus /app/output   # counts only
# then grep the artifacts for the actual names before accepting the plan
```

Grimes / Beyoncé / MrBeast-class names have no surname to resolve to and will be scoped
permanently unfollowable. No allowlist exists yet. If any turn up, stop and decide.

**3. Note the current audit numbers** so the after-state is comparable:

```
208 occurrences of 172 single-word person ids — 12 resolvable, 0 ambiguous, 196 orphan
1931 person entities; 155 single-word (KG-only count); 10 spanning >1 show
9/678 content-quality defects — 8 absent summaries, 0 blank, 1 transcript echo
```

---

## The sequence — order is load-bearing

### 1. Deploy the image FIRST

Not optional, and not the intuitive order. The `person:unresolved-…` branch of
`is_unresolved_speaker_placeholder` ships **with the image**. Migrating a corpus that an older
image is still serving means that image does not recognise the scoped ids, so:

- every scoped id renders as a **followable entity card** (scoping preserves `name`, so it shows
  as "Jensen")
- its derived interests can mint `person:unresolved-…` tokens into user profiles

That is strictly worse than not migrating at all. The reverse order is safe: a new image over an
unmigrated corpus behaves exactly as today, and a half-migrated corpus is fine.

### 2. Verify the image is live before migrating

```bash
# the deployed sha should be the one carrying #1685
gh api repos/chipi/podcast_scraper/commits/main --jq '.sha[0:8]'
```

### 3. Dry-run the migration

```bash
podcast-scraper upgrade plan --corpus /app/output
```

Expect roughly: *"N episodes scanned, M would change (X healed to a full name, Y
episode-scoped)"*. If `healed` is a large fraction, stop — production measured only 12 of 208 as
resolvable, so a big heal count means the roster is being read differently than expected.

### 4. Apply

```bash
podcast-scraper upgrade run --yes --corpus /app/output
```

Idempotent: re-running plans an empty map and writes nothing.

### 5. Rebuild BRIDGES — the step that is easy to miss

m0007 rewrites `.gi.json` and `.kg.json` **only**. `*.bridge.json` also carries `person:` CIL ids,
and the CIL read surfaces (`server/cil_queries.py` — timeline, position-arc, conversation-arc)
walk every bridge file at request time. After migration they will keep serving pre-migration bare
ids that no longer exist in the KG.

**Nothing errors.** The surfaces return stale cross-references silently, which is exactly the
class of failure this whole change set exists to stop. `build_bridge` runs from metadata
generation, so this means a reprocess pass over the corpus — or a deliberate, recorded decision
to accept the staleness.

### 6. Rebuild enrichment and the search index

```bash
make index-two-tier-docker   # host target cannot run on Intel Mac (no x86_64 wheels)
```

**Delete `episode_fingerprints.json` alongside `lance_index`**, or every episode is skipped and
you get a silent empty index. Verify episode coverage afterwards — an empty index also "succeeds".

### 7. Re-run the capability audit and compare

```bash
gh workflow run inspect-prod-corpus.yml -f checks=capability_audit --ref main
```

Needs `prod` environment approval. Expect the bare-name population to collapse toward zero; the
`### Bare person names` section is the direct before/after.

---

## What this sequence does NOT fix

- **Existing follows.** Interests are per-user token lists; m0007 never touches them. A followed
  `person:sam` matches zero episodes afterwards — silently, no error. No tool exists. Single-user
  system, so the agreed answer is to shrug, but it is a real dangling reference.
- **Already-exported PKM vaults.** The export now filters placeholders, but a vault already
  downloaded is outside the system and cannot be migrated.
- **Who "Alex" actually is.** Scoping stops the harm; resolution is #1801.
- **The 180 shared surnames / 420 prefix-overlapping ids.** A different problem (full names
  disagreeing with each other), different machinery.

---

## Rollback

- **Image**: redeploy the previous sha. Safe at any point — an older image over a migrated corpus
  is the bad state described in step 1, so roll the corpus back too if you roll the image back.
- **Corpus**: restore from the step-0 backup. There is no reverse migration; scoped ids carry
  enough information to be promoted later (surface name + episode are in the id, and
  `unresolved_persons_in_episode()` enumerates them), but nothing automates un-scoping.

---

## Related documents, and which to trust

| Document | Status |
|---|---|
| `CORPUS_INTEGRITY_REPAIR_RUNBOOK.md` | **current** — m0007 procedure and the damage table live there |
| `RELEASE_PLAYBOOK.md` | **current** — general release process, not change-specific |
| `PROD_RUNBOOK.md` | **current** — steady-state operations |
| `docs/wip/PROD-CORPUS-REPAIR-PREP-2026-08-17.md` | older prep for the #1657 GI repair; **not** this change |
| `docs/wip/2026-08-19-session-handover.md` | historical; its #1632 line is annotated as superseded |
