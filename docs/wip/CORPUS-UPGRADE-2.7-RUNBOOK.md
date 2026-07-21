# Corpus upgrade runbook — 2.6.x → 2.7 (schema/ontology migration)

Deploying 2.7 requires a **corpus migration** (RFC-090 native reindex + RFC-097 GI
schema 2.0→3.0). The migration is **in-place** and **stops at the first failing
step**, so a bad migration can leave a partially-migrated corpus. This runbook
proves it on real data first, then applies it to prod with a rollback net.

## The migrations (2.6.x → 2.7)
- `0001_faiss_to_lance` — historical, now a **no-op** (FAISS retired).
- `0002_two_tier_native_reindex` — builds the native two-tier index (RFC-090).
- `0003_gi_v3_typed_mentions` — rewrites GI docs to **schema 3.0** (typed
  MENTIONS_PERSON/ORG, claim/observation), bumps `schema_version` 2.0→3.0 (RFC-097).

## Phase A — PRACTICE (no prod touch): the drill
Run **`drill-corpus-upgrade.yml`** (`gh workflow run drill-corpus-upgrade.yml`). It
restores the real prod backup into a throwaway CI runner, runs the exact upgrade in
the published image, and asserts **no data loss** (episode count before == after) +
`upgrade verify` + a 2.7 smoke. **Green = the migration is proven on real prod data.**
Do not proceed to prod until it's green.

## Phase B — PROD (only after the drill is green)
1. **Fresh backup + Hetzner snapshot** — the external rollback point. (Corpus static
   → the last `snapshot-prod-*` is valid; snapshot the box before touching it.)
2. **Deploy 2.7** (`deploy-prod`, `PROD_DEPLOY`). App boots against the un-migrated
   corpus (2.7 supports corpus code ≥ 2.6.0).
3. **Preview**: `upgrade run --dry-run --corpus-dir <corpus>` → eyeball the plan.
4. **Apply with a local snapshot**:
   ```bash
   # run inside the app container; point --snapshot-dir at a PERSISTENT host-mounted
   # path (a sibling of the corpus volume is container-ephemeral).
   upgrade run --yes --corpus-dir /app/output --snapshot-dir /app/output/../.upgrade-snapshots
   ```
   The `--snapshot-dir` copy is written **before** any mutation → instant local
   rollback. (Must be OUTSIDE the corpus root — the CLI refuses an inside path.)
5. **Verify**: `upgrade verify` + `upgrade status` (up to date) + a smoke (search +
   a GI query).

## Rollback
- **Partial/failed upgrade** → restore from the **`--snapshot-dir` copy** (replace the
  corpus with it), or the fresh external backup; redeploy the 2.6.1 image if needed.
- The migrations are **idempotent** (safe to re-run) once the cause is fixed.

## Reusable
The drill + `--snapshot-dir` are permanent: run the drill before **any** future major
version whose migration touches the corpus (this is the "prove the migration on real
data before prod" gate).
