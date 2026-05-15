# ADR-092: Corpus Snapshot Backup Manifest and Newest-Compatible Restore Default

- **Status**: Accepted
- **Date**: 2026-05-12
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md)
- **Related PRDs**: —
- **Tracking**: [GitHub #763](https://github.com/chipi/podcast_scraper/issues/763)

## Context

Corpus **snapshot** backups are tagged and restored in GitHub Actions. Compatibility between **packed
corpus bytes** and **deployed reader code** is today inferred from dates and human release notes.
That fails for **rollback**, **schema migration**, and **DR** when “latest” is not safe.

**`corpus_manifest.json`** at the corpus parent ([ADR-074](ADR-074-multi-feed-corpus-parent-layout-and-manifest.md))
serves multi-feed **discovery** inside a live tree. It is **not** a substitute for **release
artifact** metadata on **`snapshot.tgz`**.

## Decision

1. Every published corpus snapshot ships **`snapshot.manifest.json`** with at minimum:
   **`schema_version`**, **`corpus_format_version`** (integer, bump **only** on breaking reader-facing
   on-disk or schema changes), **`created_at`**, **`producer`** identity (**`git_sha` and/or
   `image_digest`**), and **`archive.relative_path`** (+ recommended **`archive.sha256`**).
2. **Dual placement**: the manifest exists **inside** the tarball at a documented path **and** as a
   **separate sibling artifact** next to `snapshot.tgz` so consumers can read compatibility without
   downloading the full archive.
3. **Default restore selection** (when the operator does not pin a tag): choose the **newest**
   candidate backup for which **`corpus_format_version`** is **supported** by the deployed reader;
   if **none** match, **fail closed** and require an explicit backup pin / override.
4. **`corpus_format_version` bump rule**: bump **only** when an **older reader** would error,
   mis-read, or corrupt data against the new dump; **not** for non-breaking bug fixes. “Migration
   releases” that read an old format and write a new one still follow this rule: bump when old
   readers **cannot** open **new** dumps.

## Rationale

- **Integer format version** keeps comparisons trivial in Actions and shell; **semver string** adds
  parsing surface without clarifying semantics for corpus bytes.
- **Sibling manifest asset** avoids large downloads for compatibility checks.
- **Newest compatible default** matches operator intent under normal operations; **fail closed**
  prevents silent wrong-version restores during incidents.

## Alternatives Considered

1. **Tags only / no manifest** — Rejected; does not encode reader compatibility
   ([#763](https://github.com/chipi/podcast_scraper/issues/763)).
2. **Manifest only inside tarball** — Rejected as the **only** copy; forces full fetch to inspect
   metadata.
3. **Default to exact producer SHA match** — Rejected for **routine** DR (too strict); may be added
   later as an **optional strict mode**.

## Consequences

- **Positive**: Machine-readable **compatibility**, safer defaults, clearer audit trail (`producer`,
  optional `backup_workflow`).
- **Negative**: Backup and restore workflows must **stay in sync** with schema **v1**; future
  `schema_version` bumps need a small compatibility shim in consumers.
- **Neutral**: Distinct from **`corpus_manifest.json`**; operators must learn two manifests’ roles.

## Implementation Notes

- **Normative detail**: [RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md).
- **Single implementation surface**: repo **scripts** (emit / validate / select / download) plus thin
  **`make` targets**; **GitHub Actions** call those entrypoints so local restore and CI restores stay
  aligned (RFC §5).
- **Workflows** (current tree): `backup-corpus-prod.yml`, `backup-corpus.yml`,
  `drill-restore-corpus.yml`, `prod-restore-corpus.yml` (paths may change; RFC remains source for
  behavior).
- **Code**: landed in `scripts/ops/corpus_snapshot/`, thin **`make`** targets, and the workflows
  above; index **Code** column reads **Yes** ([#763](https://github.com/chipi/podcast_scraper/issues/763)).
- **Cross-surface steady vs recovery playbooks:** [ADR-093](ADR-093-canonical-stack-contract-and-environment-adapters.md),
  [STACK_CONTRACT.md](../guides/STACK_CONTRACT.md).

## References

- [GitHub #763](https://github.com/chipi/podcast_scraper/issues/763)
- [RFC-084: Corpus snapshot backup manifest and version-aware restore](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md)
- [Corpus snapshot manifest and restore (all surfaces)](../guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md)
