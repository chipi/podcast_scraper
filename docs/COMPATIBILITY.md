# Code/content compatibility matrix

Forward and backward compatibility between **running code** (GHCR image / git tag) and **on-disk corpus**
(artifacts produced by some prior pipeline run). Update this file on every release.

Related: [GitHub #796](https://github.com/chipi/podcast_scraper/issues/796) (system contract:
`produced_by`, `/api/health` preflight), [GitHub #797](https://github.com/chipi/podcast_scraper/issues/797)
(operator framework + smoke script), [PROD_RUNBOOK — Code/content compatibility](guides/PROD_RUNBOOK.md#codecontent-compatibility),
[Corpus artifacts and viewer surfaces](architecture/CORPUS_ARTIFACTS_AND_SURFACES.md).

| Code version | Min corpus version it can read | Max corpus version it can read | Notes |
| --- | --- | --- | --- |
| 2.6.1 | 2.4.0 | 2.6.x | Operator compatibility framework (#797): PROD_RUNBOOK section, `CORPUS_ARTIFACTS_AND_SURFACES.md`, `post_deploy_smoke.sh`, this matrix. No new required artifact fields. |
| 2.6.0 | 2.4.0 | 2.6.x | First release with corpus-level `produced_by` stamp (#796). GIL 2.0 / KG 1.2 read via migration helpers. Rollback to 2.5.x still supported when corpus was not mutated by 2.6-only writers. |
| 2.5.0 | 2.4.0 | 2.6.0 | `corpus_manifest` schema 1.1 (`cost_rollup`, PR #650). Older manifests still load with defaults. |
| 2.4.0 | 2.0.0 | 2.5.x | RFC-072 canonical GI/KG identity (GIL 2.0, KG 1.2 finalized). |

## How to update on release

1. Add a row for the version you are tagging.
2. State **min** corpus semver the new code still reads cleanly.
3. State **max** corpus semver you have tested (usually the same minor line).
4. Note rollback window: which prior code tag remains safe after this corpus may have been touched.
5. Run `make pre-release` — it fails if the current `pyproject.toml` version is missing from this file.
