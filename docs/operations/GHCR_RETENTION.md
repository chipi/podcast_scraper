# GHCR image retention (#802)

Weekly workflow [`.github/workflows/ghcr-retention.yml`](../../.github/workflows/ghcr-retention.yml)
prunes old container package versions for:

- `podcast-scraper-stack-api`
- `podcast-scraper-stack-viewer`
- `podcast-scraper-stack-pipeline-llm`
- `podcast-scraper-stack-pipeline-ml`

## Keep set

Per image, versions are **kept** when any of the following apply:

| Rule | Rationale |
| --- | --- |
| Latest **20** by `created_at` | ~3 weeks of daily pushes; enough for SHA rollback |
| Tag matches `v*` | Release tags kept indefinitely |
| Tag `main` | Current main HEAD image |
| Tag `sha-<short>` from last **5** successful `deploy-prod.yml` runs | Recent prod deploy history |

Everything else is eligible for deletion.

## Operator workflow

1. **Dry-run (default):** `make ghcr-prune-dry-run` or dispatch workflow with `dry_run=true`.
2. Review stdout / job summary (keep vs delete counts).
3. **Apply:** workflow_dispatch with `dry_run=false` and `confirm_apply=APPLY`.

Cron (Sunday 04:00 UTC) runs **dry-run only** until the operator enables destructive mode on
dispatch.

## Local dry-run

```bash
make ghcr-prune-dry-run
```

Requires `gh` authenticated with `read:packages` and `delete:packages` (delete only when using
`--apply`).
