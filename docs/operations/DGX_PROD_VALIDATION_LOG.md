# DGX prod validation log (ADR-096)

Record pre-prod runs of `cloud_with_dgx_whisper_primary` before flipping prod default.

| Week ending | Environment | Fallback rate | Notes |
| --- | --- | --- | --- |
| (pending) | pre-prod | n/a | DGX hardware not yet on tailnet |

Target: under 1% fallback sustained for 4 weeks; spot-check transcript quality vs `cloud_balanced`.
