# Repository settings audit (operator)

Periodic manual checks. GitHub does not expose all of these via API.

## ADR-097 — self-hosted runner safety

- [ ] **Require approval for all outside collaborators** is enabled (Settings, Collaborators and teams).
- [ ] `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md` matches every workflow using `self-hosted` or `dgx-spark`.
- [ ] DGX runner registered as **ephemeral** only.
- [ ] No deploy/release/security workflow targets self-hosted.

## Related

- [ADR-097](../adr/ADR-097-self-hosted-gha-runner-policy.md)
- [DGX_RUNBOOK](../guides/DGX_RUNBOOK.md)
