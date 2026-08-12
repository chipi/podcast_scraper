# `tailscale/` — tailnet ACL as code

Source-of-truth for the operator's Tailscale ACL. Synced to the live tailnet by
[Tailscale's native GitOps action](../.github/workflows/tailscale-acl.yml).
Implements
[RFC-082 Decision 2](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md).

> **Not OpenTofu.** Per [ADR-128](../docs/adr/ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md)
> (2026-07-28) the ACL moved out of OpenTofu so a one-line grant no longer forces
> a full-estate apply. The `tailscale_acl` resource was removed from tofu state
> (`tofu state rm`) and config; only `tailscale_tailnet_key.prod` still lives in
> [infra/terraform/tailscale.tf](../infra/terraform/tailscale.tf). **Editing
> `policy.hujson` and running `tofu apply` does nothing.**

## Files

- [`policy.hujson`](policy.hujson) — the tailnet ACL in HuJSON
  (JSON with `//` comments + trailing-comma tolerance).

## How a change ships

Driven entirely by [`.github/workflows/tailscale-acl.yml`](../.github/workflows/tailscale-acl.yml),
which triggers only on changes to `tailscale/policy.hujson`:

1. Edit `policy.hujson` on a branch.
2. Open a PR → the workflow runs in **`test`** mode: ACL tests plus a dry-run
   diff. Nothing is pushed to the tailnet.
3. Review + merge to main (branch protection is the gate).
4. The push to main re-triggers the workflow in **`apply`** mode, which syncs
   the policy to the live tailnet. No manual step — merging *is* the apply.

`workflow_dispatch` runs it on demand with a `test`/`apply` choice (default
`test`).

### Credentials

The action authenticates with a dedicated OAuth client holding the `acl`
(policy-write) scope — `secrets.TS_ACL_OAUTH_CLIENT_ID` /
`secrets.TS_ACL_OAUTH_SECRET`, plus `vars.TAILNET_NAME`. This is a separate
client from the `tag:gha-deployer` join client, which has `auth_keys` scope and
cannot write ACLs. OAuth clients don't expire; the previous `TS_API_KEY` lapsed
on 2026-08-03 (HTTP 401) and blocked every ACL sync until it was replaced.

If any of the three are absent — a fork PR, for instance — the workflow emits a
warning and skips rather than failing.

## Local validation

The file is HuJSON, not strict JSON. A quick syntax check (strip `//` comments,
parse remainder as JSON) catches most mistakes:

```bash
python3 -c 'import json, re; t = open("tailscale/policy.hujson").read(); json.loads(re.sub(r"//.*", "", t))'
```

For a deeper check (resolves tag references, validates against Tailscale's
schema), use the Tailscale CLI: `tailscale debug check-policy-file policy.hujson`
— requires Tailscale login.

## Cross-references

- [ADR-128 — decouple the tailnet ACL from the Hetzner tofu stack](../docs/adr/ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md)
  (why this is a GitOps action and not `tofu apply`)
- [ADR-143 — Tailscale OAuth migration + tag self-ownership](../docs/adr/ADR-143-tailscale-oauth-migration-and-tag-self-ownership.md)
- [RFC-082 — Decision 2: Tailscale](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [#717](https://github.com/chipi/podcast_scraper/issues/717) — this ticket
- [#714](https://github.com/chipi/podcast_scraper/issues/714) — Tailscale OAuth client + tag prereqs
- [Tailscale ACL syntax (HuJSON)](https://tailscale.com/kb/1018/acls)
