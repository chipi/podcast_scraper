# `tailscale/` — tailnet ACL as code

Source-of-truth for the operator's Tailscale ACL. Synced to the live tailnet
by [infra/terraform/tailscale.tf](../infra/terraform/tailscale.tf)'s
`tailscale_acl` resource on every `tofu apply`. Implements
[RFC-082 Decision 2](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md).

## Files

- [`policy.hujson`](policy.hujson) — the tailnet ACL in HuJSON
  (JSON with `//` comments + trailing-comma tolerance).

## How a change ships

1. Edit `policy.hujson` on a branch.
2. Open a PR. The infra-CI workflow (#719) runs `tofu plan` and posts the diff
   as a comment so you can preview which devices/rules change.
3. Merge to main.
4. Operator (or scheduled `infra-apply.yml`) runs `tofu apply`. The
   `tailscale_acl` resource pushes the new policy to the tailnet API.

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

- [RFC-082 — Decision 2: Tailscale](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [#717](https://github.com/chipi/podcast_scraper/issues/717) — this ticket
- [#714](https://github.com/chipi/podcast_scraper/issues/714) — Tailscale OAuth client + tag prereqs
- [Tailscale ACL syntax (HuJSON)](https://tailscale.com/kb/1018/acls)
