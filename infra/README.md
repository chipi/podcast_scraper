# `infra/` — production VPS infrastructure-as-code

OpenTofu configuration that provisions the always-on production Hetzner VPS,
registers it with the operator's Tailscale tailnet, and prepares it to run the
podcast_scraper stack. Implements [RFC-082](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md).

## Layout

```text
infra/
├── .sops.yaml                  # age public-key config for state encryption
├── .gitignore                  # excludes plaintext state, plans, .terraform/
├── tofu                        # wrapper script — handles sops decrypt/encrypt
├── README.md                   # this file
├── terraform/
│   ├── *.tf                    # OpenTofu config (server, network, Tailscale)
│   └── terraform.tfstate.enc   # sops+age-encrypted state (committed; appears
│                               # after first apply)
├── cloud-init/
│   └── prod.user-data          # first-boot bootstrap (Docker, Tailscale, repo pull)
└── deploy/
    └── deploy.sh               # called by .github/workflows/deploy-prod.yml
```

## Quickstart

> **Full operator runbook:** see [PROD_RUNBOOK.md](../docs/guides/PROD_RUNBOOK.md).
> Account-prerequisite checklist: [#714](https://github.com/chipi/podcast_scraper/issues/714).

One-time setup (per operator):

1. `brew install opentofu sops age`
2. `age-keygen -o ~/.config/sops/age/keys.txt`
3. Copy the public key (the `# public key:` comment) into [`.sops.yaml`](.sops.yaml), replacing the `age1PLACEHOLDER…` value.
4. Save the **private** key to 1Password as `sops/podcast-scraper/tofu-state-age-key`.
5. Stage `HCLOUD_TOKEN`, `TS_AUTHKEY` (device-join), `TS_API_KEY` (terraform's tailscale provider; Free-plan substitute for OAuth — see [PROD_RUNBOOK.md "Tailscale credentials"](../docs/guides/PROD_RUNBOOK.md)), `TFSTATE_AGE_KEY` in repo Actions secrets (see [#714](https://github.com/chipi/podcast_scraper/issues/714)).

Per-operation:

```bash
cd infra
export HCLOUD_TOKEN=$(op read 'op://Personal/Hetzner Cloud/podcast-scraper-prod/api-token')
export TF_VAR_tailscale_api_key=$(op read 'op://Personal/Tailscale/podcast-scraper/api-key')
./tofu init
./tofu plan
./tofu apply
```

The `tofu` wrapper auto-decrypts state before invocation and re-encrypts after.

## State encryption model

State lives at `terraform/terraform.tfstate.enc` (sops+age, committed). Plaintext
**never** touches the repo or shell history. The wrapper:

1. Decrypts `.enc` → `terraform/terraform.tfstate` (plaintext, for tofu only).
2. Runs the requested subcommand against that plaintext file.
3. Re-encrypts back to `.enc` if state changed (sha256 comparison).
4. Shreds the plaintext on any exit (trap, including Ctrl-C).

If the wrapper crashes mid-flight, check `git status infra/terraform/` and
confirm no plaintext `.tfstate` is staged. If it is, **do NOT commit** — re-run
`cd infra && ./tofu version` (a no-op invocation) to re-trigger the trap and
clean up.

## Cross-references

- [RFC-082 — Decision 5: IaC](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [PROD_RUNBOOK.md](../docs/guides/PROD_RUNBOOK.md) — full operator runbook
- [#716](https://github.com/chipi/podcast_scraper/issues/716) — this scaffolding ticket
- [#719](https://github.com/chipi/podcast_scraper/issues/719) — `infra-ci.yml` (PR plan) and `infra-apply.yml` (manual apply) workflows
