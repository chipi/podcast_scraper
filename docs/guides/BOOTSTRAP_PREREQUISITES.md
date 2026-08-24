# Bootstrap prerequisites (accounts + credentials)

Self-contained checklist of the one-time **account and credential** work an operator must
complete **before** the first `tofu apply` and first deploy in
[PROD_RUNBOOK § First-time bootstrap](PROD_RUNBOOK.md#first-time-bootstrap). Everything here
lives in this repo so a second operator can bootstrap from the repo alone (the bus-factor
goal of [#805](https://github.com/chipi/podcast_scraper/issues/805)).

> **Tailscale auth model (migrated 2026-08-03):** this project now uses **OAuth clients** for all
> Tailscale auth — they **do not expire**, unlike the old Free-plan Personal tokens. See
> [ADR-143](../adr/ADR-143-tailscale-oauth-migration-and-tag-self-ownership.md) and
> [PROD_RUNBOOK § Tailscale credentials (OAuth clients)](PROD_RUNBOOK.md#tailscale-credentials-oauth-clients)
> for the full architecture. Three OAuth clients (GH Actions secret pairs):
>
> - `TS_INFRA_OAUTH_CLIENT_ID`/`_SECRET` — terraform `tailscale` provider
> - `TS_OAUTH_CLIENT_ID`/`_SECRET` — GHA runner + VPS tailnet join
> - `TS_ACL_OAUTH_CLIENT_ID`/`_SECRET` — ACL GitOps action
>
> **Secret retrieval is password-manager-agnostic.** Commands below show values as
> `<placeholders>`. Retrieve each from whatever secrets store you use (`pass`, a `.env` you keep
> offline, `op read …` if you use 1Password, etc.). No specific manager is required.

---

## A. Operator laptop tooling

- [ ] `brew install opentofu sops age actionlint shellcheck`
- [ ] `gh auth login` (GitHub CLI, authenticated to `chipi/podcast_scraper`)
- [ ] An Ed25519 SSH key at `~/.ssh/id_ed25519` (`ssh-keygen -t ed25519` if absent) — OpenTofu
      registers its public half on the VPS as the operator key.

## B. Hetzner Cloud account

- [ ] Account at [hetzner.com/cloud](https://www.hetzner.com/cloud/) with a billing method
      (EU residents: SEPA is cheapest).
- [ ] A dedicated Cloud Project named `podcast-scraper-prod` (clean billing line + scoped tokens).
- [ ] In that project: **Settings → API Tokens → Generate** with **Read & Write** scope → this is
      `HCLOUD_TOKEN`.
- [ ] Confirm an EU location is available: **Falkenstein (fsn1)** or **Nuremberg (nbg1)**
      (RFC-082 Decision 1).

## C. Tailscale (Free or Premium plan)

- [ ] A tailnet (note its name, e.g. `tail-xxxxx.ts.net`).
- [ ] **Settings → DNS → MagicDNS** enabled.
- [ ] **Settings → DNS → HTTPS Certificates** enabled (so `tailscale cert` / `tailscale serve` can
      issue Let's Encrypt for `prod-podcast.<tailnet>`).
- [ ] **Three OAuth clients** (Admin → Settings → OAuth clients):
  - [ ] **`TS_INFRA_OAUTH`** (terraform provider) — scopes: `auth_keys`, `devices:core`, `dns`, `policy_file`; tags: `tag:prod`, `tag:dr-drill`
  - [ ] **`TS_OAUTH`** (device join) — scope: `auth_keys`; tag: `tag:gha-deployer`
  - [ ] **`TS_ACL_OAUTH`** (GitOps action) — scope: `policy_file`
- [ ] **Access Controls** (`policy.hujson`):
  - [ ] Owners for `tag:prod` and `tag:dr-drill` (self-owned, per ADR-143)
  - [ ] Owners for `tag:gha-deployer`
  - [ ] Rule `tag:gha-deployer` → `tag:prod:22` and `tag:dr-drill:22` (SSH)
  - [ ] Rule operator's user → `tag:prod:443`, `:80`, `:22`; same for `tag:dr-drill`

## D. sops + age (Terraform state encryption)

- [ ] `age-keygen -o ~/.config/sops/age/keys.txt`.
- [ ] Copy the **public** key into `infra/.sops.yaml` (replacing the `age1PLACEHOLDER…` value;
      commit-safe).
- [ ] Save the **private** key contents to your secrets store as
      `sops/podcast-scraper/tofu-state-age-key`.
- [ ] Verify round-trip: `echo test | sops -e --age "$(grep 'public key:' ~/.config/sops/age/keys.txt | sed 's/.*: //')" /dev/stdin >/dev/null`.

## E. Runtime service accounts (host `.env`, staged later as `PROD_*` secrets)

These are needed for the running stack, staged once in repo settings and rendered into the host
`.env` by `deploy-prod.yml` (see [PROD_RUNBOOK § Stage `.env`](PROD_RUNBOOK.md#stage-env-workflow-staged-from-gh-secrets-841)).
Stage incrementally — a missing one just disables that feature (the stack still starts).

- [ ] **LLM providers** (whichever your prod profile uses; `cloud_balanced` default uses OpenAI +
      Gemini): OpenAI, Anthropic, Gemini, Mistral, DeepSeek, Grok API keys.
- [ ] **GlitchTip (self-hosted)**: homelab `http://homelab:8090` → create **three** projects
      (`api`, `pipeline`, `viewer`) → one DSN each. Public error ingest via `telemetry.closelistening.app`.
- [ ] **Alloy remote-write endpoints** (no signup required): `REMOTE_WRITE_URL=https://vm.<tailnet>.ts.net/api/v1/write`
      and `LOGS_WRITE_URL=https://vlogs.<tailnet>.ts.net/insert/loki/api/v1/push` — homelab VictoriaMetrics/Logs
      fronted by per-service caddy-tailscale TLS nodes (Level 3, #1665), tailnet-reachable from the VPS.
      GitOps'd via `deploy-vps-observability-endpoints.yml`, which derives the tailnet suffix from
      `vars.PROD_TAILNET_FQDN` at deploy time (never committed).
- [ ] **Backup repo PAT** (`BACKUP_REPO_TOKEN`, optional) and outbound **job webhook URL** (optional).

## F. Stage the infrastructure GHA secrets

With the above in hand, stage the **infra** credentials (the runtime `PROD_*` secrets are staged
separately per the runbook):

```bash
gh secret set HCLOUD_TOKEN              --repo chipi/podcast_scraper --app actions --body '<from §B>'
gh secret set TS_INFRA_OAUTH_CLIENT_ID  --repo chipi/podcast_scraper --app actions --body '<client-id from §C TS_INFRA_OAUTH>'
gh secret set TS_INFRA_OAUTH_SECRET     --repo chipi/podcast_scraper --app actions --body '<client-secret from §C TS_INFRA_OAUTH>'
gh secret set TS_OAUTH_CLIENT_ID        --repo chipi/podcast_scraper --app actions --body '<client-id from §C TS_OAUTH>'
gh secret set TS_OAUTH_SECRET           --repo chipi/podcast_scraper --app actions --body '<client-secret from §C TS_OAUTH>'
gh secret set TS_ACL_OAUTH_CLIENT_ID    --repo chipi/podcast_scraper --app actions --body '<client-id from §C TS_ACL_OAUTH>'
gh secret set TS_ACL_OAUTH_SECRET       --repo chipi/podcast_scraper --app actions --body '<client-secret from §C TS_ACL_OAUTH>'
gh secret set TFSTATE_AGE_KEY           --repo chipi/podcast_scraper --app actions --body "$(cat ~/.config/sops/age/keys.txt)"
gh secret set BACKUP_REPO_TOKEN         --repo chipi/podcast_scraper --app actions --body '<backup-repo-pat, optional>'
gh secret set OPERATOR_SSH_PUBLIC_KEY   --repo chipi/podcast_scraper --body "$(cat ~/.ssh/id_ed25519.pub)"
gh variable set TAILNET_NAME            --repo chipi/podcast_scraper --body 'tail-xxxxx.ts.net'
```

`PROD_SSH_PRIVATE_KEY` (the CI deploy key) and `PROD_TAILNET_FQDN` are set during bootstrap, not
here — see [PROD_RUNBOOK § GitHub Actions SSH to prod](PROD_RUNBOOK.md#github-actions-ssh-to-prod-prod_ssh_private_key)
and [§ First `tofu apply`](PROD_RUNBOOK.md#first-tofu-apply-operators-laptop).

## G. Exit criterion (smoke before `tofu apply`)

- [ ] Hetzner API token works:

  ```bash
  HCLOUD_TOKEN='<from §B>'
  curl -fsS -H "Authorization: Bearer $HCLOUD_TOKEN" https://api.hetzner.cloud/v1/server_types \
    | jq '.server_types[] | select(.name=="cx33") | {name, prices: .prices[0]}'
  # Expect a JSON object (NOT 401/403).
  ```

- [ ] `gh secret list --repo chipi/podcast_scraper` shows the §F secrets.
- [ ] Your secrets store holds: Hetzner token, three OAuth client pairs (from §C), sops-age private key.

Once all boxes are checked, proceed to
[PROD_RUNBOOK § First-time bootstrap](PROD_RUNBOOK.md#first-time-bootstrap).
