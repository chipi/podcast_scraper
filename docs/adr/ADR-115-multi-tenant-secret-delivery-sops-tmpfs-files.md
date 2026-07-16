# ADR-115: Multi-tenant secret delivery — sops/age at rest, tmpfs + file mounts at runtime

- **Status**: Accepted
- **Date**: 2026-07-08
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**: [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (the ownership-split precedent this mirrors), [ADR-011](ADR-011-secure-credential-injection.md) (credential injection — extended to file-based)
- **Security SSOT**: [Threat model](../security/THREAT_MODEL.md) T-08 (secret blast radius) — this ADR is its concrete substance
- **Tracking**: #1160 (hardening); [#1162](https://github.com/chipi/podcast_scraper/issues/1162) (OpenBao future idea)

## Context

Today prod secrets live as a **plaintext `/srv/podcast-scraper/.env`** (0600), rendered
from GH Secrets by `deploy-prod.yml`, and injected into containers as **env vars** — the
`api` then passes all 6 provider keys into every spawned `pipeline` via env. Three
weaknesses (T-08): (1) plaintext at rest on the box → any host-root (or the `docker.sock`
→root path, T-01) reads every key at once; (2) env vars leak via `/proc/<pid>/environ`,
crash dumps, and subprocess inheritance; (3) no rotation/audit/per-service scoping.

The VPS is now multi-tenant (ADR-114: orrery + future podcast-family apps). Secrets must
therefore be a **shared pattern each tenant uses with its own secrets**, not a
podcast-only `.env`. sops/age is already the repo's at-rest tool (TF state,
`terraform.tfstate.enc`).

## Decision

Adopt **B + C**: encrypt secrets at rest with sops/age; deliver them at runtime as
**files in tmpfs**, never as a persistent plaintext file and never as host/compose env.
Structure it as a **multi-tenant contract mirroring the Caddy edge** — infra owns the
mechanism; each tenant brings its own encrypted secrets from its own repo.

### 1. At rest — sops/age, per tenant, in the tenant's own repo

- The VPS has **one age recipient** (keypair). Its **public** key is published by infra;
  its **private** key lives on the box at `/etc/vps-secrets/age.key` (0400 root).
  **Provisioning (decision, 2026-07-08): option (a)** — the operator runs `age-keygen`,
  keeps the private key in the password manager, stages it as a GH secret; `deploy-prod.yml`
  installs it over the **tailnet** (like `.env`). It is **not** baked into cloud-init
  `user_data` (that is retrievable via the Hetzner metadata API). cloud-init only creates
  the `/etc/vps-secrets` dir. The box can decrypt any tenant's secrets; tenants only ever
  need the public key. Mirrors the existing sops/TF-state key flow.
- Each tenant commits `infra/secrets/<env>.enc.yaml` **in its own repo**, sops-encrypted
  to the VPS recipient. Encrypted-at-rest, safe to commit (same posture as
  `terraform.tfstate.enc`).

### 2. At runtime — decrypt to tmpfs, mount as files

- A shared **`decrypt-secrets.sh`** (owned by infra, in cloud-init — same shape as the
  tailscale-serve wrapper) decrypts a tenant's `<env>.enc.yaml` into
  **`/run/secrets/<tenant>/`** — `/run` is tmpfs, so decrypted secrets **never touch
  persistent disk** and vanish on reboot. Files are `0400`, owned by the tenant's runtime uid.
- A **narrow sudoers** entry lets the tenant's `deploy` invoke decrypt for **its own dir
  only** (`/etc/sudoers.d/99-<tenant>-decrypt-secrets`), mirroring the caddy-reload /
  tailscale-serve grants.
- Each tenant's compose mounts those files via **`secrets:`** (file-backed), not `environment:`.

### 3. Consuming the file (two tiers)

- **C1 (delivery, now):** an entrypoint shim reads `/run/secrets/<tenant>/*` and exports
  them for the process — removes the secret from the **host/compose** env and from disk.
  Quick; app code unchanged.
- **C2 (full, with #1161 / ADR-011):** the app reads the secret from the file **at point
  of use** (extend ADR-011's credential injection to a `*_FILE` / file-first source) —
  removes it from the **container** env too. The public consumer plane (#1161) gets
  **zero** provider-key files mounted.

### 4. Ownership split (mirrors ADR-114 §2)

| Piece | Owner | Lives in |
| --- | --- | --- |
| VPS age keypair (private key on box) + published public recipient | infra | this repo cloud-init / staged secret |
| `decrypt-secrets.sh` + `/run/secrets` tmpfs + narrow sudoers | infra | this repo cloud-init |
| **Per-tenant `secrets.enc.yaml`** (sops → VPS recipient) | **tenant** | tenant repo `infra/secrets/` |
| Compose `secrets:` file mounts + app file-reads | tenant | tenant repo |

podcast_scraper is **tenant zero** — it migrates off `.env`-from-GH-Secrets onto this
pattern; the `.env` render is replaced by decrypt-to-tmpfs + file mounts.

## Consequences

**Positive**

- No plaintext secrets at rest; runtime secrets live only in tmpfs.
- Secrets are files, not env → no `/proc/environ` / crash-dump / subprocess-inheritance leak.
- Per-tenant isolation: each tenant's secrets in its own `/run/secrets/<tenant>` (perms),
  encrypted to a key only the box holds; a tenant can rotate independently (re-encrypt + redeploy).
- Reuses sops/age already in the repo — **no new runtime service, no vendor**.

**Negative**

- The VPS age private key is a single high-value decryption key on the box — protect it
  (0400 root; consider TPM/`systemd-creds` wrapping later). Its compromise = all tenants.
- No dynamic/leased creds, no audit log, no automatic rotation — static secrets, manual
  rotation. → the **OpenBao future-idea** ([#1162](https://github.com/chipi/podcast_scraper/issues/1162)):
  a self-hosted, vanilla, LF-governed Vault fork for dynamic secrets + audit, if/when the
  ops cost is justified.
- Migration touches `deploy-prod.yml` (decrypt step replaces `.env` render), the systemd
  unit, compose (`secrets:`), and — for C2 — app credential loading.

**Neutral**

- GH Secrets remains only for bootstrap material (the age key staging), not app secrets.

## Alternatives considered

- **Keep GH-Secrets→`.env`** — centralized, not tenant-friendly (each tenant would need
  access to this repo's secrets); leaves plaintext-at-rest + env exposure. Rejected.
- **OpenBao/Vault now** — proper lifecycle but heavy: another service to run + unseal on a
  single VPS, itself a SPOF + prime target. **Deferred to a future-idea issue**, not now.
- **Managed SaaS (Infisical/Doppler)** — managed rotation/audit but a vendor in the secret
  path (against the vanilla/no-lock ethos) + a bootstrap token still on the box. Rejected as primary.
- **systemd-creds** — vanilla + encrypted, but Hetzner VMs likely lack a vTPM (key on disk
  anyway) and it bridges awkwardly into compose. Kept as a possible wrapper for the age key.
