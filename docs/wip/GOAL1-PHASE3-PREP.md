# Phase 3 prep — pre-public gate pre-walk + ADR-115 secrets cutover steps

Prep for the Goal-1 hard gate (Phase 3). Read-only analysis + a step list so 3.1
and 3.4 are rubber-stamps at execution time. Nothing here touches prod.

## 3.1 — Pre-public gate pre-walk (first tenant = orrery)

The [THREAT_MODEL pre-public gate](../security/THREAT_MODEL.md#pre-public-gate-run-before-any-new-public-vhost)
has 8 items. Mapped to current state for **orrery** (the first public vhost):

| # | Gate item (T-ref) | Orrery | Evidence / what's left |
|---|---|---|---|
| 1 | No `docker.sock`, no write-scope cloud keys in the public container (T-01) | ✅ | orrery is a static site — no sock, no keys. The **api** carries the sock+keys and is NEVER given a vhost; only orrery is exposed. |
| 2 | `cap_drop: ALL`, `no-new-privileges`, non-root, RO rootfs (T-01/T-04) | ⏳ | **orrery-repo-owned** — verify orrery's compose hardening before 3.2 sign-off. |
| 3 | API needs authN/authZ, CORS, rate-limit (T-06) | ✅ N/A | orrery serves static content, no API surface. |
| 4 | Caddy `admin off`; on-demand TLS off; catch-all denies unknown Host (T-02/T-03) | ✅ | base Caddyfile — `admin off`, on-demand off (default), `:443 { tls internal; abort }`. Validated with a real caddy binary + fmt-clean. Dropped by `apply-edge.sh §4`. |
| 5 | Metadata egress (169.254.169.254) blocked from the tenant (T-07/T-10) | ✅ | `apply-edge.sh §3` metadata-egress DROP; `verify-edge.sh §3` asserts it. Real-run-tested (DOCKER-USER DROP active). |
| 6 | Deployed image digest-pinned / cosign-verified (T-09) | ⏳ | **orrery deploy-side** — confirm orrery's image ref is a digest, not a floating tag. |
| 7 | Security alerting covers the new surface (T-11) | ✅ | 3 common rules (`config/grafana/alerts/common/security.yaml`), obs-sync-validated; fire once the box + log sources are live. Per-app alerts deferred to Goals 2/3. |
| 8 | Rollback proven: pull vhost + reload → public down, tailnet up (ADR-114 §7) | ⏳ | rehearse in Phase 5.2: `rm /etc/caddy/sites/orrery.caddy && systemctl reload caddy`. |

**Substrate items (4, 5, 7) are satisfied by the edge tooling.** The 3 open items
(2, 6, 8) are orrery-side / a Phase-5 rehearsal — not this repo's edge work. api
items (1, 3) are moot because the api is never exposed.

## 3.4 — ADR-115 secrets cutover steps (T-08 / M26)

**Today:** `deploy-prod.yml` renders `/srv/podcast-scraper/.env` from `PROD_*` GH
Secrets — the 6 LLM keys + Sentry DSNs are **plaintext container env**
(`docker-compose.prod.yml`). **Target:** file-mounted secrets via
`docker-compose.secrets.yml` + the baked-in secret shim (`docker/secrets-shim.sh`,
already in the api + pipeline images, boot-verified).

Remaining steps — **cutover-gate ordered** (the overlay errors at `up` if the
decrypted files are absent, so order matters):

1. 🧑 Generate the VPS **age keypair**; keep the private key in the password
   manager; publish the public recipient into `.sops.yaml`.
2. 🧑 Fill `infra/secrets/prod.enc.yaml` from `prod.yaml.template` (6 provider keys
   + 2 Sentry DSNs), sops/age-**encrypt**, and commit it (encrypted at rest).
3. 🤝 `deploy-prod.yml`: stage the age **private** key to `/etc/vps-secrets` on the
   box over the tailnet from a GH secret (NOT in user_data — retrievable via the
   metadata API; the metadata-egress guard is defense-in-depth, not the primary).
4. 🤝 `deploy-prod.yml`: run `decrypt-secrets.sh podcast .../prod.enc.yaml` →
   decrypts to `/run/secrets/podcast/*` (tmpfs, 0444-in-0700).
5. 🤝 **Only then** add `-f compose/docker-compose.secrets.yml` to the deploy `-f`
   chain (this is the cutover gate — do not add it before step 4 succeeds).
6. 🤝 Drop the 6 keys + DSNs from the `.env` render (the shim now exports them from
   the files; the empty env lines in `docker-compose.prod.yml` stay, overridden).
7. 🤝 Verify: `/api/health` 200, keys resolve from files, no plaintext keys in
   `/srv/podcast-scraper/.env`.

**Scope note:** 3.4 blocks *public exposure of the api*. The api is **not** exposed
in Goal-1 (only orrery), so this is a hardening step gated ahead of any future
api exposure (Goal-2+), not a blocker for orrery going live.
