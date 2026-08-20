# Threat model — production VPS + shared public edge

- **Status**: Living (single source of truth — update on every infra/security change)
- **Owner**: Marko Dragoljevic
- **Last reviewed**: 2026-08-06 (T-13 remote MCP — a new public tenant/vhost)
- **Review cadence**: on every change to `infra/**`, `compose/*.prod*.yml`, a new
  public vhost/tenant, a new dependency in cloud-init, or a provider/key change —
  and at minimum once per quarter.

This is the security source of truth we work against. [ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md)
records the *decision* to run a shared public edge; this document tracks the *risk*
of doing so and the controls that keep it acceptable. When the two disagree, one is
wrong — reconcile, do not paper over.

Related: [ADR-083](../adr/ADR-083-tailscale-private-ingress-always-on-vps.md) (tailnet
admin plane), orrery **ADR-078** (first public tenant), [PROD_RUNBOOK](../guides/PROD_RUNBOOK.md),
incident [2026-05-29 prod-rebuild cascade](../incidents/INCIDENT-2026-05-29-prod-rebuild-cascade.md).
Tracking: hardening [#1160](https://github.com/chipi/podcast_scraper/issues/1160) (execution plan: `docs/wip/INFRA-HARDENING-PLAN.md`); shared edge [#1158](https://github.com/chipi/podcast_scraper/issues/1158) / [chipi/orrery#381](https://github.com/chipi/orrery/issues/381).

## Assets we protect (in priority order)

1. **Cloud LLM API keys** — OpenAI / Gemini / Anthropic / Mistral / DeepSeek / Grok.
   Live in `/srv/podcast-scraper/.env` and are injected into the `api` container env.
   Theft = direct financial loss + reputational abuse.
2. **The corpus + derivatives** — transcripts, GI/KG, embeddings. Ours to host
   (audio is bridge-only). Exfil or tamper is a data-integrity and IP loss.
3. **Host root on the VPS** — controls every co-tenant, all secrets, all data.
4. **Tenant availability + integrity** — orrery, gi/kg viewer, podcast player API.
5. **Terraform state** — sops/age-encrypted; grants full infra control if decrypted.

## Trust boundaries

- **Admin/deploy plane (private):** Tailscale only. SSH is tailnet-only; the Hetzner
  firewall opens `:22` only for explicit troubleshoot CIDRs (`infra/terraform/main.tf`).
  Funnel is rejected (ADR-078). **Unchanged by the public edge.**
- **Public plane (new):** Hetzner firewall opens `80 + 443` to `0.0.0.0/0`; a single
  host-level Caddy terminates TLS and routes by `Host` to per-tenant containers on
  loopback. This is the new, internet-facing boundary.
- **Operator *read* plane (public, gated — [RFC-108](../rfc/RFC-108-operator-public-gated-surface.md)):**
  `operator.closelistening.app` serves the gi-kg-viewer + a **separate least-privilege**
  backend (`PODCAST_SERVE_OPERATOR_PUBLIC`: a curated read-only subset of the operator
  routes, each **≥creator-gated**; **no `docker.sock`, no keys**), behind the coming-soon
  cookie gate → Google OAuth → admin/creator role → CF origin-lock + rate-limit. The
  **privileged** operator api (`docker.sock` + LLM keys + `index_rebuild`/`ops`/pipeline
  triggers) **stays tailnet-only** — so **T-01's D1 basis is unchanged**. This is a *new
  public tenant that passes the pre-public gate by construction*, not an exposure of the
  private operator plane.
- **Tenant boundary (weak today):** all tenants share one host, one Docker daemon, one
  `deploy` user, one `.env`. A container escape or app RCE crosses into every tenant.

## Attacker profiles

| Profile | Capability | Primary goal |
| --- | --- | --- |
| Opportunistic scanner | Mass internet scans, known-CVE probes, brute force | Any foothold, cryptomining, botnet |
| Targeted attacker | Recon of exposed apps, chained app→host exploits | LLM keys, corpus, host |
| Malicious/compromised tenant or deploy key | Can drop a `.caddy` vhost + reload Caddy | SSRF to internal/metadata, host pivot, serve content on any host |
| Supply-chain | Compromised GHCR image tag, Caddy apt repo, or a dependency | Code execution on the VPS at deploy/build time |

## Threat register

Severity: 🔴 critical · 🟠 high · 🟡 medium · ⚪ low. Status: **open** / **mitigated** /
**accepted**. Update the status column as controls land.

| ID | Threat | Sev | Status | Control / required fix |
| --- | --- | --- | --- | --- |
| T-01 | `api` container mounts `/var/run/docker.sock` **RW** and holds all 6 LLM keys (`compose/docker-compose.prod.yml:80,155`) → any RCE/SSRF in a public api = **host root + key theft + all tenants** | 🔴 | **in progress** | **D1 (2026-07-08): accept RW socket** on the private api — socket-proxy is low-ROI + a new dep (`docker compose run` needs container-create anyway). **Conditional on the api being provably private → enforced public/private API separation tracked in #1161** (today "private" is circumstantial, not architectural). Container hardening done (T-04). Public consumer plane = separate least-privilege service (no `docker.sock`, no keys). |
| T-02 | Caddy **admin API** (`:2019`) on by default → full-config control / RCE if it ever binds non-loopback | 🔴 | **mitigated (on rebuild)** | `admin off` in the base Caddyfile (`infra/cloud-init/Caddyfile`, #1158). `caddy validate` confirms. Applies on rebuild; live box gets the engine via imperative-once. |
| T-03 | **On-demand TLS / catch-all Host** → attacker points any hostname at the IP, exhausts Let's Encrypt rate limits, or gets a default vhost | 🟠 | **mitigated (on rebuild)** | On-demand TLS left OFF (default); explicit `:443` catch-all `tls internal` + `abort` for unknown Host (validated). Same rebuild/imperative caveat. |
| T-04 | **Shared-everything tenancy** — one host/daemon/`deploy` user/`.env`; one compromise owns all | 🟠 | **mitigated + accepted residue** | Container hardening **done** (boot-verified: `cap_drop`/`no-new-privileges`/`read_only` on prod services, `a7f49cec`). **D3 (2026-07-08): accept shared tenancy** at current scale — per-app users are low-ROI vs. the shared daemon. Move orrery to its own VPS when budget allows (clean drop-out per ADR-114 §7); tracked as a future option, not a blocker. |
| T-05 | **No DDoS / WAF / edge rate-limit** — single small VPS; `fail2ban` is weak L7 | 🟠 | **in progress** | Stopgap landed: `caddy-access` fail2ban jail bans an IP on a 4xx burst (30 / 2m → 1h) — kills scanners/probers/429-abuse but NOT volumetric DDoS. **D4 (2026-07-09): accept Cloudflare free as the additive front — [ADR-118](../adr/ADR-118-cloudflare-additive-front.md).** Origin prep landed: Caddy `trusted_proxies` for CF real-IP, fail2ban keyed on `client_ip` (CF-safe), optional `cloudflare_origin_lock` TF firewall (443→CF ranges only). Operator does CF account/DNS + flips the lock per the ADR rollout. |
| T-06 | **Public API authN/authZ, CORS, input validation** — unauth API = open scraping + injection; per-user data needs authZ | 🟠 | **partial** | Player: OAuth authN (RFC-098) + **rate limiting done** — nginx `limit_req` per real client IP (`real_ip` from XFF), tighter zone on auth endpoints, `429` on excess (#1163). Still open: CORS allowlist for future mobile/native origins (web player is same-origin); per-token limits + input-validation review for the kg/gi API (#1166). |
| T-07 | `deploy` can write arbitrary `<app>.caddy` + reload → **SSRF** to internal `10.0.1.x` or cloud metadata `169.254.169.254`, or serve attacker content on any host | 🟠 | **partial** | Metadata-egress DROP landed in cloud-init (`block-metadata-egress.service`, #1160 Phase 2) — **applies on rebuild; live box pending imperative apply**. Caddy writable-sites-dir constraint + vhost review + validate-before-reload land with the edge (ADR-114 §5). |
| T-08 | **Secret blast radius** — 6 keys in one plaintext `.env` (env-injected), no billing caps | 🟡 | **in progress** | **[ADR-115](../adr/ADR-115-multi-tenant-secret-delivery-sops-tmpfs-files.md)** (accepted direction B+C): sops/age at rest, decrypt to tmpfs, file mounts not env, per-tenant + least-key. OpenBao (dynamic creds/audit/rotation) deferred to **#1162**. Operator-side: provider billing caps + per-env keys (D5). **Addendum (#1787, 2026-08-20):** the audio-archive rclone creds (`RCLONE_CONFIG_AUDIOARCHIVE_*`, incl. the rclone-*obscured* — reversible, not encrypted — SFTP pass) are staged into `.env` **unconditionally**, outside the ADR-115 tmpfs-file path, because rclone reads its config from env and the nested `docker compose run pipeline-llm` must inherit it. These creds are **delete-capable against the archive's only long-term copy** (eviction is on), so a `.env` leak is higher-impact than a read-only provider key. Blast radius is contained: the pipeline uses a **Hetzner Storage Box sub-account** (`hcloud_storage_box_subaccount.audio_archive`, applied 2026-08-20 as `u653903-sub1`) **jailed to `podcast-audio-archive/`** — verified it cannot traverse above the jail (escape → `permission denied`), and the account-level snapshots sit outside its SFTP view, so even a full wipe within the jail leaves the archive recoverable. A `.env` leak therefore exposes one directory + no snapshot-delete, not the whole box. Residual (accepted): the creds are still env-delivered plaintext (obscured pass) rather than ADR-115 tmpfs files. |
| T-09 | **Supply chain** — GHCR pulled by mutable tag (`:main`, `compose/docker-compose.prod.yml:66`); new Caddy apt repo via `curl\|gpg` in cloud-init | 🟡 | **accepted** | **D2 (2026-07-08): keep `:sha-<7>`** — prod already deploys the sha tag (not `:main`) + validates the manifest pre-deploy. Digest-pin deferred (deploy-workflow change, only verifiable on a live run). Still pin the Caddy apt version when the edge lands. |
| T-10 | **Unrestricted egress** — a compromised container can exfiltrate the corpus or reach C2/metadata | 🟡 | **partial** | Metadata IP blocked (see T-07, cloud-init, rebuild/imperative). Full egress allowlist DEFERRED (design) — a blanket allowlist on api/pipeline risks breaking GHCR/provider/PyPI/HF pulls. |
| T-11 | **Detection gap** — metrics/logs exist (Grafana, Sentry) but no *security* alerting | 🟡 | **partial** | **[ADR-117](../adr/ADR-117-multi-tenant-observability-gitops.md)** common plane built: Alloy security-log pipeline (sshd/fail2ban/Caddy → Loki, `tenant=common`) in cloud-init + common alert rules (`config/grafana/alerts/common/security.yaml`) synced via `make obs-sync`. **Applies on rebuild / live box; rules fire once the box + log sources are live.** Per-tenant app-security (auth-fail spikes, per-app 4xx/5xx + egress anomalies) deferred to Goals 2/3 — tracked under #1160 (ADR-117 §Sequencing). |
| T-12 | **TLS/headers hygiene** — HSTS ramp, CSP coverage on public sites, min TLS 1.2 | ⚪ | **partial** | Shared `(hardened)` snippet (HSTS `max-age=86400` + compression + access log) in the base Caddyfile; tenants `import hardened` (validated). Ramp HSTS after first live vhost; CSP owned per-app nginx; Caddy TLS defaults strong. |
| T-13 | **Remote MCP server — a NEW public surface for external AI agents** (RFC-112, its own vhost behind the edge). Attack surface: (a) **bearer-token theft** (OAuth access token or PAT) → read the holder's full corpus as them; (b) **over-broad grant** — `mcp_access` unlocks the **entire shared read surface** (all 38 tools, incl. UI-creator-gated perspectives), so any entitled account is a full-corpus read key; (c) **OAuth flow abuse** — open-redirect via an unregistered `redirect_uri`, PKCE downgrade, DCR spam, code replay; (d) **consent silently reused** across sessions once remembered. | 🟠 | **partial (v1; pre-public gate open)** | **AuthN/Z**: every HTTP tool call presents a bearer verified against the app's tailnet-only `/internal/mcp/verify` (fails **closed**); the MCP process never mounts the user store. `mcp_access` **re-checked live** at connect (revoke is immediate, even for an already-minted token). **Tokens**: opaque, stored **SHA-256-hashed**; PATs shown once; OAuth access 1h + **rotating** refresh; auth codes single-use 60s. **OAuth 2.1 + PKCE (S256 only)**, **public clients only** (no secret), `redirect_uri` allow-list enforced at both authorize + token (anti open-redirect, test-covered), loopback-only http per RFC 8252. **Discovery**: RFC 9728 protected-resource metadata + `WWW-Authenticate` pointer; RFC 8414 AS metadata. **Consent** screen **discloses the redirect origin + registration date** (a look-alike `client_name` can't hide where the code goes) and offers **Deny**; remembered per `(user,client,scope)`; clickjacking headers (`X-Frame-Options`/CSP). **Scope** allow-listed (`mcp:read` only — unknown scopes 400). **DoS/abuse guards**: verify runs **off the event loop** (a slow/hung app can't stall every connection); DCR **capped** (total clients + redirect_uris per client); expired grants **purged** on issue; **Origin** allow-list (DNS-rebind guard, `APP_MCP_ALLOWED_ORIGINS`). **Revocation** = admin **`mcp_access` pull** (immediate, connect-time — the effective kill-switch) **plus user-facing per-connection disconnect** (`DELETE /api/app/mcp/connections/{client_id}` — forgets consent AND drops the client's live access/refresh grants, so it dies at the next tool call; surfaced in both "Connected agents" UIs). **Audit**: security events (verify denials, token issued/denied, PAT + consent create/revoke) append to `app_audit`. **Audience-bound** (RFC 8707): OAuth access tokens carry `aud` = the MCP resource URL; the resource server rejects a token minted for a different resource (a PAT carries no `aud` and is user-scoped). **Per-principal rate-limit**: app-level in-process limits beyond the edge's per-IP — DCR ~5/min per IP, token exchanges ~10/min per client → 429 (single worker → one shared counter; resets on restart). **`Mcp-Session-Id`↔principal**: not a hijack vector — every MCP request independently presents AND re-verifies a bearer (no ambient session-cookie authority), so FastMCP's session-id is transport state, not auth; closed by design. **Accepted residue (v1, explicit):** per-user corpus **scoping** deferred (v1 shared-corpus → the grant buys gating/attribution/revocation, **not** cross-user confidentiality; there is no per-user data yet). The in-process rate-limit must move to a shared store if the api ever runs multiple workers/hosts. **Pre-public gate**: authN/Z ✓. Deploy plumbing **built + locally smoke-tested** (2026-08-06): `mcp` service (corpus-RO, low-priv image, tailnet verify), `infra/caddy/mcp.caddy` resource vhost, `player.caddy` coming-soon **exemptions** for the OAuth AS paths, nginx proxies the root AS metadata, `deploy-player.*` stage `INTERNAL_MCP_TOKEN` + install the vhost only when MCP is on. Remaining deploy-time (operator): DNS `A mcp.<domain>`, set `PLAYER_INTERNAL_MCP_TOKEN`, run deploy. CORS: not needed for claude.ai (server-side, no Origin); the in-process Origin allow-list covers browser clients. |

**Controls already in place (do not regress):** SSH tailnet-only (`main.tf:47`),
`unattended-upgrades` on (`prod.user-data`), root login disabled, TF state sops/age
encrypted, corpus restore drilled, Tailscale as admin plane, Hetzner delete/rebuild
protection on the server + volume (post-2026-05-29).

## Pre-public gate — run before any new public vhost

A tenant may be exposed on the shared edge **only if all of these pass**:

- [ ] The public-facing container has **no `docker.sock`** mount and **no write-scope
      cloud keys** in its env. (T-01)
- [ ] Container runs `cap_drop: ALL`, `no-new-privileges`, non-root, read-only rootfs
      where feasible. (T-01, T-04)
- [ ] Any API requires **authN/authZ**, tight **CORS**, and **rate limiting**. (T-06)
- [ ] Caddy `admin off`; on-demand TLS off; catch-all denies unknown `Host`. (T-02, T-03)
- [ ] Egress to `169.254.169.254` (metadata) blocked from the tenant. (T-07, T-10)
- [ ] Deployed image is **digest-pinned** (or cosign-verified). (T-09)
- [ ] Security alerting covers the new surface. (T-11)
- [ ] Rollback proven: pull vhost + reload → public down, tailnet still up. (ADR-114 §7)

> orrery (static site, no `docker.sock`, low surface) clears this gate easily. The
> risk concentrates the moment a **podcast-family API** (e.g. player `:8092`) shares
> the operator `api`'s privilege profile — that is the case this gate exists to stop.

## Revision log

| Date | Change | By |
| --- | --- | --- |
| 2026-07-08 | Initial threat model created alongside ADR-114 (shared public edge). Registered T-01…T-12. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | #1160 Phase 1: container hardening landed (`cfe25d53` no-new-privileges, `a7f49cec` caps + read_only), boot-verified. T-04 → partial. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | Decisions: **D1** accept RW `docker.sock` on private api (T-01) — conditional on public/private API separation, opened **#1161**; **D2** keep `:sha-<7>` tag (T-09 accepted); **D3** accept shared tenancy (T-04 residue), orrery-split deferred to budget. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | #1160 Phase 2 (cloud-init): SSH key-only drop-in, fail2ban (sshd jail), metadata-egress SSRF guard. T-07/T-10 → partial. **Applies on rebuild; live box needs imperative apply.** T-10 full egress + T-11 alerting deferred. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | Secret infra: **ADR-115** (sops/age at rest → tmpfs + file mounts, multi-tenant bring-your-own-secrets). T-08 → in progress. OpenBao future idea → **#1162**. Mechanism PoC-verified locally. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | ADR-115 substrate landed (cloud-init): `decrypt-secrets.sh` (file()-injected) + per-tenant `/run/secrets` + `/etc/vps-secrets` dir + podcast decrypt sudoers. Key-provisioning **decision (a)**: operator age key staged by deploy over tailnet. Remaining: deploy staging + compose `secrets:` + C1 shim + operator's key & `prod.enc.yaml`. | Marko + Claude (Opus 4.8) |
| 2026-07-09 | Edge hardening (cloud-init): `caddy-access` fail2ban jail (bans 4xx-burst scanners at the INPUT chain) + Caddy access log pinned to JSON+ISO8601 for a version-stable filter. T-05 stopgap (not DDoS). **Applies on rebuild / live.** | Marko + Claude (Opus 4.8) |
| 2026-07-09 | **D4: accept Cloudflare additive front — ADR-118.** Origin prep: Caddy `trusted_proxies` (CF real-IP) + `client_ip_headers`; fail2ban filter re-keyed `remote_ip`→`client_ip` (post-CF `remote_ip` is a CF edge — banning it = outage); optional `cloudflare_origin_lock` TF firewall narrows :443 to CF ranges (:80 stays open for ACME). Operator owns CF account/DNS + the lock flip. T-05 → in progress. | Marko + Claude (Opus 4.8) |
| 2026-07-09 | #1163 player-public built: `PODCAST_SERVE_APP_ONLY` backend (only `/api/app/*` + health, no `/api/*`, no sock/keys — test-locked), player nginx restricted to `/api/app/`, standalone `docker-compose.player-public.yml` (app-only backend + PWA, docker.sock count = 0), `infra/caddy/player.caddy` vhost. Satisfies the pre-public gate for the player. Remaining: real domain/DNS, prod OAuth secrets, CORS for mobile (#1166), deploy. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | Caddy edge engine built (#1158 / ADR-114): firewall 80/443 (TF, apply-gated), cloud-init Caddy install + hardened base Caddyfile (`admin off`, on-demand-off + catch-all `abort`, shared HSTS/log snippet) + sites dir + `99-caddy-reload` sudoers. `caddy validate` green (base + `import hardened` vhost). T-02 → mitigated-on-rebuild, T-03 → mitigated-on-rebuild, T-12 → partial. Applies on rebuild; live box + firewall apply pending. | Marko + Claude (Opus 4.8) |
| 2026-07-08 | ADR-115 B+C build-out: C1 secret shim (`docker/secrets-shim.sh`) baked into api + pipeline Dockerfiles — **image-boot-verified** (file→UPPERCASE env; tolerant of absent secrets = non-breaking; `/api/health` 200). Cutover overlay `docker-compose.secrets.yml` + `infra/secrets/` scaffold (template, README, `.gitignore`, `.sops.yaml` rule). decrypt perms fixed to 0444-in-0700 for non-root pipeline reads. **Remaining (needs operator + deploy): `deploy-prod.yml` staging + add overlay to deploy chain + operator's age key & `prod.enc.yaml`.** | Marko + Claude (Opus 4.8) |
| 2026-07-22 | Pre-public gate run for the two pending surfaces — orrery vhost + the new GlitchTip **ingest** vhost (`infra/caddy/orrery-telemetry.caddy`). Both clear for launch; no hard blockers. Open: orrery `web` container hardening + image-pin (orrery-side); GlitchTip ingest is unauthenticated-by-design with no hard edge rate-limit → decision fail2ban+quota vs `caddy-ratelimit` plugin; add an ingest-flood alert. | Marko + Claude (Opus 4.8) |
| 2026-08-06 | **T-13 registered: remote MCP server (RFC-112).** New public per-user surface for external AI agents (own vhost). v1 controls in place + test-covered: fail-closed bearer verify via tailnet-only `/internal/mcp/verify` (MCP process never mounts the store), live `mcp_access` re-check at connect, SHA-256-hashed tokens (PAT once / OAuth 1h access + rotating refresh / 60s single-use codes), OAuth 2.1 + PKCE-S256 public-client-only with `redirect_uri` allow-list at authorize+token (anti open-redirect), RFC 9728 protected-resource discovery + `WWW-Authenticate` pointer, remembered+revocable consent. Accepted residue: tokens not `aud`-bound (single resource) + per-user corpus scoping deferred (v1 shared-corpus → grant = gating/attribution/revocation, not confidentiality). Pre-public: same gate (rate-limit at edge + connector CORS allow-list). | Marko + Claude (Opus 4.8) |
| 2026-08-06 | **T-13 hardening pass (fable-5 review).** Landed: verify off the event loop (DoS fix), consent screen discloses redirect origin + registration + Deny + clickjacking headers, scope allow-list (400 on unknown), DCR client + redirect_uri caps, expired-grant purge on issue, Origin allow-list (DNS-rebind guard), non-ASCII PKCE verifier fails closed, scope carried in the verify response, nginx proxies the root AS-metadata. T-13 → **partial** (was over-claimed "mitigated"); residue made explicit: no `aud`-binding, no per-principal rate-limit, no `app_audit`, no consent-revocation UI, per-user scoping deferred. Pre-public still needs the MCP resource vhost + coming-soon gate exemptions. | Marko + Claude (Opus 4.8) |
| 2026-08-06 | **T-13 residue burn-down.** Wired `app_audit` over the MCP security events (verify-seam denials, OAuth token issued/denied, PAT + consent create/revoke). Added **user-facing OAuth-connection revocation**: `GET`/`DELETE /api/app/mcp/connections` (forget consent + drop the client's live access/refresh grants → dies at next tool call), surfaced in the player + operator-viewer "Connected agents" UIs. T-13 residue now: no `aud`-binding, per-user scoping deferred, no app-level per-principal rate-limit, `Mcp-Session-Id` binding unexamined. | Marko + Claude (Opus 4.8) |
| 2026-08-06 | **T-13 residue closed to one item (operator-directed).** Added **audience-bound tokens** (RFC 8707: `aud` = MCP resource URL, minted by the AS, checked at the resource boundary in the transport middleware — a token for another resource is rejected; PATs are user-scoped, no `aud`). Added **app-level per-principal rate-limit** (`app_rate_limit`: in-process sliding window; DCR ~5/min per IP, token ~10/min per client → 429; single-worker caveat documented). Closed **`Mcp-Session-Id`↔principal** by reasoning (per-request bearer re-verify ⇒ no ambient session authority to bind). Sole remaining residue: per-user corpus scoping (v1 shared-corpus, by design). | Marko + Claude (Opus 4.8) |
| 2026-08-06 | **T-13 deep-review fixes (2nd fable-5 pass, whole arc).** Closed: **H1** — user Disconnect now also drops any un-exchanged auth *code* (was a 60s window that resurrected a 30-day grant); **H2** — code-exchange + refresh re-check `mcp_access` live (an admin pull now terminates the credential chain, not just reads); **H3** — the rate limiter evicts the OLDEST keys instead of clearing the whole table (a key-flood can't reset live counters); **M1** — the audience check fails **closed** when a resource server has no `APP_MCP_RESOURCE_URL` (was fail-open); **M3** — the `mcp` compose service got a healthcheck; **M4** — the verify-seam denial audit is throttled (bounded write-amplification); **M2** — the deploy probe now checks discovery-metadata consistency (resource/issuer). The per-IP DCR limit is documented best-effort (edge-attested XFF behind CF; the hard client cap is the backstop). Residue unchanged: per-user corpus scoping (v1 shared). | Marko + Claude (Opus 4.8) |
