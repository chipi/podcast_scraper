# Pre-public gate assessment — orrery vhost + GlitchTip ingest vhost

Runs the `docs/security/THREAT_MODEL.md` **"Pre-public gate — run before any new
public vhost"** (8 items) against the two surfaces about to go public on the shared
edge (ADR-114). Prepared 2026-07-22 so the operator sign-off (Phase 3.2) is a quick
read. **Legend:** ✅ pass · 🟡 partial / decision · 🔲 confirm (other-repo/other-box) · ⛔ blocker.

**Bottom line:** both surfaces clear for launch. No hard blockers. Open items are
(a) orrery-side container hardening + image-pin, (b) one GlitchTip ingest rate-limit
decision, (c) one alerting add (folds into the Grafana task). Details below.

---

## Surface A — orrery vhost (`infra/caddy/orrery.caddy` → box `127.0.0.1:8090` = orrery `web` nginx)

Public-facing container is orrery's **`web`** (static nginx). `pipeline-runner` and
`grafana-agent` are internal (not proxied); the read-only `docker.sock` + `GRAFANA_CLOUD_API_KEY`
live **only** on `grafana-agent`, which is **not** on the public surface.

| # | Gate item | Verdict | Notes |
|---|---|---|---|
| 1 | No `docker.sock`, no write-scope cloud keys on the public container | ✅ | `web` (static nginx) mounts no sock and carries no keys; sock/key are on the non-public `grafana-agent`. Confirm `web`'s env stays secret-free. |
| 2 | `cap_drop: ALL`, `no-new-privileges`, non-root, read-only rootfs | 🔲 | **Not present on `web`** in `orrery/compose/docker-compose.prod.yml`. Static nginx can run `cap_drop:[ALL]` + `no-new-privileges` + `read_only` (with tmpfs for cache/run). **Orrery-side add.** Low risk (static), but the gate requires it. |
| 3 | API needs authN/authZ, tight CORS, rate limiting | ✅ | N/A — orrery is a static site, no API surface. |
| 4 | Caddy `admin off`; on-demand TLS off; catch-all denies unknown Host | ✅ | Engine-level (shared base Caddyfile): `admin off`, on-demand off, `:443` catch-all `tls internal`+`abort`. Live after Phase 1 converge (T-02/T-03 mitigated-on-converge). |
| 5 | Egress to `169.254.169.254` blocked from the tenant | ✅ | `block-metadata-egress` guard in cloud-init + apply-edge (converges Phase 1). Static site has no SSRF backend regardless. |
| 6 | Deployed image digest-pinned (or cosign-verified) | 🟡 | `web` is `orrery-web:local` (locally built, not registry/digest-pinned); the podcast family pins `sha-<7>` **tags**, not digests/cosign. Accept for launch or tighten to a digest. **Orrery-side / accept.** |
| 7 | Security alerting covers the new surface | 🟡 | Shared fail2ban `caddy-access` jail + Alloy→Loki cover the edge access log for all vhosts incl. orrery. No orrery app-errors (static, no GlitchTip by design). Sufficient; an access-anomaly alert would strengthen it (Grafana task). |
| 8 | Rollback proven (pull vhost + reload → public down, tailnet up) | ✅ | `rm /etc/caddy/sites/orrery.caddy && sudo systemctl reload caddy` → orrery public down, tailnet + box unaffected, nothing shared changes (vhost header, ADR-114 §7). **Smoke-test at cutover.** |

**Orrery verdict:** clears the gate. Two to-do before flip, both **orrery-side + low-risk**:
add `web` container hardening (#2), decide image-pin policy (#6). #7/#8 are covered/documented.

---

## Surface B — GlitchTip ingest vhost (`infra/caddy/glitchtip.caddy` → `homelab:8090` GlitchTip ingest, paths only)

Not a box container — a **reverse-proxy** to the self-hosted GlitchTip on **homelab**
(tailnet), scoped to the ingest paths only (`/api/<id>/{envelope,store,security}`);
everything else 404s and the admin UI stays tailnet-private.

| # | Gate item | Verdict | Notes |
|---|---|---|---|
| 1 | No `docker.sock`, no write-scope cloud keys | ✅ | Box-side is a proxy — no container, no sock, no keys. The exposed DSN key is a **public ingest key by design** (ships in browser bundles), not a write-scope cloud credential. |
| 2 | `cap_drop`/`no-new-privileges`/non-root/read-only | 🔲 | N/A box-side. The exposed service (GlitchTip) runs on **homelab**; its container hardening is homelab-owned. |
| 3 | API authN/authZ, tight CORS, rate limiting | 🟡 **decision** | Ingest is **unauthenticated by design** (public DSN write endpoint — how Sentry/GlitchTip ingest works). In place: **path-scoped** (only envelope/store/security; admin 404), **CORS** handled by GlitchTip, abuse bounded by the fail2ban `caddy-access` jail + **GlitchTip per-project event quotas**. **Not in place:** a hard *edge* rate-limit (stock Caddy has no `rate_limit` plugin). **DECISION:** launch with fail2ban + a strict per-project quota, or add the `caddy-ratelimit` plugin (a build dependency) if abuse appears. **Recommend:** launch with fail2ban + low quota; add the plugin only if needed. |
| 4 | Caddy `admin off`; on-demand off; catch-all denies unknown Host | ✅ | Engine-level (same base Caddyfile). |
| 5 | Egress to metadata blocked | ✅ | Proxies a **fixed** upstream (`{$GLITCHTIP_UPSTREAM}` = homelab), not user-controlled → no SSRF/metadata path. Box guard covers the rest. |
| 6 | Image digest-pinned | 🔲 | N/A box-side; homelab GlitchTip image pinning is homelab-owned. |
| 7 | Security alerting covers the new surface | 🟡 | Ingest access flows to the fail2ban jail + Alloy→Loki. A specific **ingest-flood alert** (rate spike on the ingest paths) would strengthen it → **add in the Grafana task.** |
| 8 | Rollback proven | ✅ | `rm /etc/caddy/sites/glitchtip.caddy && sudo systemctl reload caddy` → ingest down, GlitchTip still tailnet-reachable, box up (vhost header). **Smoke-test at cutover.** |

**GlitchTip ingest verdict:** clears the gate with **one decision** (#3: unauthenticated
ingest + no hard edge rate-limit → accept fail2ban+quota vs. add the plugin) and **one
strengthening** (#7: ingest-flood alert, Grafana task). #2/#6 are homelab-owned.

---

## Consolidated action list (nothing here is a hard blocker)

1. **Orrery-side:** add `cap_drop:[ALL]` + `no-new-privileges` + `read_only` (+tmpfs) to the `web` service (#2-A); decide digest-pin vs local image (#6-A).
2. **Decision (operator):** GlitchTip ingest rate-limit — fail2ban + per-project quota for launch, or add `caddy-ratelimit` (dep). Recommend the former (#3-B).
3. **Grafana task:** add an **ingest-flood** alert on the GlitchTip ingest paths + an edge access-anomaly alert (#7-A/#7-B).
4. **At cutover:** smoke-test both rollbacks (#8-A/#8-B) — `rm vhost + reload` drops the public surface, tailnet stays up.
5. Items 4/5 go live automatically once Phase 1 (apply-edge converge) runs.
