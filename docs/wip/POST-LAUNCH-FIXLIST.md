# Post-launch fix list (orrery live ~12h, 2026-07-23)

Compiled after the 12h o11y health check. **Overall: healthy** — all signals flow
(metrics 1450 req/12h, logs, traces, errors, Umami analytics collecting). These are
the open items.

## A. Needs the prod SSH key (box work) — do first when back

1. **Orrery log mislabeling — ~48% of `app=orrery` logs are wrong.**
   - **Confirmed:** `app=orrery` total 38,054/12h; only 19,819 are real orrery-web;
     ~18,235 are prod-podcast infra (podcast uvicorn `GET /metrics`, cAdvisor
     `manager.go:1116` stderr, healthz) wrongly stamped `app=orrery`, no container label.
   - **Root:** the orrery grafana-agent config **deployed on the VPS ≠ the repo file**
     (`orrery: ops/observability/grafana-agent.yaml` already has a `keep` filter for
     `orrery-web|orrery-pipeline-runner-*` and does NOT set `surface=web`, but the live
     proper logs DO have `surface=web` → config drift on the box).
   - **Fix:** SSH the box → inspect the actually-running orrery agent/Alloy config →
     scope its log source to `orrery-*` containers only (drop the no-container
     fall-through) → restart `orrery-grafana-agent` → **reconcile the corrected config
     into the orrery repo** (kill the drift) → verify in VictoriaLogs that the
   - **PROPER FIX (chosen):** per ADR-121 — one node Alloy per box + per-app rule drop-ins. Retire orrery-grafana-agent; orrery ships ops/observability/orrery.alloy into /etc/alloy/config.d/ (mirrors orrery.caddy). Repo-track the base Alloy config (kills the /opt drift).
     `{env=prod}`-only stream stops.

2. **Edge / security 12h sweep** (couldn't run — key was dropped):
   - fail2ban jails: any bans in 12h of public exposure? (`sshd`, `caddy-access`)
   - Caddy access log: 4xx/5xx burst patterns = scanner probing?
   - LE cert expiry dates for orrery / telemetry / analytics (auto-renew, just confirm).
   - Confirm caddy + fail2ban + metadata-egress guard all active.

## B. Orrery repo (orrery agent's side — no prod key) — 2 real JS errors surfaced

Both are real user-facing, 1 occurrence each in GlitchTip **project 4**:
3. `TypeError: Script https://orrerylearn.com/sw.js load failed` — PWA **service worker**
   failed to load (CF cache quirk or real). Hand to the orrery agent.
4. `TypeError: Failed to fetch dynamically imported module: https://www.or…` —
   lazy-loaded module fetch failed (note the `www.` — possible www-vs-apex / CF-cache
   edge case). Hand to the orrery agent.

## C. Operator actions (no key)

5. **Player domain** — epic #1262 is parked on you registering a separate domain
   (#1263). Gate/corpus/OAuth already done; after the domain it's ~15 min to deploy.
6. **(optional) Rotate the Google client secret** — it was pasted in chat, so it's in
   the transcript. It's correctly stored in GH Secrets; regenerate in Google Console +
   re-paste if you want it out of history. Not urgent.
7. **(optional) Viewer telemetry test** — over the tailnet: open
   `https://prod-podcast.tail6d0ed4.ts.net/`, trigger a JS error, confirm it lands in
   GlitchTip project 1. (Needs Tailscale, not the prod key.)

## D. Cloudflare settings to tune (orrery zone) — free plan

Static SPA behind CF + the `telemetry` / `analytics` ingest subdomains. The two
DANGER options are called out because they'd silently break this specific setup.

**✅ Enable (safe wins):**
- **SSL/TLS → Full (strict)** — origin has a valid LE cert; strict rejects a MITM'd origin.
- **Always Use HTTPS** + **Automatic HTTPS Rewrites** (kills mixed content).
- **Minimum TLS Version → 1.2** (drop 1.0/1.1).
- **DNSSEC** — prevents domain DNS spoofing.
- **Managed WAF ruleset** (free tier) — baseline filtering.
- **HTTP/3 (QUIC)** · **0-RTT** · **Early Hints** · **Brotli** — free perf, all safe.
- Caching: default already caches orrery's static assets = the big perf win. Leave on.

**❌ Do NOT enable — breaks this setup:**
- **Rocket Loader** — reorders/defers JS → routinely breaks SPAs (orrery = Svelte). OFF.
- **Bot Fight Mode / Super Bot Fight** — challenges automated browser POSTs, i.e. the
  telemetry `/api/*/envelope` + analytics `/api/send` ingest → silent drops. OFF —
  or first add WAF **skip** rules for `telemetry.orrerylearn.com` + `analytics.orrerylearn.com`.

**⚠️ Watch:**
- Rate Limiting: OK for the orrery site; DON'T tighten it on the `telemetry`/`analytics`
  hosts (ingest needs headroom).
- Security Level: leave **Medium**; "High"/"Under Attack" challenge real users.
- CF won't cache POST by default, so `/api/send` + `/api/*/envelope` are safe.

**Validation (what's externally checkable):** min-TLS enforcement, HTTP/3 alt-svc,
Always-Use-HTTPS redirect, HSTS header, Brotli, DNSSEC, Rocket-Loader injection,
Bot-Fight not challenging ingest (POST still 200). SSL-mode + WAF need the dashboard.

## Not broken — verified healthy (no action)
- Orrery `200` + valid TLS; CDN + origin-lock intact.
- All 4 o11y signals + Umami analytics flowing.
- Both public ingest vhosts (telemetry, analytics) hardened + verified (admin 404).
