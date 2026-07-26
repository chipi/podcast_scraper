# Production Security & Ops Review — podcast_scraper public surfaces (2026-07-26)

**Reviewer:** advisor role, Fable 5 (`claude-fable-5`), read-only.
**Scope:** all three surfaces — player (`closelistening.app`), operator
(`operator.closelistening.app`), and the tailnet-only privileged plane — on the
shared single-VPS Cloudflare→Caddy edge. Security + operations weighted.
**Status:** analysis only; nothing changed, nothing acted on. Decide actions from here.

> Every risk cites a file/line/config or a live probe. VERIFIED = read/probed;
> INFERRED = reasoned, not confirmed. Blast radius is assessed against the
> *current* single-email allowlists — several risks are low-impact **today**
> precisely because only one email can sign in, and escalate the moment the
> allowlist widens at launch.

---

## Orientation — confirmed live state

- Both public vhosts up **behind Cloudflare**, serving coming-soon: `curl -sSI`
  → `HTTP/2 200` + `server: cloudflare` + `cf-ray`. VERIFIED.
- The coming-soon gate wraps **everything except static assets**. `GET
  /api/app/auth/status` and `GET /api/index/rebuild` both return `200
  text/html` (coming-soon body) — the real API is **not publicly reachable at
  all right now**; the whole posture rests on the secret preview cookie +
  `/preview` doorman. `/preview` correctly 401-challenges. VERIFIED.
- `:80` → `308` https; `www` → `301` apex; `assets/index.js.map` → `404`
  (source maps not exposed despite `.map` in the `@static` allowlist). VERIFIED.
- T-01 in code: `app.py:170-191,224-227` mount only `_OPERATOR_PUBLIC_READ_ROUTES`
  behind router-level `require_viewer_access`; `index_rebuild`/`ops`/`resilience`
  excluded (`app.py:161-168`). Public compose carries **no `docker.sock`, no
  provider keys, no exposed api port**; corpus mounted `:ro`. VERIFIED — the
  privileged-plane separation is real in the artifacts visible from this repo.

---

## A. KEY RISKS & ISSUES RIGHT NOW

### P0 — verify before relying on anything else

**P0-1 — Is the origin-lock firewall actually applied on the live box? The
entire CF WAF/DDoS/rate-limit layer is only as real as this one firewall rule.**
- GH var `CLOUDFLARE_ORIGIN_LOCK=true` exists (set 2026-07-22). TF gates `:443`
  on it (`infra/terraform/main.tf:81`; default `false`, `variables.tf:166`).
  Could **not** confirm it was `tofu apply`ed with the var true (can't reach the
  origin IP behind CF-proxied DNS; won't run TF).
- Why P0: operator viewer has **no origin rate-limiting** (P1-1); CF WAF/rate-limit
  is "optional" per ADR-118 step 6. If `:443` is still `0.0.0.0/0` and an attacker
  finds the origin IP (Hetzner ranges enumerable; historical DNS; cert SAN), they
  bypass CF entirely → Caddy → both backends.
- Blast radius: full loss of the CF protective layer for both public surfaces.
- Check (~5 min, read-only): `hcloud firewall describe <fw>` (or TF state) that the
  `:443` rule lists CF ranges; from an external host `curl --resolve
  closelistening.app:443:<origin-ip> https://closelistening.app/` should time out.

### P1 — soon

**P1-1 — Operator-public has no origin rate-limiting; RFC-108 claims parity the
code doesn't deliver (doc-vs-code divergence).**
- `grep limit_req web/gi-kg-viewer/` → nothing. Player has it:
  `web/learning-player/nginx.conf:27-29,49,55` (`lp_api` 20r/s, `lp_auth` 2r/s,
  `429`) with real-IP recovery (`nginx.conf:21-23`). RFC-108:42,53 asserts the
  operator has "edge rate-limit (T-06), same as the player."
- Blast radius today bounded by the gate + single-email allowlist; at launch the
  operator OAuth callback + curated corpus routes have no per-IP origin throttle —
  only (unverified) CF stands between an attacker and unbounded hammering.
- Fix: add a `limit_req` zone to the gi-kg-viewer nginx mirroring the player, **or**
  confirm+document CF rate-limit rules as the enforced control (then fix the RFC to
  point at CF, not origin).

**P1-2 — Coming-soon doorman bcrypt hashes committed to a PUBLIC repo, identical
on both vhosts, no rotation.**
- `gh repo view` → `isPrivate: false`. Hashes at `infra/caddy/player.caddy:56-57`
  and `operator.caddy:23-24` byte-identical (added 2026-07-24, `7da782ad`).
  Plaintext only in GH secret `PLAYER_PREVIEW_PASS` (VERIFIED present).
- The doorman bcrypt is the crown jewel: crack offline → `curl /preview -u
  marko:<pw>` → receive `Set-Cookie` → full app access. cost-14 ≈ 1-2s/guess
  (strong random = safe; a dictionary word, esp. `guest`, is crackable). Everyone
  has the hashes.
- Blast radius: bypass of the entire pre-launch gate on both surfaces at once
  (shared creds).
- Fix: confirm both plaintexts are high-entropy random; rotate if not. Strategically
  the gate secret shouldn't be a repo-committed hash (B-item). INFERRED on strength.

**P1-3 — The `≥creator` gate is auto-satisfied by the viewer's `grant=creator`
hint, so the email allowlist is the *only* real authZ boundary on the operator
surface.**
- Login URL `/api/app/auth/login?grant=creator` (`authApi.test.ts:22`,
  `UserMenu.test.ts:9`); `resolve_login_role` (`app_roles.py:85`) promotes any
  `listener` → `creator` on that hint. So `require_viewer_access`
  (`app_auth.py:81-92`) passes for anyone who can sign in.
- RFC-108 frames "double gate + role authZ"; in practice the role check adds no
  independent defense — security depends entirely on `OPERATOR_ALLOWED_EMAILS`
  (currently `marko.dragoljevic@gmail.com`, `allowlist`, VERIFIED via `gh variable
  list`). If anyone sets `OPERATOR_SIGNUP_MODE=open`, the corpus-read surface opens
  to any Google account (self-granted creator).
- Blast radius today: none (allowlist = 1). Latent footgun.
- Fix: gate operator-read on a role NOT auto-granted (allowlist-derived
  creator/admin, not the request hint), or a startup assertion that
  `operator_public` refuses `APP_SIGNUP_MODE=open`. Document the allowlist as the
  load-bearing control.

**P1-4 — Operator per-user appdata has no backup; it went live today.**
- `docker-compose.operator-public.yml:63-65` bind-mounts
  `/srv/podcast-scraper/operator-appdata` (prefs/library — real user data, not
  regenerable). No operator backup workflow; player appdata IS backed up daily
  (`backup-player-appdata-prod.yml:24`, cron `17 4 * * *`).
- Fix: clone the player appdata backup → operator path. S.

**P1-5 — Prod corpus has no scheduled backup.**
- `backup-corpus-prod.yml:28` — "schedule intentionally absent"; dispatch-only,
  deferred to #723 cutover. Corpus is expensive to regenerate.
- Blast radius: loss between manual runs if the volume dies (Hetzner delete
  protection ≠ backup).
- Fix: enable a cron now (even weekly) rather than waiting for #723. Confirm a
  restore drill has actually targeted **prod** (threat model:82 claims "corpus
  restore drilled" — verify target wasn't the codespace).

### P2 — track

- **P2-1 — Operator secrets at rest on disk; player/pipeline are not.**
  `OPERATOR_SECRETS_VIA_FILES` absent from `gh variable list` (→ off), so
  `deploy-operator.yml:227-237` writes OAuth client secret / session secret /
  Sentry DSN into `.env.operator` (0600) on disk, while `PLAYER_SECRETS_VIA_FILES=1`
  + `PODCAST_SECRETS_VIA_FILES=1` (VERIFIED) put player+pipeline on tmpfs (ADR-115).
  Operator shares the player's OAuth client + session secret → disk copy widens
  exposure. Fix: set `OPERATOR_SECRETS_VIA_FILES=1` (machinery exists,
  `deploy-operator.yml:248-279`).
- **P2-2 — `APP_ADMIN_EMAILS` contains `info@closelistening.com` (.com, not the
  `.app` you own).** VERIFIED via `gh variable list`. Harmless today (not in an
  allowlist) but an admin grant to an outside party if the allowlist widens +
  you don't own `.com`. Confirm ownership or drop it.
- **P2-3 — HSTS still ramp value** `max-age=86400`, no `preload`
  (`infra/cloud-init/Caddyfile:42`, VERIFIED live). Two live vhosts now → bump to
  `31536000; includeSubDomains; preload` when confident.
- **P2-4 — `set_real_ip_from 0.0.0.0/0`** in player nginx (`nginx.conf:21`) trusts
  XFF from any peer; safe only because the container port is loopback-bound
  (`docker-compose.player-public.yml:121`). If that binding regresses, XFF spoofing
  defeats the rate limiter + fail2ban. Narrow to the Caddy/compose bridge CIDR.
- **P2-5 — No continuous uptime/availability alerting.** Post-deploy smokes fire
  once per deploy; GlitchTip catches app errors; nothing pages if a container
  crashes or Caddy dies between deploys. A 3am OOM is silent. Add an external
  synthetic probe on `/`.
- **P2-6 — fail2ban `caddy-access` ban efficacy unverified on the live box.**
  Filter keys on `client_ip` (CF-safe, correct — `prod.user-data:150`), but threat
  model repeatedly flags "applies on rebuild; live box needs imperative apply."
  Confirm loaded on the running host: `fail2ban-client status caddy-access`.

---

## B. OPPORTUNITIES TO MOVE FURTHER (maturity roadmap)

**Edge / DDoS**
- Configure CF managed WAF + rate-limit rules (ADR-118 step 6, currently
  "optional") — the actual T-05 control; without it CF is mostly TLS + caching. **S**.
- ADR-118 Alternative B (CF Origin CA cert + close `:80`) to kill the world-open
  `:80` residue and enforce CF-only. **M** (needs ADR-115 file-delivered cert).
- Move the coming-soon gate to a CF-side control (WAF rule / CF Access) instead of
  committed bcrypt hashes. **M**.

**Secrets**
- Turn on `OPERATOR_SECRETS_VIA_FILES` (P2-1); then finish ADR-115 sops/tmpfs for
  the privileged `.env` (still env-injected per T-01/T-08). **M**.
- Provider billing caps + per-env keys (D5/T-08) — bounds an LLM-key-leak blast
  radius container hardening can't. **S-M**, provider-side.
- Documented rotation cadence for OAuth client secret, session secret, gate cookie
  (none today). **S** doc + **M** automation.

**AuthZ**
- Decouple operator-read gate from `grant=creator` self-promotion (P1-3); assert
  `operator_public` refuses `signup_mode=open`. **S**.
- CORS allowlist for eventual mobile/native origins (T-06; `app.py:351-369` still
  defaults to localhost) before any non-same-origin client ships. **S**.

**DR / blast radius**
- Scheduled prod corpus + operator-appdata backups (P1-4/P1-5) + a **prod** restore
  drill (not codespace). **S** each.
- Documented <5-min rollback per surface (ADR-114 §7); rehearse once, record timing. **S**.
- orrery-to-own-VPS split — the real isolation upgrade when budget allows (T-04,
  ADR-114 §7). **L**.

**Observability / IR**
- External synthetic uptime + per-surface availability alert (P2-5). **S**.
- Per-tenant app-security alerting (auth-fail spikes, per-app 4xx/5xx, egress
  anomalies) — deferred to Goals 2/3 per T-11. **M**.

**Supply chain**
- Digest-pin GHCR images (T-09; currently `:sha-<7>` resolved at deploy — better
  than `:main`, not immutable). **M**.
- Pin the Caddy apt version (T-09 open). **S**.

**Multi-tenant isolation**
- Shared Caddy edge + shared `deploy` user + shared `.env` is one compromise from
  all tenants (T-04). Per-tenant deploy users / a socket-proxy for the privileged
  api are the maturity steps once orrery justifies them. **L**.

**Suggested ordering:** P0-1 verify → P1-1/P1-4/P1-5 (cheap, real gaps) → CF
WAF/rate-limit rules → P1-2/P1-3 (gate hardening) → secret unification → DR drills
→ digest-pin/isolation.

---

## C. HONEST NOT-COVERED / UNKNOWNS (equal weight)

Several are load-bearing — do not read silence elsewhere as assurance.

1. **Origin-lock firewall live? (P0-1)** — GH var says `true`; couldn't reach origin
   IP / read TF state / `hcloud`. Single most important thing to confirm.
2. **CF WAF + rate-limiting configured?** ADR-118 step 6 optional; may be the only
   L7 control on the operator surface. No CF-account visibility. Check CF dashboard.
3. **Doorman password strength (P1-2)** — didn't read `PLAYER_PREVIEW_PASS` (and
   shouldn't). Dictionary word → committed cost-14 hashes crackable. Confirm random.
4. **Privileged tailnet-only plane** — not touched (as instructed). T-01 assessment
   rests on the *public* compose showing no socket/keys/port (VERIFIED); did NOT
   independently confirm the privileged api is internet-unreachable or that
   `compose-api-1` on the box matches the repo. Check off-tailnet port scan +
   on-box `docker inspect compose-api-1` vs `docker-compose.prod.yml:80,155`.
5. **fail2ban / cloud-init hardening on the *running* box** — verified config, not
   running state. Check `fail2ban-client status`, `iptables -L DOCKER-USER` (metadata
   DROP), `sshd -T | grep -i passwordauth`.
6. **Ownership of `closelistening.com` (P2-2)** — unknown. Check registrar.
7. **Did the corpus restore drill ever target prod** vs codespace? Not confirmed.
8. **CF-sees-plaintext acceptance** — ADR-118 accepts CF terminating TLS for
   public/consumer content; the operator-read corpus (asset #2) now flows through CF
   in plaintext at their edge. Accepted trade-off, worth a conscious re-confirm now
   that it's the *operator* corpus.

**Assumptions:** live `compose-api-1`/Caddy match repo `production`; `gh
variable/secret list` reflects what the last deploy staged; only the two
allowlisted-to-one-email surfaces are internet-exposed from this repo (orrery shares
the edge from another repo — its vhost hardening + image pinning out of scope, flagged
as a shared-edge cross-tenant concern).

---

## Meta — how this review was run

- Run on **Fable 5** via `Agent(subagent_type:"advisor")` with **no `model`
  override** (the `block-opus-subagents` hook blocks explicit `model:"fable"` at its
  catch-all but allows the no-model advisor path, which inherits the Fable 5
  frontmatter). An earlier Sonnet pass was discarded, not merged into this.
- Optional hook fix if explicit `model:"fable"` should stop biting: mirror the
  explicit-opus/advisor branch — `const isFable = /(^|[-/])fable([-/]|$)/.test(model);
  if (isFable && ROLE_OPUS.has(agent)) allow();` before the catch-all. Not required;
  the clean habit is to omit `model` on tiered roles and let the frontmatter drive.
