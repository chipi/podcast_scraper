# Cloudflare WAF + rate-limit — draft to apply (P2-5 / T-05)

Apply in the **`closelistening.app`** zone (operator + player + telemetry are all
subdomains of it → one zone, scope by `http.host`). Everything below is free-tier-oriented;
where free-tier limits bite, notes call it out. Uncertain free-tier exact caps are flagged —
the CF UI shows the allowed ranges when you create each rule.

---

## A. Managed WAF ruleset
**Path:** `closelistening.app` zone → **Security → WAF → Managed rules**.

- Deploy the **Cloudflare Managed Ruleset**. On **Pro+** this is the full Managed Ruleset +
  OWASP Core; on **Free** it's the limited "Cloudflare Free Managed Ruleset" (high-impact CVEs
  only). Turn on what your plan exposes.
- Action for the ruleset: **Managed Challenge** (gentler — legit users pass silently) or
  **Block** for high-severity.
- **Honest caveat:** the *full* WAF (OWASP, custom managed rules) needs **Pro ($20/mo)**. Free
  gives a real but limited baseline. If you want the full L7 ruleset, that's the upgrade — flag
  for a cost decision; not required to get *a* baseline today.

---

## B. Rate-limiting rules
**Path:** `closelistening.app` zone → **Security → WAF → Rate limiting rules → Create rule**.

> Free plan allows a **limited number** of rate-limiting rules (roughly 1; Pro gives more) and
> may restrict the window (10s). If you can only make **one**, use **Rule 0 (combined)**. If
> you have room, use the granular Rules 1–3.

### Rule 0 — combined (use if free tier caps you to one rule)
- **Name:** `sensitive-endpoints-throttle`
- **If incoming requests match (expression editor):**
  ```
  (starts_with(http.request.uri.path, "/api/app/auth/")) or (http.request.uri.path eq "/preview")
  ```
- **Rate:** `20` requests per `10` seconds
- **Counting characteristics:** by **IP** (default)
- **Then:** **Managed Challenge** (or Block), duration `60s`
- Covers both surfaces' OAuth entrypoints + the `/preview` doorman in one rule.

### Rule 1 — operator auth (granular)
- **Name:** `operator-auth-throttle`
- **Expression:**
  ```
  (http.host eq "operator.closelistening.app" and starts_with(http.request.uri.path, "/api/app/auth/"))
  ```
- **Rate:** `10` / `10s` per IP → **Managed Challenge**, `60s`.

### Rule 2 — coming-soon doorman brute-force (granular)
- **Name:** `preview-doorman-throttle`
- **Expression:**
  ```
  (http.request.uri.path eq "/preview")
  ```
- **Rate:** `5` / `60s` per IP → **Block** (or Managed Challenge), `60s`.
- Rationale: `/preview` is the basic-auth (bcrypt) doorman — the only crackable pre-launch
  surface; throttle brute-force here hard.

### Rule 3 — player auth (granular, optional)
- **Name:** `player-auth-throttle`
- **Expression:**
  ```
  (http.host eq "closelistening.app" and starts_with(http.request.uri.path, "/api/app/auth/"))
  ```
- **Rate:** `10` / `10s` per IP → **Managed Challenge**, `60s`.

---

## Why Managed Challenge over Block for the auth rules
A login endpoint gets legitimate retries; behind a shared/corporate IP a hard **Block** could
lock a real user out. **Managed Challenge** lets a human through silently while stopping scripts.
Reserve **Block** for `/preview` (brute-force) where there's no legit high-frequency use.

## After you apply — verify
- CF dashboard → **Security → Events**: hammer `/preview` a few times from a throwaway network;
  you should see the rate-limit rule fire.
- Confirm normal sign-in still works (don't over-tighten the auth rule).

## Relationship to what's already in place
- **Origin-lock is on** (`:443` → CF ranges only), so CF is the *only* path in — these rules are
  the real front-line L7 control. The origin nginx limits (PR #1334 P1-1) are defense-in-depth
  behind them.
- These replace what the origin fail2ban can't do post-CF (it can't ban an IP that never
  connects to the origin directly).
