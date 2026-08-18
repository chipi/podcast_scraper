# Session handover — 2026-08-18/19

Written at the end of a long session. Read this before touching prod.

---

## 1. What prod looks like now

| | Value |
|---|---|
| Deployed image | `sha-cd22625` (pinned override, **not** the newest published) |
| Healthy GI episodes | **678** (was 568) |
| Legacy placeholders | **0** (was 110) |
| Episodes with no GI block | 0 |
| Damaged preprocessing runs | 2 — the 32-episode reprocess addresses these |
| Corpus size | 48 GB, of which **46.5 GB is audio** (`media` 30.5 · `audio-cache` 12.6 · `.tmp_media` 3.4) |
| Last verified backup | 2026-08-18 11:30, 368 MB (audio excluded) |
| Gateway key | alias `proj-podcast-prod`, `sha256:3b88f1c6ee41`, budget \$25 |

`sha-cd22625` was chosen deliberately over `sha-1444ebc`: the newer image carries the #1678
learning-player merge, and shipping a large unrelated app change during a corpus repair makes
attribution impossible if anything misbehaves. **Prod is therefore behind main on app code.**

---

## 2. The bug that ate the session

**All 11 prod credentials live in `/dev/shm/podcast-secrets/`, and that directory does not
survive a deploy.**

Docker copies secrets into a container at *create* time, so containers made during a deploy keep
working — the API stays up, health checks pass, everything looks fine. Any container created
*later* starts with nothing. That includes every pipeline job, because they are all
`docker compose run`.

ADR-115 (#1250, 2026-07-21) introduced this when it moved credentials out of `.env` for good
security reasons. It split one fact into two:

> "prod is deployed" stopped meaning "prod has credentials"

Nothing expressed that. The requirement lived in one commit message (`2317839e`, 2026-08-10,
"host dir not persistent") and a comment in a single workflow. Every workflow written afterwards
rediscovered it in production — four times before tonight.

### How it presented

A gateway 401. I read that as "the API key is wrong", had a **live production key deleted and
re-minted** to fix it, and ran three deploys chasing it. The key was always correct. Two further
"fixes" to the probe (both real bugs, both landed) still returned 401, because the actual answer
was that nothing was on the box at all.

The check that ended it took one request: *does the delivered key exist, and does it
authenticate?* Answer: the whole directory was missing.

**The lesson, stated plainly so it is not lost:** a bare status code is not evidence. I reasoned
from one for hours instead of going to look, because looking cost an approval click and 90
seconds. That is why #1687 exists.

### What was shipped for it

| Commit | What |
|---|---|
| `43bc1cb4` | prove-before-promote gate on `.env` (inert under VIA_FILES — documented as such) |
| `7cae2aa6` | D5 joins `compose/docker-compose.secrets.yml` — the key was never **mounted** |
| `c0d73ea1` | D5 reads `/run/secrets` directly — `--entrypoint` replaces the shim that **exports** it |
| `6ee0e02e` + `81cafc03` | the step-0 diagnostic that finally asked the right question |
| `c59260ad` | `.github/actions/stage-prod-secrets` + `check_prod_secret_staging.py` + make target + CI step |
| `78afaa17` | same two fixes applied to `mint`'s verify probe — the copy that was missed |
| `6f5bbfe4` | join the secrets overlay in gi-repair / reprocess / inspect; gate now requires **both** halves |

**Both halves are required and are separate bugs**: staging *delivers* the files, the compose
overlay *mounts* them, and the entrypoint shim *exports* them. Fixing one and not the other
produces a confident, wrong green.

---

## 3. Still open

### Highest priority — the real fix
The secrets directory should **persist**: root-owned, or `/run/podcast-secrets` instead of
`/dev/shm`. Then "deployed" implies "credentialed" and no workflow needs to know any of this.
Needs root on the box. Until then `check_prod_secret_staging.py` is the seatbelt, not the brakes.

### #1687 epic — Operator API for prod diagnosis
Filed tonight, with children #1688–#1692. Start with **#1690** (`/operator/secrets/status`) —
one endpoint, no dependencies, and it is precisely the question that went unasked for five hours.

### Unmerged branch — `fix/cleaner-and-open-items` (7 commits)
Never PR'd. Contains real work:
- `c6893892` transcript cleaner: cap the UNION of sponsor spans (**#1641–#1645**). 6 of 36 fixture
  episodes retained 12–16% of their text; the cleaner was deleting the episode and keeping the ad.
- `0ded2202` quota-403 failover pinned end-to-end (**#1634–#1639**), budget named in the message
- `ee06ac2b` by-design summary degradation reported as warning, not error (**#1632**)
- `d816c8a1` `podcast_obs` sys.path shadowing removed
- `efca9e94` doc-structure gate stopped failing on vendored files
- `9ad9c031` tests-and-docs audit
- `e58b8670` prove-before-promote gate (already cherry-picked to main as `43bc1cb4`)

### Smaller, known
- `PROD_ANTHROPIC_API_KEY` is empty (0 bytes on the box). `cloud_balanced` unaffected;
  **`cloud_thin` would fail** at `gi_value_gate_provider` if selected from the UI.
- `mint … action=mint` fails when its own alias exists; the remedy is a separate `delete_alias`
  input the operator has to know about. Should self-heal.
- `reprocess-prod.yml` has no `timeout-minutes` (inherits the 360-minute default).
- #1683's ranking/personalisation audit landed on main from another agent — not reviewed here.

---

## 4. Operational notes worth keeping

- **The transcript cache must be off for any repair.** Its key is media-hash +
  `preprocessing_fingerprint(cfg)`, and that fingerprint reads `pp=on` whether preprocessing
  succeeded or fell back to raw audio. Neither component differs between a damaged run and its
  repair, so a cache hit re-serves the exact transcript you are trying to replace.
  `use_transcript_cache` already defaults to `false` — keep it that way.
- **`attempts=0` is ambiguous.** A run that never preprocessed and a repair served entirely from
  cache both record it, and look identical to healthy. Assert positively with
  `verify_recent_runs` (`attempts >= 1`, `completed == attempts`).
- **`make format-check` cannot see F401 or black violations in new files** — only `make lint`
  catches the former, only `format-check` the latter. Run **both**. I broke main's lint tonight
  by running one and not the other; another agent had to fix it (`f261d607`).
- **Disabling the fix is the only proof a regression test has teeth.** My first cleaner test
  passed with the fix removed — it pinned nothing. So did the first version of the staging gate,
  which reported "OK — 1 workflow" while five went unchecked.
- **o11y is healthy** (checked 2026-08-18 22:50): all 7 services up, VictoriaMetrics 10,180
  series, VictoriaLogs ingesting from **both** `cluster=homelab` and `cluster=vps`
  (`instance=prod-podcast`), GlitchTip ingest working (581 events in 6h). Zero error events
  during the repair — the good kind of quiet.

---

## 5. If you are picking this up cold

1. Read §2. The secrets-persistence issue is the thing most likely to bite you.
2. Before believing any green signal from prod tooling, ask what it actually measured. Three
   separate checks lied tonight, and two of them were ones I had just written.
3. `scripts/tools/rehearse_gateway_key_gate.sh` runs the deploy gate's logic against a real
   LiteLLM gateway — use it before changing that gate.
4. `make check-prod-secret-staging` must stay green. It exists because the same bug recurred
   four times.
