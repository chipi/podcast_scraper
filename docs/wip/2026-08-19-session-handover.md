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
  — **superseded 2026-08-20 by #1686.** Severity now tracks RECOVERABILITY, not intent: an
  in-flight retry emits no Sentry event at all, a recovered summary emits nothing, and a
  summary that is genuinely LOST is reported at `error`. Net fewer events than #1632
  removed. This line stays as the record of what was decided on 08-19.
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

---

## 6. Issue ledger — what to close, what to work

### A. Close when `fix/cleaner-and-open-items` merges
The fix is written, tested and committed; it is simply not on main yet. **Do not close before
merging** — the issues are the only thing tracking that this work exists outside one branch.

| Issue | Fixed by | Evidence |
|---|---|---|
| #1641 #1642 #1643 #1644 #1645 | `c6893892` | one bug, five reports. The union of sponsor spans merged transitively and covered 86% of a transcript; 6 of 36 fixture episodes retained 12–16%. Regression test verified red-then-green against the fixture that reproduces it. |
| #1634 #1635 #1636 #1637 #1638 #1639 | `0ded2202` + `f6c77fcd` (already on main) | one incident, six reports, and they **predate their own fix**: `sha-1c6b3de` is 2026-08-11, the allowlist→denylist inversion landed 2026-08-17. What remained was `terminal_message` not naming the budget — which is the entire content of #1639. |
| #1632 | `ee06ac2b` | the retry-then-degrade the issue asks for already existed. The defect was that a by-design degradation reached GlitchTip at `error` severity, so triage filed it as a bug. |

### B. Verified done this session — close after a last look
- **#1676** — "prod points at the homelab gateway, not the prod one". D4 pins `litellm_api_base`
  onto the box each deploy, and tonight's probe returned **HTTP 200 against
  `http://100.124.111.115:4001/v1`** with `sha256:3b88f1c6ee41` matching the gateway's live key.
  The ADR-142 regression this tracked is closed in fact; confirm the wording matches.
- **#1655** — "relabel and re-derive the episodes damaged by #1646". The GI repair took
  placeholders 110 → 0 and healthy GI 568 → 678, independently verified. Check the issue's exact
  scope before closing — if it also covers the preprocessing damage, it closes with the reprocess.
- **#1657** (epic, corpus integrity) — the deploy-then-repair arc is done. Close once the
  32-episode reprocess is verified with `verify_recent_runs`.

### C. Genuinely open, worth picking up
- **#1679 — audio archive built but not enabled in prod.** This is the answer to the corpus
  being **46.5 GB of audio out of 48 GB**. `archive backfill` was built (session task #16) and is
  not wired. Until it is, recoverable audio keeps expiring and every backup decision is shaped by
  bytes that should not be on that disk.
- **#1677 — stack tests prove less than they appear to.** Problem statement only; solution
  deliberately deferred. The DGX-profile idea was raised as one option.
- **#1687 + #1688–#1692** — the operator API epic filed tonight. Start with **#1690**.

### D. Not issues, but known work
- **Secrets persistence** (§3) — the real fix for tonight's outage. Needs root on the box.
- **Tier 2 quota testing is now UNBLOCKED.** It was blocked all session because the homelab
  LiteLLM was down with Docker; both are back and verified (`/v1/models` → 200,
  model `homelab-flash-0731`). The plan: mint a key with `max_budget`/`rpm_limit` via
  `POST 127.0.0.1:4001/key/generate`, drive it into a **real** 429/403, and prove the pipeline
  fails over rather than hard-stopping. That is the one thing the #1634–#1639 unit tests cannot
  prove — they assert against a fixture of litellm's error, not litellm itself.
- **The original #43 question is still unanswered**: can search actually run inside the
  `pipeline-llm` image? Docker died before it could be tested and nobody has retried since it
  came back. `make index-two-tier-docker` is the target.
- **Audit findings** (`docs/wip/2026-08-18-tests-and-docs-audit.md`): 3 tests that pass whether
  the code works or not; 12 declared-but-unused pytest markers (so `pytest -m golden` silently
  selects nothing); and `make lint` + `make format-check` should be one target, because running
  either alone reads as green while the other is red — which I then did, twice, in one day.
- **`verify_recent_runs`** must be run against the 32-episode reprocess. `attempts=0` is
  ambiguous between "never preprocessed" and "served from cache", so only a positive assertion
  (`attempts >= 1`, `completed == attempts`) distinguishes a real repair from a no-op.

---

## 7. The 32-episode reprocess did NOT work — read this before re-running it

**Run 32193448098, 2026-08-18 22:36Z → 04:41Z (365 min, killed by the 360-minute ceiling).**

It consumed six hours, made **1,600+ gateway calls with a 100% success rate** (zero 429s, zero
auth errors, zero 5xx), and **changed nothing**:

```
BEFORE                                   AFTER (audit run 32216677478)
runs DAMAGED : 2                         runs DAMAGED : 2
  run_1ebba1af  16 ep, 16 att, 15 done     run_1ebba1af  16 ep, 16 att, 15 done   (identical)
  run_d4405a87  16 ep, 16 att, 14 done     run_d4405a87  16 ep, 16 att, 14 done   (identical)
worklist: 32 episodes                    worklist: 32 episodes                   (identical)
```

The only movement anywhere: *"runs that attempted NO preprocessing at all"* went **9 → 8**. So
something landed, but nothing that removes an episode from the worklist.

### What was ruled out during the run
- **Not ingesting new episodes.** `catalog_episode_count` held at 678 for the whole run, feeds
  at 14, month histogram unchanged. The command was correctly scoped:
  `--reprocess-episode-ids /app/output/preprocessing_repair_worklist.txt --no-transcript-cache
  --litellm-api-base http://100.124.111.115:4001/v1`.
- **Not a credentials failure.** Every gateway call returned 200.
- **Not a budget stop.** A $25 exhaustion appears as 429s; there were none.
- **Not slow because the episodes are long.** They ARE long — the damaged feed averages 88 min
  vs 51 min corpus-wide (n=49, median 85, max 171) — but length does not explain zero output.

### The lead worth following first
**Task #33 of this session: "`--reprocess-source` is inert under corpus layout —
`make redo-diarization` silently does nothing."** We have already found one reprocess path that
runs, burns resources, reports success and writes nothing. This has the identical signature.
The `--reprocess-episode-ids` path may be inert for the same reason.

### Do this, in this order
1. **Diagnose, do not re-run.** Read the `--reprocess-episode-ids` path against the corpus
   layout the way #33 was diagnosed: where does it write, and does anything read it back?
2. **Test on ONE episode** with an explicit `timeout-minutes`. Never dispatch 32 blind again.
3. **Only then batch** — and set a real timeout; `reprocess-prod.yml` currently declares none and
   silently inherits 360 minutes, which is how we discovered the ceiling.

### Observability gaps this exposed, all feeding #1687
- GitHub withholds logs until completion, and the kill left the container's stdout unflushed, so
  the workflow log contains **no per-episode output at all** — only runner cleanup.
- Progress could not be measured from run timestamps (the repair rewrites in place, so run dir
  names never change) nor from the episode API (no preprocessing state field).
- The only working progress signal all night was **gateway call volume in VictoriaLogs** — and it
  showed a healthy, busy pipeline producing nothing. Success rate is not progress.
