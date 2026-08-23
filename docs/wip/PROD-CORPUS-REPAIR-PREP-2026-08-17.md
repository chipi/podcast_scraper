# Production corpus repair — prep sheet (2026-08-17)

**What this is:** the concrete pre-flight for running
[CORPUS_INTEGRITY_REPAIR_RUNBOOK.md](../guides/CORPUS_INTEGRITY_REPAIR_RUNBOOK.md) against the
**production** corpus. The runbook is the procedure; this sheet is what must be true before it
starts, what is already proven, and what is still unknown.

Written after #1661 merged (`f6c77fcd`) + hotfix `7127e396`.

---

## Operator decisions already made

| Decision | Value |
| --- | --- |
| **Step 6 ASR budget** | **APPROVED** — spend on Deepgram, or run ASR on DGX. 2026-08-17. |
| Missing ffmpeg | FATAL, never a silent degrade (#26) |
| Missing optional ML package | DEGRADES, loudly, with a ledger row (#1661) |
| Aged-out episode fix | PARKED — measured, not needed (see below) |

---

## Preconditions — ALL must hold before step 1

> **Updated 2026-08-18.** Status below is measured, not assumed. Items marked DONE were
> verified; the rest are open. See "What changed on 2026-08-18" at the end for the full log.

| Precondition | Status |
| --- | --- |
| `DEEPSEEK_API_KEY` staged for prod | **DONE** — verified non-empty by deploy-prod's pre-flight |
| `PROD_LITELLM_API_BASE` set | **DONE** — `prod` env var, `http://100.124.111.115:4001/v1` |
| Read-only measurement path on prod | **DONE** — `inspect-prod-corpus.yml` (never yet run against prod) |
| Container images published to GHCR | **DONE** — `sha-62cc3a4`, api + viewer + pipeline-llm |
| Merged code deployed to production | **DONE** — prod runs `sha-62cc3a4` |
| Gateway ROUTING (D4) | **DONE** — deploy pins the override onto the box each run |
| Gateway AUTH (D5) | **DONE** — container -> prod gateway returns **HTTP 200** |
| Corpus snapshot/backup taken | **OPEN** — workflow fixed, no successful snapshot yet |
| Image carries `git_sha` | **UNKNOWN** — see below; not disproved, not shown |

### The one thing that will silently undo the above

`LITELLM_PROJ_PODCAST_PROD_KEY` (GH secret) still holds the OLD homelab key. The working key
was minted on the gateway and written to `/srv/podcast-scraper/.env` directly, because CI
cannot write a GitHub secret and the repo is PUBLIC — printing a live credential would
publish it in an Actions log.

**Therefore: do not run `deploy-prod` until that secret is updated.** `deploy-prod` renders
`.env` from it and would revert the key, putting D5 back to 401. Update it with the value of
`LITELLM_API_KEY` in `/srv/podcast-scraper/.env`, from a shell.

### On `git_sha`

`/api/health` reports `git_sha: "unknown"`, but that field sits inside `corpus_produced_by`
next to `produced_at: 2026-08-14` — it describes the CORPUS, not the running image, and has
not changed because no pipeline run has happened since the deploy. It says nothing about #30
either way. The real test is a pipeline run's manifest.


- [ ] **The merged code is deployed to production.** `f6c77fcd` + `7127e396` are on `main` but a
      deploy has NOT been run. Without it production is still running the pre-fix pipeline and a
      repair would be pointless — worse, it would re-damage what it repairs.
      Deploy the three planes IN ORDER, each `workflow_dispatch` with its own confirm
      string: `deploy-prod.yml` (`PROD_DEPLOY`) -> `deploy-operator.yml` (`OPERATOR_DEPLOY`)
      -> `deploy-player.yml` (`PLAYER_DEPLOY`). Only the first is needed for the corpus
      repair — it carries the api + pipeline images. The `deploy-all-prod.yml` orchestrator
      was DELETED 2026-08-18: it required a second PAT (`DEPLOY_ORCHESTRATOR_PAT`, never
      staged) purely to chain three dispatches, shipped inside an unrelated PR, and failed
      on its first-ever run.
- [ ] **The image carries `git_sha`.** ADR-132's exact-code backstop only exists if the image was
      built with the `GIT_SHA` build arg (#30). Verify inside the deployed image before trusting
      any provenance the repair writes.
- [ ] **`DEEPSEEK_API_KEY` is set** (task #23). Every configured failover ladder points at the
      deepseek tier; without the key the ladder detects failures and recovers nothing. A startup
      pre-flight prints `FAILOVER LADDER BROKEN` at every run start until it is set. Compose
      already forwards the variable — only the value is missing.
- [ ] **A corpus snapshot/backup exists.** Steps 3 and 6 write in place.

---

## Pre-flight measurements — read-only, run these FIRST

These answer the questions nobody can answer from here. None of them writes anything.

**Run them with `inspect-prod-corpus.yml`, not by hand.** Added 2026-08-18. Until then these
had no execution path on prod at all: `reprocess-prod.yml` was the ONLY workflow that ran corpus
tooling on the box, and it is the expensive WRITE step — so the cheap measurements needed a human
shell while the money step was one click. That is why the two questions below stayed UNVERIFIED
for the whole epic.

```
gh workflow run inspect-prod-corpus.yml --repo chipi/podcast_scraper \
  -f checks=all -f write_worklist=true
```

Reports land in the run summary, so a baseline is an artifact attached to a run rather than
terminal scrollback. Findings do NOT fail the workflow — findings are the expected state before a
repair. The equivalent by hand, if you have a shell on the box:

```bash
make corpus-gi-integrity-check   CORPUS_DIR=<prod-corpus>   # step 1 baseline
make corpus-preprocessing-audit  CORPUS_DIR=<prod-corpus>   # step 2 baseline
```

Record both verbatim. Steps 8–9 compare against them, and a count that does not move is the
failure mode the whole runbook exists to prevent.

**Open questions these settle:**

| Question | Current status |
| --- | --- |
| Are ~112 placeholders still in production? | **UNVERIFIED.** The figure predates the epic. Step 1 answers it. |
| How much of production was transcribed from unpreprocessed audio? | **UNVERIFIED.** The 60 % figure is from a *local pre-fix* corpus (9 of 15 runs), not production. Step 2 answers it. |
| Will step 6 silently no-op on aged-out episodes? | **MEASURED, no.** 9/9 of the damaged work-list still served by their feeds; archives run 71–2950 items. Re-run `scratchpad/check_feeds.py` against the *production* work-list before step 6 — production may hold older episodes. |

---

## What is proven, and where

Validated locally on real corpora with the merged code:

| Runbook step | Evidence |
| --- | --- |
| 1 — GI integrity gate | PASS on `pipeline-run/corpus-out`: 40 corpus members from 76 metadata files — the membership rule correctly excluded 36 superseded runs |
| 2 — preprocessing audit | Found **9 damaged runs** on `podcast-acceptance-corpus` (the pre-fix corpus) |
| 4 — work-list | 9 episode ids written, production-shaped (UUIDs + `substack:post:` ids) |
| 3 — `gi-repair` | Proven earlier in the epic (1 → 4 insights, gate FAIL → PASS). **Cannot be re-rehearsed locally: zero placeholders exist in any local corpus.** |
| 6 — re-transcribe | **NOT rehearsed.** Needs ASR; no whisper locally and it costs money. First real exercise will be production. |
| 7–9 | Not rehearsed — depend on 6 |

---

## Step 6 — the command, and the two traps in it

```bash
podcast-scraper --config <profile> --feeds-spec <corpus>/feeds.spec.yaml \
  --output-dir <corpus> --skip-existing --single-feed-uses-corpus-layout \
  --no-transcript-cache \
  --reprocess-episode-ids <corpus>/preprocessing_repair_worklist.txt
```

1. **`--no-transcript-cache` is not optional.** The cache key is the original media hash plus
   `preprocessing_fingerprint(cfg)`, and that fingerprint is computed from *config* — it reads
   `pp=on` whether preprocessing succeeded or fell back to raw audio. Neither key component
   changes between the damaged run and the repair run, so without this flag step 6 scores a cache
   hit and re-serves the exact transcript it was launched to replace, and step 8 goes green on
   unrepaired data. (Entries written *since* #1661 are safe — a run that falls back to raw audio no
   longer writes a cache entry at all. This flag covers everything written before.)
2. **`--single-feed-uses-corpus-layout` is required** or cross-run resolution never fires and every
   episode reports "no transcript".

`--reprocess-episode-ids` implies `--reprocess-existing-only`. Before that implication existed, a
one-episode work-list preprocessed **12 unrelated episodes** before being killed.

---

## Step 8 — a green audit is NOT sufficient

The preprocessing audit's damage rule is `completed < attempts`, and it reports `attempts == 0` as
*not damaged* — correctly, since a run that never attempted preprocessing damaged nothing. But a
step-6 run served entirely from cache **also** records `attempts: 0`. It is indistinguishable from
healthy while having repaired nothing.

So assert positively, against the run dirs step 6 created:

```bash
find <corpus> -name metrics.json -newermt '-1 hour' -exec sh -c '
  echo "$1: $(jq -c "{attempts: .preprocessing_attempts, completed: .preprocessing_count,
                      transcribed: .transcribe_count}" "$1")"' _ {} \;
```

Expected: `attempts >= 1`, `completed == attempts`, `transcribed >= 1`. All zeros means the cache
served it.

---

## Known limits going in — say these out loud rather than discover them

- **Step 6 has never been run.** Its first execution will be against production. Use
  `--max-episodes` for a cautious first pass.
- **A reprocess does NOT fix placeholder episodes.** Four flag combinations were rehearsed and
  every one skipped them: the skip predicates key on file *presence* and never look at GI. That is
  what `gi-repair` (step 3) exists for, and why it rewrites in place.
- **`make corpus-placeholder-check` is not a valid exit criterion.** It asks only "is the bad
  string absent?", which PASSES on a corpus whose artifacts were deleted and never regenerated.
  Use `corpus-gi-integrity-check`.
- **`metrics.json` is run-level.** A one-episode run attributes exactly; a multi-episode run can
  only say "this run has damage". Per-episode attribution needs the #22 ledger row, which by
  construction exists only on runs made *after* that change — never on the damaged ones.
- **Zero-insight artifacts are legal now.** 112 placeholders quietly becoming 112 *empty* artifacts
  would satisfy a naive check while having re-derived nothing. Step 1 reports that count
  separately — read it.

---

## What changed on 2026-08-18

Everything here was measured this session. Where an earlier belief turned out wrong, the wrong
belief is kept alongside the correction — the wrong ones cost hours and would otherwise be
re-derived.

### The deploy was blocked by something nobody was looking at

**No container image had been published since 2026-08-11.** `stack-test.yml` triggers on
`workflow_run` gated on **Python application succeeding**, and that workflow had been failing
since the `#1661` merge. So every Stack test run since was `skipped`, not queued — and
`deploy-prod.yml` resolves its image to *"the NEWEST `sha-<7>` tag actually PUBLISHED to GHCR —
NOT github.sha"*.

**The trap:** deploying in that state would have **succeeded** while shipping the Aug 11
pre-fix image, with every step green. The same silent-wrong-result class the epic exists to
remove, sitting in the deploy path. Do not deploy until Stack test has succeeded on a commit at
or after the fix.

Root cause of the red: the e2e **coverage** gate, not a failing test. Runs read
`361 passed, 0 failed` next to `Total coverage: 38.09%` against a `--cov-fail-under=39`.
`coverage-unified` was a pure cascade — no `coverage-data.e2e` artifact, because the job exits
before writing it. Fixed by adding 17 e2e tests (**38.10% → 39.40%**), not by lowering the
threshold: the Makefile states the policy (*"no subtree omit […] add pytest E2E until the gate
passes"*) and all 13 new modules already sat at **90.6% unit** coverage, so the gap was
end-to-end coverage, not untested code.

Worth knowing: that gate is a whole-package ratio against a fixed floor, so it decays whenever a
module lands that e2e does not drive. It cleared by 0.33pp before `#1661` and broke on the first
sizeable merge; it now clears by 0.40pp. It will happen again.

### LiteLLM routing — the repair would have billed the wrong gateway

`reprocess-prod.yml` invokes the CLI as `--config <profile>`, which **never reads the box's
`viewer_operator.yaml`** — the only place prod's gateway override lived (D4). `cloud_balanced.yaml`
pins the homelab gateway, and profiles are generated (ADR-112) so editing the pin is not
sanctioned. Root cause: LiteLLM was the **one** provider namespace with no `--*-api-base` flag
while eight siblings had one.

Fixed in `#1676`: added `--litellm-api-base`, a `litellm_api_base` input on `reprocess-prod.yml`
(empty warns and uses the profile's homelab pin), a deploy-time re-render of the override into
`viewer_operator.yaml` (D4), and a post-deploy check that the key authenticates **at the
configured base** (D5).

**You do not need to pass `litellm_api_base` at dispatch.** Corrected 2026-08-18 after the
operator asked "didn't we do that already?" — they had. `deploy-prod.yml` read
`vars.PROD_LITELLM_API_BASE` while `reprocess-prod.yml` read only its own dispatch input and
ignored the variable, so the gateway had to be retyped every run and forgetting it billed
homelab silently. That made the expensive outcome the default. Now the input OVERRIDES for a
single run, the variable is the default, and only when NEITHER is set does it fall back to the
profile pin — with a warning naming the variable to set.

### Prod SSH — and what was actually wrong

Operator SSH to prod timed out. Diagnosed wrong twice before measuring:

1. *sshd or host firewall* — wrong.
2. *fail2ban ban* — plausible (the jail is real: `maxretry=5, findtime=10m, bantime=1h`) but
   **disproved**: a 1h ban expires and the failure did not.

Actual cause: the mini is a **tagged** node (`tag:homelab-host`), and a tagged device has no user
owner, so `autogroup:admin -> tag:prod:22,...` never matched it. It kept exactly its tag's ports —
443 and 4001 answered, 22 and 80 timed out.

A third belief was also wrong: this was **not** a regression. The mini has been tagged since
2026-07-22 and its `tag:prod` grants (2026-08-12) never included `:22`; Tailscale SSH does not
cover it either (the `ssh` block's only dst has ever been `tag:dgx-llm-host`). The recollection of
"SSH to prod from the mini" was almost certainly the `:443` HTTPS API granted the day before the
first corpus load, or a session from the untagged admin laptop.

`:22` was granted 2026-08-18. **The durable fix is `inspect-prod-corpus.yml`** — it runs as
`tag:gha-deployer`, so the baselines no longer depend on which device the operator is sitting at.

Also: **`additional_authorized_keys` cannot add a key to the running box.** `main.tf` has
`lifecycle { ignore_changes = [user_data, ssh_keys] }`, so a change there is a silent no-op that
only affects a freshly built server. That guard is deliberate — a rotated key once cascaded into
`ssh_keys = [...] # forces replacement` and **destroyed prod** (#839).

### `git_sha` is `unknown` in production, confirmed live

`GET /api/health` on prod returns:

```json
"corpus_produced_by": {"code_version":"2.7.0.dev0","git_sha":"unknown","produced_at":"2026-08-14T10:16:31Z"}
```

Two facts: prod is running **pre-merge** code (`produced_at` predates the merge), and **#30 is
live** — ADR-132's exact-code backstop does not exist on the box. **After deploying, re-check
that `git_sha` is populated.** If it still reads `unknown`, nothing the repair writes can be tied
to the code that produced it, which is the provenance the epic was about.

### Updated order of operations

```
1. Stack test succeeds -> sha-<7> images in GHCR        (prerequisite, ~1-2h)
2. backup-corpus-prod.yml                               (steps 3 and 6 write in place)
3. deploy-prod.yml  confirm=PROD_DEPLOY                 (approval; watch "Gateway auth (D5)")
   then deploy-operator.yml (OPERATOR_DEPLOY), then deploy-player.yml (PLAYER_DEPLOY)
4. verify /api/health now reports a real git_sha
5. inspect-prod-corpus.yml  checks=all write_worklist=true   (the baselines; READ the work-list)
6. gi-repair --dry-run, then for real                   (placeholders; no ASR)
7. reprocess-prod.yml  confirm=PROD_REPROCESS
   -- litellm_api_base can stay EMPTY; it defaults to vars.PROD_LITELLM_API_BASE
   -- bounded first pass; step 6 has never been executed anywhere
8. inspect-prod-corpus.yml  verify_recent_runs=true     (positive assertion, not a green audit)
```

**D5 outcomes worth reading precisely:** `200` = the container reaches prod's own gateway.
`000` = it cannot route to the host tailnet IP (fix: an `extra_hosts` entry mirroring the
`homelab` pattern). `401` = the key is valid only at homelab's gateway — which is D5's original
finding, that the app "worked only by way of homelab".
