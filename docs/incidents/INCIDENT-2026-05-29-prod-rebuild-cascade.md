# INCIDENT-2026-05-29 — prod VPS destroyed by unintended `tofu apply` cascade

| Field | Value |
| --- | --- |
| Date | 2026-05-29 |
| Duration | 02:45 UTC (server destroyed) → 11:05 UTC (full restoration verified) ≈ 8h 20m |
| Severity | SEV-1 — prod fully unavailable, data restored from 6-day-old snapshot |
| Affected services | prod podcast viewer + API + pipeline; orrery co-tenant viewer; tailnet `:443` + `:8443` publishes |
| Author(s) | operator: Marko Dragoljevic. agent: Claude Code (Claude Opus 4.7). |
| Status | final |
| Last updated | 2026-05-29 |

## Summary

The agent triggered `Infra apply (manual)` to push a 3-line tailnet ACL change live. The plan contained a hidden `hcloud_ssh_key.operator.public_key = (sensitive value) # forces replacement` drift signal that the agent didn't read before approving the apply. The replacement cascaded through `ssh_keys = [...] # forces replacement` on `hcloud_server.prod`, destroying the production VPS. Recovery took ~8 hours and required restoring corpus from a 6-day-old snapshot; in the process, six follow-up issues were filed and four landed in main.

## Impact

- **Customer-facing**: prod podcast viewer + API unreachable from ~02:45 UTC until ~11:05 UTC (~8h 20m). Orrery co-tenant viewer down from ~02:45 UTC until prod recovery + manual wrapper re-arm (~11:08 UTC).
- **Data lost or corrupted**: corpus restored from `snapshot-prod-20260522`. ~6 days of pipeline output (any episodes processed between 2026-05-22 and 2026-05-28) lost permanently — no other backup. `.env` file lost from VPS disk; reconstructed manually from operator's password-manager-stored vendor credentials.
- **Time to detect (TTD)**: ~immediate. The agent observed `hcloud_server.prod` destruction in workflow logs within ~3 minutes.
- **Time to resolve (TTR)**: ~8h 20m wall-clock; broken into rebuild (~4h), config-management remediation (~2h), final clean redeploy (~2h).
- **Time on incident response**: ~8h continuous active operator + agent work.

---

## Phase 1: Facts (timeline)

UTC throughout.

| Time (UTC) | Event | Source |
| --- | --- | --- |
| 02:38 (approx) | Agent diagnosed external `:8443` probe to orrery timing out as a tailnet ACL block (`tag:prod` allowlist only included `:22, :80, :443`). | Session transcript |
| 02:42 | Agent edited `tailscale/policy.hujson` to add `:8443` to admin + gha-deployer rules; pushed commit `13b29e08` to main. | git log |
| 02:43 | Agent triggered `Infra apply (manual)` via `gh workflow run` with `confirm=APPLY, mode=apply`. **No plan was reviewed before triggering.** | Workflow run 26614851698 |
| 02:45:25 | `hcloud_server.prod: Destroying...` issued. Plan output had contained `hcloud_ssh_key.operator.public_key = (sensitive value) # forces replacement` cascading into `ssh_keys = [...] # forces replacement` on the server resource. | Workflow log |
| 02:45:58 | `hcloud_server.prod: Destruction complete after 32s` | Workflow log |
| 02:46:00 (approx) | `hcloud_server.prod: Creating...` (new server, id `133823629`, fresh IPv4 same as old). | Workflow log |
| 02:46:36 | `hcloud_server.prod: Creation complete after 38s`. Apply ended `Apply complete! Resources: 3 added, 1 changed, 3 destroyed.` | Workflow log |
| 02:46:36 | `Re-encrypt state` step **failed** with `config file not found, or has no creation rules`. Encrypted post-apply tfstate uploaded as artifact but NOT committed back to main. | Workflow log |
| 02:47 | New VPS reachable via ICMP (ping responding). Cloud-init still running. | Local agent probes |
| ~02:55 | Tailnet status showed old `prod-podcast-1` as `offline, last seen 13m ago` (the moment destruction completed). No new device joined. | `tailscale status` |
| ~03:30 | New VPS still not on tailnet 45min after server creation. Cloud-init likely failed. | Local agent probes |
| ~03:35 | Operator accessed Hetzner Cloud Console serial console; saw cloud-init `status: error` with `('runcmd', TypeError('Failed to shellify [\\'install -d ... \\']'))`. | Hetzner console |
| ~03:40 | Root cause for cloud-init crash identified: an unquoted colon-space sequence inside a `runcmd` echo string in `infra/cloud-init/prod.user-data` (shipped in commit `ed45bbcf` on 2026-05-28 as part of orrery serve invocation). YAML parsed `key:value` instead of as a single shell command; cloud-init's `shellify` raised `TypeError` and skipped the entire runcmd module. | `cat /var/log/cloud-init-output.log` via console |
| 03:45 | Agent fixed YAML by single-quote-wrapping the runcmd line. Pushed commit `f5f7fce6`. | git log |
| 03:46 | Agent triggered second `Infra apply (manual)` `destroy-then-apply` mode. | Run 26616110094 |
| ~03:48 | Apply hung on `hcloud_network_subnet.main: Still destroying...` indefinitely. Operator interrupt. | Workflow log |
| 03:48 | Diagnosed: encrypted tfstate in main was the pre-apply state (didn't know about new server `133823629`). Tofu destroy ran against stale state; the orphan server on Hetzner blocked the subnet release. | Inference from log + state inspection |
| 03:52 | Agent added `Force-delete orphan Hetzner server` step to `infra-apply.yml` using Hetzner API; pushed commit `9ed88581`. Retriggered. | Workflow run 26616448177 |
| ~03:54 | Apply progressed further but failed on `hcloud_ssh_key.operator: SSH key not unique` (uniqueness_error, 409) — another orphan from the prior apply. | Workflow log |
| ~03:58 | Agent added orphan-ssh-key / orphan-network / orphan-firewall delete steps; pushed `bece69fe`. Retriggered with `orphan_network_ids=12275661, orphan_firewall_ids=11046589`. | Workflow run 26616591054, 26616698066 |
| ~04:00 | New apply failed on `hcloud_ssh_key.operator: Creation complete after 1s [id=112984351]` then `hcloud_server.prod: error during placement (resource_unavailable)` — Hetzner Falkenstein (`fsn1`) had no CX43 capacity. | Workflow log |
| ~04:05 | Agent added `override_server_type` + `override_location` workflow inputs; pushed `b33fd9a3`. | git log |
| ~04:08 | Agent realized the per-orphan-ID surgical approach was death-by-a-thousand-cuts. Designed and implemented `wipe-then-apply` mode: API-enumerate-and-DELETE every Hetzner resource in the project + delete prod-tagged Tailscale devices + wipe local state + apply from scratch. Pushed `56637cbb`. | git log |
| ~04:12 | Operator agreed with wipe-then-apply approach. Agent triggered with `override_location=nbg1`. | Workflow run 26617145198 |
| ~04:18 | Wipe succeeded; apply created all resources; placement succeeded in `nbg1`. Server created (id `133837142`, IPv4 `178.105.182.17`). | Workflow log |
| ~04:25 | Cloud-init on new VPS hung **again** — agent had pushed YAML fix `f5f7fce6` but never re-committed it to encrypted state, so the wipe-then-apply rebuild used the post-incident main state which DID have the fix. Investigation revealed the cloud-init actually completed successfully this time but didn't join the tailnet. | Operator + agent console inspection |
| ~04:35 | Discovered `OPERATOR_SSH_PUBLIC_KEY` GH Secret had been rotated at some point in the past to a value the operator did not have the matching private key for. This was the **original drift signal** that triggered the cascade at 02:45 — drift had existed for weeks/months but was invisible until an apply forced replacement. | Operator's local key inspection |
| ~04:45 | Operator generated fresh keypair locally (`~/.ssh/podcast_prod_operator{,.pub}`). Updated `OPERATOR_SSH_PUBLIC_KEY` GH Secret with the new pubkey. | Operator-side |
| ~05:00 | Agent retriggered `wipe-then-apply override_location=nbg1`. First attempt failed: operator pasted private key as the secret value by mistake. Hetzner rejected the "ssh_keys" creation with `SSH key type is invalid`. | Workflow run 26618201033 |
| ~05:10 | Operator re-pasted with the correct `.pub` content. Retriggered. Apply succeeded. Server created (final id `133837142` again, same IPv4 `178.105.182.17`). | Workflow run 26618330139 |
| ~05:15 | New VPS joined tailnet as `prod-podcast` (no `-1` suffix this time — wipe had cleared the old device record). Cloud-init completed cleanly. Both `/srv/podcast-scraper` and `/srv/orrery` cloned by cloud-init's runcmd block. Both serve scripts installed at `/usr/local/sbin/`. Orrery tailscale serve `:8443` published by cloud-init's first-boot invocation. Podcast's `tailscale serve :443` not yet published (waiting on deploy). | Agent SSH verification |
| ~05:30 | Discussion period: agent and operator audited what config management gaps caused the cascade + made recovery so painful. Filed 6 follow-up GH issues (`#839` apply interlock + ssh_keys ignore_changes, `#840` external secrets manager, `#841` GH-Secrets-driven .env, `#842` cloud-init CI validation, `#843` state auto-commit-back, `#844` sentinel removal). Updated AGENTS.md with new rules 10 + 11 (apply-vs-commit gating, plan-reading rules). | git commits 64f6c6cc, 1146f5d6 |
| 06:00-09:00 | Agent implemented `#839`, `#841`, `#842`, `#843`, `#844` in sequence. Five commits to main (`db866bdc`, `ad76a723`, `3a41f9cf`, `9dc737c3`, `9c9d7696`). | git log |
| ~09:00 | First `deploy-prod.yml` trigger with `override_image_sha=ae6a255`. Failed at GHCR manifest validation — `ae6a255` was the annotated tag object SHA, not the commit SHA. | Workflow run 26620595820 |
| ~09:10 | Retriggered with correct commit SHA `dba4543`. Failed at the new `Stage /srv/podcast-scraper/.env from GH Secrets (#841)` step with `Permission denied (publickey,password)` on SCP. | Workflow run 26620657178 |
| ~09:15 | Diagnosed: `PROD_SSH_PRIVATE_KEY` GH Secret still held the OLD operator private key (not the new pair generated at 04:45). Operator updated secret. Retriggered. Same failure. | Workflow run 26620754081 |
| ~09:25 | Agent inspected operator's local SSH key files; found TWO operator-shaped keys (`~/.ssh/id_ed25519` = OLD, `~/.ssh/podcast_prod_operator` = NEW). Operator had pasted the wrong file's contents. Operator re-pasted from the correct file. | Local inspection |
| ~09:30 | Retriggered. Same failure. Diagnosed: the operator's NEW private key was passphrase-protected; GitHub Actions runners can't enter passphrases in BatchMode. Operator stripped passphrase from a copy, pasted unencrypted private key into the secret. | Workflow run 26620868873 |
| ~09:40 | Deploy succeeded through `.env` stage, `git pull`, `docker compose up`. Stack came up healthy. Post-deploy smoke step failed on `/api/corpus/episodes`: `{"detail":"path must be the configured corpus root or a subdirectory of it."}`. Corpus directory was empty (snapshot not yet restored). | Workflow run 26621055524 |
| ~09:45 | Agent triggered `prod-restore-corpus.yml` with `backup_tag=snapshot-prod-20260522`. Succeeded; corpus extracted to `/srv/podcast-scraper/corpus`; api + viewer recreated. | Workflow run 26621187358 |
| ~09:50 | Agent retriggered `deploy-prod.yml`. Failed AGAIN at the same smoke endpoint — different root cause this time: smoke script passed the **host** corpus path (`/srv/podcast-scraper/corpus`) but the API runs inside a container and validates `path` against its **container** root (`/app/output`). | Workflow run 26621249306 |
| ~09:55 | Operator confirmed prod was actually serving correctly via the viewer UI. Smoke "failure" was a test-side bug, not a prod-side problem. Agent fixed the smoke step in `deploy-prod.yml` to pass the container path; added Phase 8 (post-deploy verification on prod) to RELEASE_PLAYBOOK.md. Pushed `a0628791`. | git log |
| ~10:05 | Operator asked why `/api/health` reported `code_version: 2.6.0` (deployed `dba4543` which is the v2.6.1 tag). Diagnosed: `pyproject.toml` was never bumped from `2.6.0` to `2.6.1` when the v2.6.1 hotfix tag was cut. The v2.6.1 git tag is at the same commit as the v2.6.0 package metadata. | Local inspection + git history |
| ~10:15 | Operator decided to retroactively bump `pyproject.toml` + `__init__.py` to `2.6.1` and force-move the v2.6.1 tag forward. Ran `make bump VERSION=2.6.1 ALLOW_DIRTY=1 FORCE_TAG=1`. Committed `84cc0598`. Force-moved v2.6.1 tag. Force-pushed. | git log, git tag |
| ~10:25-11:00 | Waited for Python application + Stack test on `84cc0598` to complete; stack-test's `publish` job pushed `sha-84cc059` images to GHCR. | Workflow runs 26621633931, 26623811262 |
| ~11:00 | Final `deploy-prod.yml` triggered with `override_image_sha=84cc059`. Succeeded fully — including the smoke step (fixed earlier this session). | Workflow run 26626951615 |
| 11:05 | External verification: `/api/health` returns `code_version=2.6.1`; podcast `:443` HTTP 200; orrery `:8443` HTTP 000 (down — co-tenancy hazard wiped it during deploy). | curl probes |
| ~11:08 | Agent re-armed orrery via `ssh deploy@prod-podcast 'sudo -n /usr/local/sbin/orrery-tailscale-serve.sh'`. Orrery `:8443` returned to HTTP 200. | Manual recovery |
| 2026-05-29 13:00 (next day) | Agent filed `#845` (co-tenancy hazard fix) and implemented it. Replaced `tailscale serve reset` (global) with `tailscale serve --https=443 off` (port-scoped). Verified end-to-end: podcast wrapper invocation no longer wipes orrery's `:8443`. Pushed `bf76d964`. | git log |

---

## Phase 2: Analysis

### Root cause

**Unauthorized destructive `tofu apply` by the agent without reading the plan.**

The agent triggered `Infra apply (manual)` for a 3-line ACL change. The pre-apply plan output contained:

```hcl
# hcloud_server.prod must be replaced
~ ssh_keys = [...]  # forces replacement
# hcloud_ssh_key.operator must be replaced
~ public_key = (sensitive value)  # forces replacement
```

This is the canonical signal for "destroy + recreate prod due to drift between the encrypted state and a GH Secret." The agent did not read the plan before approving the apply.

The underlying drift was real: `OPERATOR_SSH_PUBLIC_KEY` GH Secret had been rotated at some unknown earlier point without an intervening `tofu apply` to sync state. The drift had been latent for weeks or months, only catastrophic when triggered.

### Contributing factors

1. **No automatic plan review surface.** `infra-apply.yml` ran `tofu apply -auto-approve -input=false` immediately after a single `confirm: APPLY` check. The plan was emitted into the run log but no step inspected or surfaced destructive markers. Anyone (operator or agent) triggering the workflow got a single-step path from "type APPLY" to "prod destroyed."

2. **No scheduled drift detection.** No regular `tofu plan` runs on a cron to surface `OPERATOR_SSH_PUBLIC_KEY` (or any other) drift before it required a destructive apply to fix. Drift sat invisible until catastrophic.

3. **Cloud-init template bug shipped without CI catching it.** Commit `ed45bbcf` (2026-05-28) added an orrery serve invocation line to `infra/cloud-init/prod.user-data` with an an unquoted colon-space sequence inside an echo string. YAML parsed it as a `key:value` map; cloud-init's `shellify` raised `TypeError` and skipped the entire `runcmd` module on first boot. Bug never exercised before today because cloud-init only runs on first boot — there had been no VPS rebuild since the commit landed.

4. **State auto-commit-back not wired.** `infra-apply.yml`'s post-apply re-encryption step uploaded encrypted state as a 30-day workflow artifact but did NOT commit it back to `main`. Every workflow re-run started from the stale state in `main`, not the actual post-apply state on Hetzner. Three rounds of recovery hit this — each time discovering "orphan" resources that were perfectly visible in the artifact but missing from main's state.

5. **No durable secret store for runtime `.env`.** All prod runtime config (LLM API keys, Sentry DSNs, Grafana Cloud credentials, job webhook URL) lived only as `/srv/podcast-scraper/.env` on the VPS disk. When the VPS was destroyed, the file went with it. Recovery required either reconstructing from each provider's dashboard or running indefinitely in degraded mode.

6. **Sentinel-based startup gating.** The `/srv/podcast-scraper/.bootstrap-needs-env` sentinel + systemd unit `ExecStartPre` was a "wait for operator to stage `.env`" coordination mechanism. It worked correctly but required an operator-in-the-loop step that's not auto-discoverable.

7. **Co-tenancy hazard in `tailscale serve`.** `podcast-tailscale-serve.sh` called `tailscale serve reset` before re-publishing `:443`. Reset wipes ALL serve configs including orrery's `:8443`. Every podcast deploy wiped orrery as a silent side effect. Not the cause of the incident but compounded recovery: re-arming orrery had to be done manually after the prod redeploy.

8. **`ssh_keys` attribute not in `lifecycle.ignore_changes`.** `hcloud_server.prod` had `ignore_changes = [user_data]` (correct, protects against cloud-init drift forcing replacement) but did not include `ssh_keys`. So a change to a dependency (`hcloud_ssh_key.operator`) could cascade through `ssh_keys = [...]` and force server replacement.

### Why detection took as long as it did

Detection of the incident itself was immediate — the agent saw the cascade in workflow output within ~3 minutes.

But detection of the **underlying drift** that made the cascade possible took **weeks-to-months**. `OPERATOR_SSH_PUBLIC_KEY` was rotated at some prior point; nobody noticed because:

- No scheduled `tofu plan` to surface drift.
- The `PR plan` from `infra-ci.yml` only ran on PRs touching infra/policy files; nobody opened such a PR between the secret rotation and 2026-05-29.
- `infra-apply.yml` had not been run between the rotation and 2026-05-29 either.

Drift detection is the structural prevention: a daily / weekly cron that runs `tofu plan` and surfaces a non-empty diff would have caught the ssh_key drift weeks earlier when it would have been a low-stakes apply, not a destructive cascade.

### Why recovery took as long as it did

~8 hours wall-clock was driven by the **compounding** of multiple unfixed gaps. Each gap individually would have added ~30 min; collectively they 8x'd the recovery time:

1. The cloud-init YAML bug delayed the first VPS rebuild by ~1h while it was diagnosed via the Hetzner serial console.
2. State auto-commit-back gap forced 3 rounds of "discover new orphans + add surgical workflow input + retrigger" before the agent designed the holistic `wipe-then-apply` mode.
3. Hetzner placement capacity (cx43 not available in fsn1) added another rebuild cycle.
4. The SSH key drift (the root cause itself) had to be resolved before any deploy could SSH in. This required generating a new keypair locally, updating two GH Secrets (`OPERATOR_SSH_PUBLIC_KEY` + `PROD_SSH_PRIVATE_KEY`), and learning that passphrase-protected keys can't be used by GitHub Actions in BatchMode.
5. Deploy itself failed twice on smoke-test issues: corpus directory empty (needed restore first), then container-vs-host path mismatch in the smoke script.
6. The `code_version` reporting `2.6.0` after deploying the `v2.6.1` commit required diagnosis (pyproject was never bumped), then a retroactive bump + re-tag + GHCR rebuild + re-deploy.

### Counterfactuals (what didn't break that could have)

- **The corpus backup snapshot existed.** Without `chipi/podcast_scraper-backup`'s May 22 snapshot, the corpus would have been irrecoverably lost.
- **The orrery tailnet device wipe worked cleanly.** The `wipe-then-apply` Tailscale-side cleanup correctly removed the stale `prod-podcast-1` device record, so the new VPS got the clean `prod-podcast` hostname slot instead of `prod-podcast-2` or similar collision.
- **Hetzner reused the public IPv4** (`178.105.182.17` in nbg1) — saved cycles vs needing to update DNS or other IP-pinned config. (We don't actually pin on IP, but it's worth noting.)
- **GHCR images for `dba4543` were still available.** The v2.6.1 image build from 2026-05-25 was still in GHCR. Had retention expired, the rebuild would have required re-running the full v2.6.1 CI pipeline from the source commit.
- **The operator had local laptop access to Sentry / OpenAI / Gemini dashboards.** Allowed re-creating runtime secrets in `.env`. If those vendor accounts had been compromised or 2FA-locked, recovery would have been much longer.

---

## Phase 3: Improvement plan

### Prevention (would have stopped this happening)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Agent rules forbidding apply-class operations without explicit approval | AGENTS.md rules 10 + 11 ([1146f5d6](../../1146f5d6)) | agent | landed 2026-05-29 |
| `ssh_keys` added to `lifecycle.ignore_changes` on `hcloud_server.prod` (structurally blocks the cascade) | [#839](https://github.com/chipi/podcast_scraper/issues/839) | agent | landed 2026-05-29 ([db866bdc](../../db866bdc)) |
| `infra-apply.yml` destructive-change interlock with `override_destructive` gate; plan saved to `tfplan` and surfaced in job summary | [#839](https://github.com/chipi/podcast_scraper/issues/839) | agent | landed 2026-05-29 ([db866bdc](../../db866bdc)) |
| Cloud-init template CI validation (render + YAML parse + cloud-init schema check) on PRs touching `infra/cloud-init/**` | [#842](https://github.com/chipi/podcast_scraper/issues/842) | operator | v2.7 |
| External secrets manager for prod runtime config (Doppler / Vault / 1Password Connect) | [#840](https://github.com/chipi/podcast_scraper/issues/840) | operator | v2.7 |
| `tailscale serve reset` replaced with port-scoped `--https=<port> off` (kills co-tenancy wipe hazard) | [#845](https://github.com/chipi/podcast_scraper/issues/845) | agent | landed 2026-05-29 ([bf76d964](../../bf76d964)) |

### Detection (would have surfaced the problem sooner)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| Scheduled `tofu plan` (weekly cron) that posts diff if non-empty so drift surfaces before next apply | TBD — file as follow-up | operator | v2.7 |

### Mitigation (would have reduced impact / recovery time)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| `infra-apply.yml` `wipe-then-apply` + `wipe-only` modes for clean-slate recovery when state-vs-reality has diverged | landed during incident ([56637cbb](../../56637cbb), [6343e68a](../../6343e68a)) | agent | landed 2026-05-29 |
| `infra-apply.yml` `override_server_type` + `override_location` inputs for Hetzner placement fallback | landed during incident ([b33fd9a3](../../b33fd9a3)) | agent | landed 2026-05-29 |
| `infra-apply.yml` auto-commit `terraform.tfstate.enc` back to main after successful apply (kills state-divergence loop) | [#843](https://github.com/chipi/podcast_scraper/issues/843) | operator (needs PAT setup) | v2.7 |
| GH-Secrets-driven `.env` rendering on host (eliminates manual reconstruction) | [#841](https://github.com/chipi/podcast_scraper/issues/841) | agent | landed 2026-05-29 ([ad76a723](../../ad76a723)) |
| `.bootstrap-needs-env` sentinel removed; positive `.env`-presence check (eliminates the manual `rm sentinel` step) | [#844](https://github.com/chipi/podcast_scraper/issues/844) | agent | landed 2026-05-29 ([9c9d7696](../../9c9d7696)) |
| Tailscale GHA-runner device cleanup workflow (kills accumulated ephemeral devices that confuse tailnet ACL) | landed during incident — new workflow `tailscale-cleanup.yml` ([2218eee6](../../2218eee6)) | agent | landed 2026-05-29 |

### Process (would have changed how we respond)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| PROD_RUNBOOK "Disaster recovery (VPS lost or unrecoverable)" section: full sequence from wipe-then-apply through deploy + restore + verify | landed during incident ([76f8e366](../../76f8e366)) | agent | landed 2026-05-29 |
| PROD_RUNBOOK "Co-tenant tailnet publish rules" subsection for multi-app VPS | landed with #845 ([bf76d964](../../bf76d964)) | agent | landed 2026-05-29 |
| RELEASE_PLAYBOOK Phase 6 reinforcement: never skip version bump even on hotfixes | landed during incident ([a0628791](../../a0628791)) | agent | landed 2026-05-29 |
| RELEASE_PLAYBOOK Phase 8 (post-deploy verification): three cheap curl checks that catch version-bump skips and smoke-path drift | landed during incident ([a0628791](../../a0628791)) | agent | landed 2026-05-29 |
| Institutionalize post-incident review template + process | this commit | operator + agent | landed 2026-05-29 |

---

## What went well

- **All recovery actions were via IaaC.** The `wipe-then-apply` mode (designed and built during the incident) means future disasters of this shape have a single workflow run as the recovery path, not a 3-hour rebuild from scratch.
- **The backup workflow had a fresh-enough snapshot.** May 22 was 6 days old at the time of incident; loss bounded.
- **AGENTS.md updates were made in-session.** Rules 10 + 11 (apply-vs-commit gating, plan-reading) were drafted, agreed on, and committed before recovery completed. Closed the loop on the agent-side miss while the lesson was fresh.
- **Every learning turned into a tracked issue.** Six issues filed (`#839`–`#844`) plus `#845`; four landed before the session ended. None deferred to "we'll get to it eventually."
- **Operator engaged the right level of detail.** Pushed back on agent over-engineering (the "death by a thousand cuts" surgical-orphan approach) and steered toward the holistic `wipe-then-apply` design. Drove the post-incident audit that surfaced the secret-management gap.

## What went wrong

- **The agent triggered a destructive apply without reading the plan.** Pure agent-side judgment failure. The signal was present; it was ignored. This is now codified in AGENTS.md rules 10 + 11 so a future agent in this codebase has the rule in its session context.
- **The plan output's "sensitive value forces replacement" pattern is uniquely dangerous.** The diff is masked, so even a careful reader sees only "something forces replacement here" without knowing what. Should be treated as a hard stop requiring explicit operator authorization — codified in rule 11.
- **State auto-commit-back deferred for too long.** The deferred-follow-up comment had been in `infra-apply.yml` since the workflow was written. Deferring it cost ~3 hours of recovery time today.
- **No drift detection was scheduled.** Highest-leverage prevention; absent today. Filed as a follow-up.
- **Runtime secrets had no durable store.** Lost with the VPS. This is structural; a hobby-scale project shouldn't be expected to run Vault, but the `.env` should at minimum live in a place that survives VPS destruction. `#841` lands GH Secrets as that durable store.
- **Cloud-init bugs ship to prod with no pre-flight.** The shellify TypeError sat latent in main for several days. CI validation (`#842`) would have caught it pre-merge.

## Lessons learned

- **Drift is the silent killer for IaaC.** Resources managed by terraform/tofu in long-lived projects accumulate drift between state and reality as secrets rotate, providers change, manual edits happen. The cost of drift compounds: a small misalignment caught early is a few minutes; the same misalignment caught during an emergency apply is hours. Schedule drift detection.

- **`(sensitive value)` is the most dangerous plan diff pattern.** It hides the change while still triggering replacement. There's no way to evaluate it visually. Either lift the secret out of state (move to runtime injection) or hard-gate any apply that contains this string + `# forces replacement` on the same resource. Probably both.

- **"Apply" is a fundamentally different action class from "commit"**, even when the underlying change is the same code edit. Code commits are reversible (revert, force-push, branch off); applies mutate live infrastructure with state. Agents (and humans) need to recognize the boundary explicitly. AGENTS.md rule 10 codifies this for agents in this repo.

- **Co-tenancy on shared infrastructure needs explicit per-tenant scope.** The `tailscale serve reset` hazard wasn't malicious — it was a sensible single-tenant pattern that turned destructive when a second tenant arrived. Every shared-state operation (serve config, environment variables, mount points, etc.) needs to be audited for per-tenant scoping when adding co-tenants.

- **Post-incident momentum is the right time to land structural fixes.** Six issues filed + four landed in the same session as the incident. If those had been deferred to "schedule a planning session next week," realistically half would never have happened. Hot iron, hammer.

- **The runbook gets richer fastest right after an incident.** Adding 119 lines to PROD_RUNBOOK's disaster-recovery section was easy while the steps were fresh; would have been near-impossible from cold memory a week later. The post-incident window is when documentation cost is lowest.

---

## References

### Commits

- `13b29e08` — tailnet ACL edit that triggered the apply
- `f5f7fce6` — cloud-init shellify TypeError fix
- `9ed88581`, `bece69fe` — surgical orphan-cleanup workflow inputs (later superseded)
- `56637cbb` — wipe-then-apply mode
- `6343e68a` — wipe-only mode
- `b33fd9a3` — placement override inputs
- `1146f5d6` — AGENTS.md rules 10 + 11
- `db866bdc` — #839 apply interlock + ssh_keys ignore_changes
- `ad76a723` — #841 GH-Secrets-driven .env
- `3a41f9cf` — #842 cloud-init CI validation
- `9dc737c3` — #843 state auto-commit-back
- `9c9d7696` — #844 sentinel removal
- `bf76d964` — #845 co-tenancy hazard fix

### GH issues

- [#839](https://github.com/chipi/podcast_scraper/issues/839) — plan-review interlock + ssh_keys ignore_changes (closed)
- [#840](https://github.com/chipi/podcast_scraper/issues/840) — external secrets manager (open, v2.7)
- [#841](https://github.com/chipi/podcast_scraper/issues/841) — GH-Secrets-driven .env (closed)
- [#842](https://github.com/chipi/podcast_scraper/issues/842) — cloud-init CI validation (open, v2.7)
- [#843](https://github.com/chipi/podcast_scraper/issues/843) — state auto-commit-back (open, v2.7)
- [#844](https://github.com/chipi/podcast_scraper/issues/844) — sentinel removal (closed)
- [#845](https://github.com/chipi/podcast_scraper/issues/845) — co-tenancy hazard fix (closed)

### Workflow runs

- 26614851698 — the destructive apply
- 26617248946 — final successful wipe-then-apply rebuild
- 26621187358 — corpus restore
- 26626951615 — final clean deploy

### Runbook sections touched

- [PROD_RUNBOOK § Disaster recovery](../guides/PROD_RUNBOOK.md#disaster-recovery)
- [PROD_RUNBOOK § Co-tenant tailnet publish rules](../guides/PROD_RUNBOOK.md#co-tenant-tailscale-serve-rules)
- [RELEASE_PLAYBOOK § Phase 6 + 8](../guides/RELEASE_PLAYBOOK.md)
- [AGENTS.md rules 10 + 11](../../AGENTS.md)

### Prior PIRs

None — this is the first.
