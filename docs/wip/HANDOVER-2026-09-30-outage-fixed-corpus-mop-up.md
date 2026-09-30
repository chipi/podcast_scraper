# Handover 2026-09-30 — the reboot outage and its fixes, and the corpus mop-up that is left

Supersedes the 2026-09-29 handover (renamed to this file). Written 10:06 CEST on 2026-09-30,
the beta-customer day. Times in the body are UTC unless marked local.

## 1. Where things are

| what | state |
| --- | --- |
| prod images | every surface on `sha-9949699`, deployed by "Deploy ALL" run `36676676865` (smoke green after one re-run) |
| `main` | `d56430770` — five commits on top of the deployed sha, **none deployed yet** (§4) |
| CI for `b9c07ff8f` | green except the stack test and Python app, which were still running at 10:05 local; `4c9618820` and `d56430770` are committed but **not pushed** |
| overnight repair queue | finished before the crash: 46 succeeded, 0 failed, 2 cancelled |
| prod secrets in RAM | **all three `/dev/shm/*-secrets` dirs empty since 06:14**. Every container runs on its mounted copies; any restart before the next deploy fails `Exited (127)`. The next deploy fixes it, and it then lasts until a reboot (§2) |
| `:8005` (vllm-translate) | reachable over the tailnet since `010d07126` (ACL applied; `curl` returns 401 like `:8003`) |
| nightly | still disabled — last step |

## 2. The outage (2026-09-30) and what changed because of it

**What happened.** The VPS went down hard at 03:26 (the journal just stops) and came back at 04:03.
The public player and operator APIs were down ~2 h, until recovered by hand.

- `player-api-1` and `operator-api-1` were `Exited (127)` with no app logs: `runc` failed the bind
  mount of a `/dev/shm/*-secrets/*` file.
- The two nginx containers crash-looped one layer downstream.
- `compose-api-1` was up and healthy **with no keys**. `podcast-scraper.service` recreates it at boot
  without `docker-compose.secrets.yml`.
- Docker did not start `alloy` at boot (`StartedAt` stayed 2026-09-16; cause unknown), so prod
  metrics and logs were dark the whole time.
- The "prod-podcast dark" alert (03:36) and "prod logs dark" alert (03:42) **did fire**, per the
  Grafana alert annotations.

**Fixed, on `main`:**

| commit | what |
| --- | --- |
| `f68b4388f` | The recovery workflow's first real run went green while recovering one surface of three. Now fixed and tested (`test_restage_prod_recreate.py`): `printf %s` instead of `%q`; an unknown surface fails; each container is recreated at its own tag; a keyless control plane counts as down. **Root cause:** logind's `RemoveIPC=yes` deleted all three secret dirs at the end of every `deploy` SSH session. `RemoveIPC=no` was applied on prod by hand (operator-approved, `/etc/systemd/logind.conf.d/10-keep-deploy-shm.conf`) and added to cloud-init. `DEPLOY_GOTCHAS.md` §1/§1b corrected. |
| `010d07126` | `:8005` granted wherever `:8003` is. |
| `b9c07ff8f` | `scripts/ops/prod_recovery_check.sh` (read-only JSON check) + `prod_health_lib.sh` (the "what is broken" rules, sourced by both the check and the recreate, parity-tested); `request_id` input + `run-name` on `restage-prod-secrets.yml`. |

**Unattended recovery** is designed in agentic-ai-homelab **RFC-0005** (Fleet 3 v1, a `fleetd`
fleet on the mini that checks prod every 5 min and dispatches + approves `restage-prod-secrets.yml`).
The operator and another agent are reviewing and implementing it. The podcast_scraper
prerequisites that are **not** done yet:

- the forced-command `authorized_keys` entry (cloud-init + live box; the public key comes from the
  mini);
- a `DEPLOY_GOTCHAS` note once the fleet is live.

**Until the fleet is live, after any reboot run `restage-prod-secrets.yml`** with `surfaces: all`
and `recreate: true`.

**Note for anyone checking keys:** `docker exec <c> env` shows provider keys EMPTY even when they are
there. The secrets shim exports them inside the server process only. List `/run/secrets` instead.

## 3. The corpus after the overnight queue (measured 2026-09-30 ~07:00)

**KG, newest artifact per episode (2,002):**

| provider | episodes |
| --- | --- |
| NVFP4 | 1,457 |
| `podcast-flash-0731` | 541 |
| `provider:extraction_failed` | 3 |
| `topic_labels` | 1 |

The 4 bad graphs are:

| episode | why | repair |
| --- | --- | --- |
| `0aa74076…` (pdrl.fm) | **emptied by relabel run `d65c2fcf`** — extraction failed during the relabel and overwrote a working graph | `rederive_only` after the deploy |
| `e63a2cc2…` (WSJ) | **emptied by relabel run `74245ab1`**, same way | `rederive_only` after the deploy |
| `52b31e33…` (omnycontent) | transcript lookup could not find the transcript (truncated title) | `rederive_only` after `4c9618820` is deployed |
| `6ea24954…` (omnycontent) | same | same |

`ad0d25f18` stops the first shape recurring: a re-run whose extraction fails now keeps the existing
working graph (counted as `kg_failures`) instead of writing an empty one. The two relabel logs show
only "provider extraction produced no topics/entities" — they predate the WARNING-with-reason change
(`e5057a160`), so **why** extraction failed is unknown. The re-run on the new image will say.

**Relabel violations: 15** (from 31 last night), **all `not_collapsed_one_spkr`**. By feed: 9 NPR
(`rss_feeds.npr.org_7ce5b183`), and one each in anchor.fm, substack, megaphone, simplecast, pdrl.fm
and omnycontent. `scripts/audit/relabel_worklist.py` is the generator.

## 4. Code on `main` / local, not deployed

| commit | state | what |
| --- | --- | --- |
| `e5057a160` | pushed | KG extraction failures log WHY at WARNING (type, message, finish_reason, bounded reply snippet) |
| `ad0d25f18` | pushed | a failed re-extraction keeps the working graph (§3) |
| `f68b4388f` `010d07126` `b9c07ff8f` | pushed | §2 |
| `4c9618820` | **local** | `run_index._transcript_beside` follows the record's `transcript_file_path` when the title was truncated — only inside this run's `transcripts/`, only accepted extensions, only when `names_the_same_episode` agrees (#2082's corrupt pointers stay refused). Measured read-only on prod: 1,835 found by exact stem, **165 newly found**, 2 still none. Those 2 point into a DIFFERENT run's `transcripts/` and are refused on purpose; neither is a damaged graph. |
| `d56430770` | **local** | a work-list run that finished NONE of its episodes now exits 1 (was: only one that matched none). Prod job `7b4465dd` matched both omnycontent episodes, refused both, and recorded `succeeded`. Partial repairs still succeed, with the unfinished ids at ERROR. Checked first that successful relabel/rederive runs record completion (every recent log reads `repaired N/N`). |

## 5. Remaining order — do not reorder

1. Push `4c9618820` + `d56430770`, wait for CI, then **Deploy ALL** on that sha (operator gate). It
   also restages the empty secret dirs (§1).
2. `rederive_only` for the 4 bad KG episodes (§3), then re-measure.
3. The 15 relabels — safe to run only after step 1 (the keep-the-graph guard).
4. Re-measure KG + relabel with the same two scripts.
5. **Full search rebuild** — `reindex-prod.yml mode=rebuild with_clusters=true`, operator approval.
   It is still required: the incremental prune refuses a legitimate shrink (#2158).
6. Re-enable nightly — **last**.

**Operator decisions pending:**

- **Re-enrich the 10 from #2097 step 3.** Not checked whether it was ever done.
- **The 51 GPU re-diarizations (#2097 step 9).** Approved before the finding that the 26 collapse
  cases were attribution, not diarization. They were not re-measured against that.

## 6. Parked — discuss after the chain (operator's instruction; carried)

- **Let the index prune a legitimate shrink.** Record expected per-stage output counts in manifests
  so the indexer can tell "legitimately smaller" from "died partway". Then repairs would stop
  forcing full rebuilds.
- **A work-list size ceiling** — cap by count instead of by feed.
- **Scope the post-run `enrich-edges`** to the run's episodes instead of all 2,002 (~4 min per run).
- **A prod MCP path for agents.** No client is wired to the prod obs MCP.
- **New today:** the operator API's first request after a recreate took **69,979 ms**
  (`/api/corpus/digest`, cold cache). The first visitor after every deploy or reboot waits that long.
- **New today:** `podcast-scraper.service` brings the control plane up keyless at boot (§2).
  RFC-0005's recovery covers it within minutes; a boot unit that doesn't do it at all would remove
  the window.

## 7. Operational notes worth not rediscovering

- **SSH**: `ssh -o BatchMode=yes root@prod-podcast`. The key drops out of the local agent now and
  then; `ssh-add` fixes it.
- **Operator key inside `compose-api-1`**: `cat /run/secrets/app_operator_api_key`.
- **`compose-api-1` has no `/tmp`** since the 2026-09-30 recreate: `docker cp` into it fails. Pipe a
  script in instead: `ssh … 'docker exec -i compose-api-1 python -' < script.py`.
- **Cancel route** is `POST /api/jobs/{job_id}/cancel`. A cancel that returns `status: running` may
  still have worked — read the registry row.
- **The post-run tail is not a hang**: ~4 min of `enrich-edges` + `search.reindex` after
  `result: episodes=`.
- **Count episodes, not files** — newest artifact per `episode_id`.
- **Metrics**: `https://vm.<tailnet>.ts.net/api/v1/query`. Grafana alert history
  (`/api/v1/rules/history`) returns 500; `/api/annotations?type=alert` works with the obs reader
  token from inside `player-obs-1`.
- **Worktree hooks**: committing from a secondary worktree needs `PYTHONPATH=$PWD/src`, because the
  pre-commit collection check imports the venv's editable install, which points at the main checkout.
- **zsh does not word-split** an unquoted `$VAR`; pipe through `xargs` or use an array.

## 8. NOT verified — do not treat as settled

- **Why extraction failed** for `0aa74076` / `e63a2cc2` during the relabels. The logs predate the
  reason logging.
- **Why Docker did not start `alloy` at boot.** Unknown. It was restarted by hand at 06:05.
- **Whether the 03:36 / 03:42 alerts reached the operator.** They fired; delivery unchecked.
- **Whether `relabel_only` fixes all 15 remaining `not_collapsed_one_spkr` episodes.** Last night's
  relabels took the total from 31 to 15, so it fixes most; why these 15 survived is not examined.
- **The GI artifacts of repaired episodes** — Insight counts not checked.
- **The 2 transcripts still unresolved** after `4c9618820` (Baseten / megaphone, "The Coolest
  Diffusion Research" / flightcast). Their pointers cross runs. Whether the existing sibling-run
  search resolves them was not run.
- **Carried, not re-verified:** char-offset misalignment warnings from `enrich-edges`; obs token
  staging; who removed `jobs.paused`; #2158 partial emission; ~107 not-on-any-roster persons (#2096);
  87 newline-less transcripts (#2098–#2100); the WEB-tier enrichment passes.
