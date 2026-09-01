# Handover: prod cadvisor fixed (#1887) + the metric gaps that remain

Date: 2026-08-31, ~02:40 UTC. Written for whoever picks this up next — podcast-side or
homelab-side. Everything below was measured on the boxes, not inferred.

---

## PART 1 — DONE: prod per-container metrics now have a `name` label

### The defect

`container_memory_working_set_bytes{instance="prod-podcast"}` had **54 series, 0 with a
`name` label**. Every per-container dashboard panel for prod rendered empty. Data existed
but was keyed only by cgroup path (`/system.slice/docker-<64-hex>.scope`), so it was
unusable without a manual hash→name mapping.

### Root cause (confirmed at source, not assumed)

cadvisor **v0.49.1** on a host using the **containerd snapshotter**
(`docker info` → `Driver=overlayfs`, `io.containerd.snapshotter.v1`). Images live under
`/var/lib/containerd/...`, so `/var/lib/docker/image/overlayfs/layerdb` **does not exist**.
cadvisor's docker factory needs it to map cgroup → container, fails for every container,
and falls back to raw cgroup scanning with no `name`/`image`/`container_label_*`:

```
E0826 manager.go:1116] Failed to create existing container:
/system.slice/docker-<id>.scope: failed to identify the read-write layer ID ...
open /rootfs/var/lib/docker/image/overlayfs/layerdb/mounts/<id>/mount-id: no such file or directory
```

400 of the last 400 log lines were this, continuously since 2026-08-05.

This is google/cadvisor#3643, fixed by PR #3709, first released in **v0.54.0**.

### IMPORTANT: v0.60.5 does not exist

#1887 and several relayed messages (mine included) referenced **v0.60.5**. That tag has
never been published. Real tags from `gcr.io/v2/cadvisor/cadvisor/tags/list` top out at:

```
... v0.50.0, v0.51.0, v0.52.0, v0.52.1, v0.54.1, v0.55.1
```

Editing the compose to v0.60.5 and running `up -d` would have failed the pull and left prod
with **no** container metrics. **Use v0.55.1** (newest, contains #3709).

Note also: the v0.52.1 fix in #1887 was correct for the **DGX** (its cause was a different
one — Docker API client 1.41 below the daemon's minimum 1.44) but was **never right for
prod**, whose daemon reports `MinAPI=1.40`, so 1.41 is acceptable there. Two boxes, two
different root causes, same symptom. Do not collapse them.

### What was changed

Host `prod-podcast`, file `/opt/vps-observability/docker-compose.yml`, one line:

```diff
-    image: gcr.io/cadvisor/cadvisor:v0.49.1
+    image: gcr.io/cadvisor/cadvisor:v0.55.1
```

then `docker compose up -d cadvisor` (that stack has only `alloy` and `cadvisor`; alloy was
not recreated — verified still up since 2026-08-15, 0 restarts).

**Method used to keep prod safe** — repeat this if you touch cadvisor again:
1. Ran v0.55.1 as a **throwaway container on a spare port** with read-only mounts, so the
   live stack was untouched. Compared side by side: v0.49.1 → 0 named, v0.55.1 → 16 named,
   0 layerdb errors.
2. Removed the probe, confirmed the live container was still v0.49.1 / 0 restarts.
3. Backed up the compose, changed one line, recreated only that service.

**Backup**: `/opt/vps-observability/docker-compose.yml.bak-20260831-023702`
**Rollback**: restore that file, `docker compose up -d cadvisor`. ~30 seconds.

### Verified result

```
container_memory_working_set_bytes{instance="prod-podcast",name!=""}
  before: 0        after: 15
```

15 named = **all 15 running containers** (alloy, cadvisor, compose-api-1, compose-viewer-1,
litellm, litellm-postgres, litellm-spend-push, operator-api-1, operator-viewer-1,
orrery-web, player-api-1, player-digest-scheduler-1, player-learning-app-1, player-mcp-1,
player-obs-1). Full coverage, not partial. cadvisor log: 0 layerdb errors,
`Registration of the docker container factory successfully`.

Ephemeral pipeline containers (`compose-pipeline-llm-run-*`) are already matched by the
Alloy keep rule in `config.d/operator.alloy`, so they should appear named during a run.
**Not yet observed** — verify on the next pipeline run.

---

## PART 2 — NOT DONE, THE ONE THING LEFT: neither fix is in version control

`/opt/vps-observability/` on prod is **root-owned and not a git checkout**. The v0.55.1 pin
exists only on the box. Any rebuild/reprovision of that host reverts to v0.49.1 and the
defect returns silently.

Per #1887 the source of truth is `infra/observability/hosts/prod-podcast/docker-compose.yml`
in `chipi/agentic-ai-homelab` (commit `8161a2a` set it to v0.52.1 there).

**Action needed — this is the whole remaining task:**

1. Set that repo's prod pin to **v0.55.1** — not v0.52.1 (wrong root cause for prod), not
   v0.60.5 (does not exist).
2. Port the widened cadvisor keep-list into the repo's version of
   `config.d/base.alloy`. The live regex is in PART 3; copy it verbatim, including the
   `container_memory_rss` split (the `_bytes` bug will silently come back otherwise).
3. Consider aligning the DGX to v0.55.1; it is on v0.52.1, which works there but leaves the
   two hosts on different versions for no reason. The DGX would also benefit from the same
   keep-list widening — PSI on the DGX is directly relevant to #1886 GPU co-tenancy.

Until (1) and (2) land, a rebuild of prod-podcast silently reverts both fixes.

---

## PART 3 — DONE: keep-list widened 15 → 28 metric names, and an old bug fixed

prod cadvisor exports **60** `container_*` metric names; VictoriaMetrics stored only **15**.
The filter is a keep-list at `/opt/vps-observability/config.d/base.alloy:43`
(`prometheus.relabel "cadvisor_keep"`). Now stores **28**.

**Backup**: `/opt/vps-observability/config.d/base.alloy.bak-20260831-025341`
**Reload used**: `docker kill -s HUP alloy` → log says `config reloaded`, 0 errors, and
alloy was **not restarted** (still up since 2026-08-15, 0 restarts). Rollback = restore the
backup and HUP again.

### 3a. A latent bug — fixed

The regex said `container_memory_(usage|working_set|max_usage|rss)_bytes`, which expands to
`container_memory_rss_bytes`. cadvisor exports **`container_memory_rss`** — no `_bytes`
suffix — so RSS was explicitly requested and had **never** been collected. Split into its
own alternative. Now 15 named series.

### 3b. What was added, and what it is actually good for

| metric | status | note |
|---|---|---|
| `container_pressure_{cpu,io,memory}_{waiting,stalled}_seconds_total` | **15 named** | PSI — direct starvation signal, better than utilisation for #1888 |
| `container_cpu_{user,system}_seconds_total` | **15 named** | separates CPU-bound from syscall/IO-bound |
| `container_memory_rss` | **15 named** | the 3a bug fix |
| `container_scrape_error` | working, sum = 0 | meta-observability: non-zero means cadvisor is failing on a container. Would have surfaced this whole defect months earlier |
| `container_health_state` | 69 series | healthcheck state |
| `container_fs_{usage,limit}_bytes` | **44 series, ZERO named** | see caveat below |

**Caveat on disk — read this before building a panel.** `container_fs_usage_bytes` is NOT
per-container on this host. Every series is `id="/"`, `name=""`, scoped by `device`:

```
container_fs_usage_bytes{device="/dev/sda1", id="/", name=""}  5.95e10   # 59.5 GB, matches df
```

Per-container disk requires the read-write layer mapping, which under the containerd
snapshotter does not exist — v0.55.1's #3709 fix restored container **identity** for
CPU/memory/PSI, not per-container disk sizing. So what we get is **machine disk per device**,
which is still useful (it now lives in the same store as everything else) but must not be
labelled "per-container disk" on a dashboard. Per-container disk usage remains unavailable
on this host; do not spend time trying to make it work via cadvisor.

**Measured cardinality after the change:** 1907 `container_*` series for `instance="prod-podcast"`.

The remaining ~32 dropped names (blkio device totals, per-packet counters, inode/sector
counts, memory failcnt/swap/mapped_file) are genuinely low value here; leaving them dropped
is a deliberate default, not an oversight.

### Sanity check that it is all live

```
container_memory_working_set_bytes{instance="prod-podcast",name!=""}   -> 15
container_memory_rss{...,name!=""}                                    -> 15
container_pressure_io_waiting_seconds_total{...,name!=""}              -> 15
sum(container_scrape_error{instance="prod-podcast"})                   ->  0

topk(5, container_memory_working_set_bytes{...,name!=""}):
  litellm 1224 MB | operator-api-1 816 MB | player-api-1 735 MB
  compose-api-1 650 MB | alloy 294 MB
```

---

## Quick reference

```sh
# access (the key exists locally; this is what unblocked the whole thing)
ssh -i ~/.ssh/podcast_prod_operator deploy@prod-podcast   # read-only work
ssh -i ~/.ssh/podcast_prod_operator root@prod-podcast     # config edits

# current state
docker inspect cadvisor --format '{{.Config.Image}}'      # -> v0.55.1
curl -s http://127.0.0.1:8081/metrics | grep -c 'name="'  # named series at source

# the success criterion, against homelab VictoriaMetrics (tailnet)
curl -s "http://$(tailscale ip -4 homelab):8428/api/v1/query" \
  --data-urlencode 'query=count(container_memory_working_set_bytes{instance="prod-podcast",name!=""})'
```

Related: #1887 (this issue), #1886 (DGX load characterisation — the consumer of these
metrics), #1888 (pipelining; PSI metrics above would materially improve its evidence).
