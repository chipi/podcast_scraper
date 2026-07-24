# Grafana dashboards — podcast application (ADR-121 ownership split)

Source-of-truth JSON for the **podcast application's** Grafana boards, so the
Grafana reorg (2026-07-23) survives a Grafana rebuild instead of living only in
Grafana's SQLite DB.

## Ownership split

The VPS Grafana was reorganised into three folders to stop mixing *generic prod
infra* with *the podcast app* (which is just one service the box happens to host,
a sibling of orrery):

| Folder | Owns | Lives in repo |
| --- | --- | --- |
| **Podcast Player** | consumer player (`closelistening.app`) — logs, later container/API metrics | **here** (`podcast-player.json`) |
| **Podcast Operator** | operator API + viewer (`compose-api-1`/`compose-viewer-1`) | **here** (`podcast-operator.json`) |
| **Production Infra** | host (node-exporter), containers (cAdvisor), edge/security — generic, nothing podcast-specific | **agentic-ai-homelab** (`hosts/prod-podcast/grafana/`) |

Rationale: the podcast app boards belong with the podcast repo; the generic host
boards belong with the infra/homelab repo that owns `base.alloy`, the cAdvisor
compose, and the homelab landing page (`homelab-home/gen.sh`, which gained the
matching "Production" column).

## Data sources (on the homelab Grafana)

- `VictoriaMetrics` (uid `victoriametrics`) — metrics (host + cAdvisor + `job=api`)
- `VictoriaLogs` (uid `victorialogs`) — logs (`app=podcast` / `app=player`, set by the
  `podcast.alloy` / `player.alloy` drop-ins)

## Import / update

Export (what produced these files) and re-import go through the Grafana API:

```sh
GF=http://admin:<pw>@homelab:3000
# export one board
curl -s "$GF/api/dashboards/uid/<uid>" | jq '.dashboard | del(.id,.version)' > <name>.json
# import / update (keeps the uid, so deep links survive)
curl -s -X POST "$GF/api/dashboards/db" -H 'Content-Type: application/json' \
  -d "$(jq -n --slurpfile d <name>.json '{dashboard:$d[0], folderUid:"<folder-uid>", overwrite:true}')"
```

Folders are **provisioned from the directory structure** (`agentic-ai-homelab`
`.../grafana/provisioning/dashboards/dashboards.yml`, `foldersFromFilesStructure:
true` + empty `folder`): each subdir under `grafana/dashboards/` is a Grafana
folder (`Homelab/`, `Production Infra/`, `Podcast Operator/`, `Podcast Player/`),
so folder uids are auto-managed — don't hardcode them.

Board uids are generic, not `vps-podcast-*`: `prod-infra-host-overview`,
`prod-infra-containers`, `prod-infra-edge-security` (Production Infra);
`podcast-operator-overview`; `podcast-player-overview`.

> Known gap (tracked): the **Containers** board (Production Infra) is empty on
> Docker 29 — cAdvisor can't emit container `name` labels with the containerd
> image store. Tracked in the exporter-replacement issue (#1272); not a dashboard bug.
