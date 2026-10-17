# Host systemd units (prod VPS)

## prod-ops-health (#1877)

`prod-ops-health.timer` starts `prod-ops-health.service` daily at 06:00 UTC. The service runs
`scripts/ops/prod_ops_health_scheduled.sh` from the box checkout as `deploy`. That wrapper runs
`scripts/ops/prod_ops_health.sh` (#1876), which pushes `prod_ops_health_*` to the homelab
VictoriaMetrics, and on a red run (exit 1) or missing prerequisites (exit 5) raises a GlitchTip
event from inside `compose-api-1` with that container's own DSN.

`prod-ops-health.yml` stays for a manual run from GitHub; it has no schedule.

**Install.** A rebuild installs and enables the timer from cloud-init (`prod.user-data`, the
container-metrics block, note 20). On a live box, once the checkout carries the wrapper (any deploy
after the commit), as root:

```sh
install -m644 /srv/podcast-scraper/infra/systemd/prod-ops-health.* /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now prod-ops-health.timer
```

**Verify.**

```sh
systemctl list-timers prod-ops-health.timer   # next run 06:00 UTC
systemctl start prod-ops-health.service       # one run now (two 1-token completions)
systemctl status prod-ops-health.service      # exit status of that run
journalctl -u prod-ops-health --since today   # per-check GREEN/ORANGE/RED and "metrics pushed"
```

A broken scheduler cannot go unnoticed: the homelab page greys the section after 26 h without a
run, and the Grafana rule "Prod ops health check stale (no run in 27h)" fires at 27 h.

**Undo:** `systemctl disable --now prod-ops-health.timer`, remove the two unit files,
`systemctl daemon-reload`, and restore the `schedule:` block in `prod-ops-health.yml`.
