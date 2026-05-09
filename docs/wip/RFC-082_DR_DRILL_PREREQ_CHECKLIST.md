# RFC-082 DR drill — prerequisite checklist (WIP)

**Purpose:** Complete these items **before** scheduling the timed disaster
recovery drill in GitHub [#724](https://github.com/chipi/podcast_scraper/issues/724).
Tracking issue for this list: [#751](https://github.com/chipi/podcast_scraper/issues/751).

**Related docs:** [RFC-082 § Disaster recovery](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md),
[PROD_RUNBOOK — Disaster recovery](../guides/PROD_RUNBOOK.md#disaster-recovery).

---

## 1. Isolation (must not touch real prod)

- [ ] Dedicated Hetzner project (or equivalent) for the drill; quotas and billing confirmed.
- [ ] Tailscale: hostname / MagicDNS distinct from prod; ACL or tags (for example `tag:dr-drill` per #724) so drill nodes are obvious.
- [ ] OpenTofu / Terraform state and var-files scoped only to the drill project (no apply against prod state).
- [ ] Optional: second age key / credential set so drill decrypt paths do not overlap prod (#724 Dependencies).

---

## 2. Backup and restore (closed loop)

- [ ] Backup workflow understood (`backup-corpus-prod` or documented manual equivalent for the drill host).
- [ ] Drill path: backup from drill stack → verify release on `chipi/podcast_scraper-backup` → restore **that** tag. From a machine with `gh` auth, pin the tag: `PODCAST_BACKUP_TAG=<tag> make restore-corpus` (optional `PODCAST_BACKUP_REPO=owner/repo`; default backup repo in the repo [Makefile](https://github.com/chipi/podcast_scraper/blob/main/Makefile)). When `PODCAST_BACKUP_TAG` is unset, `restore-corpus` still restores **latest** only.
- [ ] Corpus seed: enough data on the drill host for backup/restore to be meaningful (synthetic or policy-approved subset).

---

## 3. Observability tagging (Sentry + Grafana Cloud)

Pick one **`PODCAST_ENV`** value for the entire drill stack (for example `dr-drill`). Do **not** use `prod` on a non-prod host; events and metrics share the same Sentry projects / Grafana stack as today but must filter by `environment` / `env`.

**After `docker compose up` on the drill host:**

1. **API** — confirm env inside the container (compose project name may differ):

   ```bash
   docker exec "$(docker ps -q -f name=api)" sh -c 'echo PODCAST_ENV=$PODCAST_ENV'
   ```

2. **Grafana Agent** — same check for the agent container; labels `env` and `release` come from these vars ([compose/grafana-agent.yaml](https://github.com/chipi/podcast_scraper/blob/main/compose/grafana-agent.yaml)).

3. **Viewer + Sentry** — nginx injects `window.__PODCAST_ENV__` from `PODCAST_ENV` ([docker/viewer/nginx-prod.conf.template](https://github.com/chipi/podcast_scraper/blob/main/docker/viewer/nginx-prod.conf.template)); fetch HTML and confirm the script snippet, or open the viewer and verify in DevTools.

4. **Grafana Cloud Prometheus** — use your drill value in place of `<drill-env>`:

   ```text
   up{component="api",env="<drill-env>"}
   ```

   RFC-082 smoke examples use `env="prod"`; for the drill, substitute `<drill-env>` everywhere you query.

5. **Sentry** — filter issues by environment matching `PODCAST_ENV` ([sentry_init.py](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/utils/sentry_init.py)). Optional hard isolation: separate DSNs / projects for drill.

6. **PostHog** (if the viewer build includes it) — decide whether drill traffic should use a separate project or stay disabled.

---

## 4. Deploy and smoke baseline

- [ ] Drill host reaches GHCR and runs the same compose overlay story as prod / pre-prod.
- [ ] Synthetic `.env` for drill (test API keys) documented; operators know what is fake vs prod.

---

## 5. Handoff to #724

When the boxes above are done, add a short **drill-day script** (ordered steps + timer boundaries) and execute [#724](https://github.com/chipi/podcast_scraper/issues/724). File `docs/wip/RFC-082_DR_DRILL_RESULTS.md` from that issue’s deliverable.
