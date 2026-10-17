---
name: prod-readonly
description: Look at the podcast prod VPS without changing it — one command as deploy@, docker compose of the stack with the right --env-file, or one original pulled from the audio archive with its temp copy on the box removed and checked gone. Use when asked to "check on prod", "look at the corpus on prod", "count X on prod", "what is running on prod", or "pull the archived original for an episode" — after the no-SSH routes in docs/guides/PROD_VALIDATION_QUICKREF.md have not answered it.
---

# prod-readonly

`scripts/prod.sh` holds the prod details that have gone wrong, so nobody types them by hand:

```bash
S=.claude/skills/prod-readonly/scripts/prod.sh
$S run 'find /srv/podcast-scraper/corpus -name "*.audioref.json" | wc -l'
$S compose ps --format '{{.Service}} {{.State}}'
$S fetch-archive sha256/27/84/<sha256>.mp3 <scratchpad>/orig.mp3
```

| Pinned | Why |
| --- | --- |
| `deploy@prod-podcast`, `~/.ssh/podcast_prod_operator`, `IdentitiesOnly` | the least-privilege user every workflow uses. **Never try other usernames**: fail2ban bans this machine's tailscale IP after a few failures (2026-09-19, lost access mid-investigation) |
| `docker compose --env-file /srv/podcast-scraper/.env -f compose/…` | Compose reads `.env` beside `compose/`, not the cwd: without the flag, "required variable PODCAST_DOCKER_PROJECT_DIR is missing a value" (2026-10-17) |
| `fetch-archive` in `/tmp/prod-readonly-<time>-<pid>`, removed on every exit path and checked gone | the archive pull used to leave 85 MB files in the box's `/tmp` until someone cleaned them by hand |
| the pipeline image already on the box | it carries rclone and reads the archive credentials from `.env`; nothing is pulled |

For a multi-line remote script, quotes in quotes, JSON or inline Python, use the `host-run` skill
instead of a long `run '…'`.

## Before using it

1. **Try without SSH first.** `docs/guides/PROD_VALIDATION_QUICKREF.md` (API + observability
   probes), the `o11y-review` skill, and the observability MCP (`obs_surface`, `obs_investigate`)
   answer most "is prod OK / why is X slow" questions with no SSH at all.
2. **Read-only means read-only.** The script cannot tell a read from a write; you can. Anything
   that changes prod (a restart, a migration, a file edit, `docker compose up`, a job) needs the
   operator's explicit go for that one action.
3. **Deploys and post-deploy steps belong to the prod session.** Tell it what you found
   (SendMessage), with the evidence; do not run them.
4. **Prod fixes are text-file fixes** (CLAUDE.md): never offer a DGX run, a relabel or a full
   reindex.
5. **Archive audio is real episode audio.** Fetch it into the session scratchpad, never into the
   repo, and delete it when done.

## Checked (2026-10-17)

- `run`: `hostname` → `prod-podcast`; the reference count → 2486.
- `compose ps` → `api running`, `viewer running`.
- `fetch-archive` of a known key → 85,341,121 bytes, then `OK: removed … on the box`, and no
  `/tmp/prod-readonly-*` left. A missing key → rclone `directory not found`, exit 2, the temp dir
  still removed.

## Not covered

- Anything that writes to prod, on purpose.
- Other hosts (the DGX, the mini): `dgx-status`, `host-run`.
- The operator and player surfaces' compose projects (`operator`, `player`): `compose` here is the
  podcast stack only.
