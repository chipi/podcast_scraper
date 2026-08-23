# Prod is spending on Deepgram with nothing dispatching it — diagnostic brief

**For an agent with SSH access to the prod VPS.** Read-only until section 4. Target: 10 minutes.

## What is known

- The operator's Deepgram balance has been falling for hours, still falling now.
- A corpus run directory was written at **2026-08-19 10:33:35Z**.
- **No GitHub Actions run exists after 07:00Z**, and no reprocess workflow ran today at all
  (`gh run list --limit 20` — every entry `completed`). So the work is being started ON THE BOX.
- Yesterday's picture, for contrast: 45 episodes carry a 2026-08-18 run stamp, 81 carry
  2026-08-19. The corpus serves 678 episodes total.
- Serving-side sampling is NOT a reliable liveness check here — the corpus API serves "newest
  run per episode", so an in-flight run is invisible until its episodes complete. Two samples 60s
  apart showed no change while the operator could see spend continuing. Do not use it to conclude
  "nothing is running".

## The leading hypothesis, and why

`api` is declared `restart: unless-stopped` (compose/docker-compose.prod.yml:89) and hosts the
operator job queue. Its sweeper promotes queued work on a **30-second loop**
(`server/queue_sweeper.py:46`, `DEFAULT_SWEEP_INTERVAL_SECONDS = 30.0`). `pipeline-llm` by
contrast has **no** restart policy — it is one-shot `docker compose run --rm`
(compose/docker-compose.prod.yml:228).

That asymmetry predicts exactly the observed symptom: **each pipeline container exits, and ~30s
later the sweeper promotes the next queued job.** Killing a container therefore does nothing
durable, no Actions run is needed, and it continues indefinitely as long as the queue is
non-empty or something keeps refilling it.

The competing hypothesis — one long-lived container left over from the 2026-08-18 reprocess that
was killed at the Actions timeout — is distinguished by a single observation in §1.

## 1. THE DISCRIMINATOR — run this first

```bash
docker ps -a --filter name=pipeline \
  --format '{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.CreatedAt}}\t{{.Command}}'
```

- **One container, `Up ~15 hours`** → leftover from last night's killed reprocess. Hypothesis B.
- **Several containers created minutes apart, most `Exited`, one `Up`** → the sweeper is
  promoting job after job. Hypothesis A. **The gap between `CreatedAt` values is the loop
  period** — note it.
- **Nothing at all** → the spend is not coming from a pipeline container; jump to §3.

## 2. What is it actually doing, and is it Deepgram?

```bash
ID=$(docker ps -q --filter name=pipeline | head -1)

# The command line tells you WHICH work: --reprocess-* flags, an `enrich` subcommand,
# a feeds-spec ingest. This is the single most informative fact after §1.
docker inspect "$ID" --format '{{json .Args}}' | tr ',' '\n'
docker inspect "$ID" --format 'started={{.State.StartedAt}} restart={{.HostConfig.RestartPolicy.Name}}'

# Live log. Look for transcription lines and their CADENCE — one every few minutes means
# episode-by-episode ASR; a burst means retries.
docker logs --timestamps --tail 200 "$ID"

# Is it talking to Deepgram right now?
docker exec "$ID" sh -c 'command -v ss >/dev/null && ss -tn || netstat -tn' 2>/dev/null | head
```

## 3. The queue — what is feeding it

The corpus is a **named docker volume (`corpus_data`), not a bind mount**, so reach it through a
container rather than a host path:

```bash
API=$(docker ps -q --filter name=api | head -1)

# Queue state, per-job logs, and whether the pause flag is already set.
docker exec "$API" ls -la /app/output/.viewer/ /app/output/.viewer/jobs/ 2>/dev/null

# Most recent job logs — these name the job type and its arguments.
docker exec "$API" sh -c 'ls -t /app/output/.viewer/jobs/*.log | head -5 | \
  while read f; do echo "=== $f"; tail -30 "$f"; done'

# The api's own log shows the sweeper promoting: grep for the promote/drain lines.
docker logs --timestamps --since 3h "$API" 2>&1 | grep -iE "promot|drain|sweep|queued|job " | tail -40
```

**The question this answers: is the queue draining a fixed backlog, or is something re-enqueuing?**
A backlog drains to zero and stops. Re-enqueuing does not. There is a known ingest→enrichment
auto-chain in this repo (see `docs/homework`, H7) — if each completed job enqueues the next, that
is an unbounded loop and it is the actual bug.

Also worth one look — something outside docker:

```bash
crontab -l 2>/dev/null; sudo crontab -l 2>/dev/null
systemctl list-timers --all | head -20
```

## 4. STOPPING IT (only after the above)

Order matters. Killing the container first just gives the sweeper a fresh slot in 30s.

```bash
# 4a. Hold the queue FIRST. Touch-file switch; the sweeper checks it every cycle
#     (queue_sweeper.py:49, PAUSE_FLAG_RELPATH = ".viewer/jobs.paused").
docker exec "$API" sh -c 'mkdir -p /app/output/.viewer && touch /app/output/.viewer/jobs.paused'
docker exec "$API" ls -l /app/output/.viewer/jobs.paused          # confirm it exists

# 4b. THEN stop the running job. SIGTERM with grace so in-flight cost is still recorded.
docker stop --timeout 20 $(docker ps -q --filter name=pipeline)

# 4c. Verify it stays stopped — wait past one sweep interval.
sleep 45; docker ps --filter name=pipeline
```

Deleting `jobs.paused` resumes the queue, so this is reversible and safe to leave set.

## 5. What to report back

1. §1 output verbatim — **one long-lived container, or many short-lived ones?**
2. The pipeline container's `.Args` — which subcommand and flags.
3. Whether the queue is draining a finite backlog or being refilled, and by what.
4. How many jobs ran since 2026-08-19 00:00Z, and their type.
5. Whether `jobs.paused` was ALREADY present (if so, the sweeper is not the culprit and the
   hypothesis in §0 is wrong — say so).

## Context the fixes have NOT reached prod

Everything below is committed on `hotfix/cost-containment-and-scope-gate` and is **not in the
running image**. Do not expect to see any of it in prod logs:

- a per-run spend ledger that spans feeds and counts at the provider choke point
- a pre-flight gate that prices a selection and refuses it over the cap
- ASR pre-authorisation (refuse the call before making it)
- batch halt on a tripped cap
- `timeout` + unbuffered + box-local logs in `reprocess-prod.yml`

The prod image still has the 2026-08-18 behaviour: a `$5` cap that cannot fire, because
`stages/transcription.py` has no cost check and the first one that sees ASR spend runs after
`transcription_thread.join()`.
