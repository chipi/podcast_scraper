# Findings — deploy sha-d67492c + speaker-attribution chain (2026-09-22)

Running log. Appended as the chain progresses. Epic #2097; runbook
`docs/wip/POST-DEPLOY-SPEAKER-ATTRIBUTION-2026-09-14.md`; handover received 2026-09-22.

---

## Deploy (COMPLETE, green)

`deploy-all-prod` run 35698136925 — all 8 jobs success. 9/9 containers on `sha-d67492c`,
`.env` `PODCAST_IMAGE_TAG=sha-d67492c`, `restarts=0`. Rollback target was `sha-825a6fb` (unused).

### F1. Cold start makes the api look broken for ~2.5 minutes — it is not

`compose-api-1` reported **unhealthy, failing streak 4** right after recreate. Cause:

    GET /api/corpus/digest -> 200 in 149531.2ms      <- 149 SECONDS, first call after recreate
    then: digest -> 200 in 574ms / 596ms / 591ms     <- warm

The healthcheck has a 10s timeout; the first digest call saturates the worker and starves it.
`restarts=0`, `OOMKilled=false`, and `/api/health` returned 200 throughout. It self-cleared to
`healthy, streak=0` about 40s after I first looked.

Worth fixing at cause rather than living with: either warm the digest cache at startup the way
`search warmup complete` already does, or give the healthcheck a `start_period` that covers a
cold digest. Today it produces a scary-but-false unhealthy window on every single deploy.

### F2. The post-deploy operator smoke failed for the SAME reason — transient

`operator / Post-deploy live smoke` failed at 07:16:15, inside the saturation window (the 149s
digest completed 07:16:09). Failure shape:

    waiting for getByTestId('login-button')
    6 passed, 1 failed

Re-run with no code change: **success**. The gate cookie was set and 6/7 tests passed, so this
was not auth and not a credential — the SPA's button had not rendered while the api was blocked.

### F3. Two wrong diagnoses of mine, and the method that fixes them

Both were mine, both cost time, both have the same root:

1. I curled `https://operator.closelistening.app/api/app/auth/login` from my laptop, got
   `200 text/html`, and called it "user-facing breakage — an operator can't sign in". **Wrong.**
   `infra/caddy/operator.caddy` gates everything behind a preview cookie / Basic auth with a bare
   `handle {}` fallback serving the coming-soon page. No credential -> coming-soon page. That 200
   was CORRECT behaviour.
2. I then blamed a stale `OPERATOR_PREVIEW_PASS`. **Wrong.** The job log shows the preview cookie
   was set and the gate passed.

**Method:** any external probe of a gated surface measures the GATE, not the app. Probe the
container directly on the box (`docker exec … http.client`) to separate app from edge. And note
`urllib.request.urlopen` FOLLOWS redirects — it turned a correct `307 -> accounts.google.com`
into an apparent `200 text/html`, which nearly became a third wrong diagnosis. Use
`http.client` and read `.status` when a redirect is the thing under test.

### F4. SECURITY — the preview gate cookie is in a session transcript

While grepping the smoke job log I printed the live value of `cl_op_preview=…` (the operator
surface's gate cookie). It is a real credential and it is now in a chat transcript. **Rotate it**
(`__OPERATOR_PREVIEW_COOKIE` in the host-side `.env`, substituted into `operator.caddy` at deploy).
Not urgent — the gate is a pre-launch doorman, not the auth boundary — but it should not sit
there.

### F5. What this image fixed, verified on prod

Person photos + org logos now serve. Checked by magic number, not status code, direct to
`player-api-1` with no auth header:

    person:elon-musk        200 image/jpeg 29,372B ffd8ff (JPEG)
    person:demis-hassabis   200 image/jpeg 30,046B ffd8ff
    person:no-such-human    404 application/json          <- absent still 404s, not 500
    org:37signals           200 image/png  42,343B 89504e (PNG)

And #38's gauge is live in VictoriaMetrics:

    podcast_pipeline_last_success_age_seconds{command_type="full_incremental_pipeline"} = 5880s (1.63h)
    podcast_pipeline_last_success_age_seconds{command_type="corpus_enrichment"}         = 5370s (1.49h)

Threshold is 36h, so `podcast-ingest-stalled` reads Normal — no firing gap. Three series each
(one per api instance, all reading the same file); the rule's `max()` collapses them.

---

## Step 2 gate — `upgrade_dry_run` (PASSED)

Run 35699487495.

### F6. m0002 no-ops — the check no local drill could answer

    -rw-r--r-- 1 deploy docker 271457515 Sep 22 05:43 corpus/search/metadata.json
    [DRY-RUN] 0002_two_tier_native_reindex: index already present — no-op

Disk: corpus **9.6G**, `/dev/sda1 150G, 101G avail, 30%` — ~10x the snapshot's need.

### F7. The snapshot validation was representative — numbers match near-exactly

                            handover expected    prod dry-run
    0007 rewrites                    153              153     exact
    0008 stamps                    2,265            2,265     exact
    0009 promote / demote     ~3,076 / 208      3,077 / 208   +1 promote
    0010 rewrites                     93               93     exact
    hand-read demotions              ~59               59     exact

Full 0009 line also reports: 259 already correct, 109 no usable roster, 196 pre-listening guess,
410 roster names with no matching node (need a re-enrich, not this migration), 0 unparsable.

### F8. NEW — not in the handover's expectations

    0009: 2 node(s) matched MORE THAN ONE roster entry and were left untouched

Untouched is the safe outcome, but it is an unlisted case. Worth a look at step 1c to see which
two, in case the ambiguity means something.

### F9. "version 2.7.0.dev0 -> 2.7.4" is a MISLEADING gate — drop it from the runbook

The handover asks to require "ten migrations pending, version 2.7.0.dev0 -> 2.7.4". The dry-run
prints the ten migrations (0001..0010) and **no version header at all**, so "ten pending" is read
by COUNTING the enumeration.

More importantly, 2.7.4 **was never released** (operator, 2026-09-22 — correct). It is internal
migration bookkeeping. The code says so itself, `m0005_gi_v3_1_route_and_tag.py:39`:

    # Shares the 2.7.1 release marker with 0003/0004 (all unreleased, landing in the same next
    # train over the deployed 2.7.0.dev0). to_version is a ledger LABEL + optional --to-version
    # ceiling, not a run gate — the runner applies any migration whose id is absent from the ledger.

and m0004 repeats it: "the runner applies any migration whose id is absent from the corpus ledger
**regardless of version**".

    pyproject.toml:7 / __init__.py:59   version = "2.7.0.dev0"   <- the only REAL version
    m0001, m0002   to_version 2.7.0  |  m0003–m0007  2.7.1       <- all unreleased labels
    m0008          to_version 2.7.2  |  m0009        2.7.3
    m0010          to_version 2.7.4                              <- ledger label, not a release

**Consequence:** the version string is decorative, and phrasing it as a gate invites the reading
that a 2.7.4 release exists. The real, checkable gate is **ten migration ids absent from the
ledger** — which the dry-run did show. The runbook should say that instead.

**Wider point (operator, 2026-09-22):** the project has been on `2.7.0.dev0` the whole time —
2.7.1, 2.7.2, 2.7.3 and 2.7.4 were all labels for a train that never departed. Four
release markers now name releases that do not exist, and every reader of the runbook will infer
they do (this conversation is the proof). Two honest ways out, no third:

  a) cut the releases, so the labels mean something; or
  b) stop stamping `to_version` and let ledger-by-id be the only story — it is already the only
     thing the runner honours (`runner.py` applies any migration whose id is absent, regardless
     of version).

Not actioned — flagged only. Operator's own words: "maybe we should have".

---

## Tooling / process findings (not chain-specific)

### F10. `vue-tsc --noEmit -p tsconfig.json` checks NOTHING here

It exits 0 with **zero bytes of output** against a deliberate type error, and against a missing
`ref`/`watch` import that would have crashed a component at runtime. The real gate is
`npm run build` (`vue-tsc -b`, build mode, resolves project references), which fails correctly:

    src/components/OrgCardContent.vue(38,20): error TS2304: Cannot find name 'ref'.

I claimed "typecheck clean" twice off the broken form. Prove a gate can fail before quoting it.

### F11. stack-test does not trigger on player or server-route changes

`stack-test.yml`'s `paths:` filter did not match either `a1451f486` (server routes) or
`dbc5cdfd8` (player + server routes) — it never ran for them, not even "skipped". So the person
photo change reached prod without a real-browser stack test. It was covered by 1534 player unit
tests + `app-e2e`, but that is not the same thing.

### F12. Both prod workflows sit behind the `prod` environment approval gate

`deploy-all-prod` and `inspect-prod-corpus` both land in `waiting` with
`awaiting environment: prod, approvers: chipi`. Expected and correct — noting it so the
step-by-step cadence accounts for a human click per dispatch.

### F13. The 11 `substack:post:*` work-list ids — theory disproven, NO code change needed

Investigated on the hypothesis that `_on_disk_guid_index` (`scraping.py:392-394`) drops records
with no `guid`, making those ids unreachable by `--reprocess-episode-ids`. Measured on prod
(compose-api-1, sha-d67492c, `/app/output`):

    metadata files scanned           2312
    episodes with NO guid               0   <- `if not guid: continue` NEVER fires
    guid != episode_id                  0   <- the audit did NOT write the wrong field
    filename with no leading digits     0   <- the other silent skip also never fires

    substack:post:193375117 -> guid AND episode_id are the SAME string
    _on_disk_guid_index('/app/output') -> 2002 entries, 238 substack guids, sampled ids all True
    per-feed indices: 42 of 42 work-list ids resolve, 0 unresolved

Both proposed fixes — an `episode_id` fallback in the index, and emitting both fields from
`transcript_pairing_audit.py` — would address something that does not occur on this corpus.
**"42 repairable" really was 42.**

**The real explanation is FEED SCOPE, and it fails silently.** The `substack:` prefix does not
imply a substack feed. The 11 split across two; the full 42 span eight:

    rss_feeds.megaphone.fm_755f5437   7      rss_feeds.simplecast.com_999571cd  5
    rss_feeds.simplecast.com_2e104dc7 6      rss_feeds.megaphone.fm_370fb395    5
    rss_rss.flightcast.com_c63dc3c0   6      rss_feeds.megaphone.fm_3581c092    5
    rss_api.substack.com_8c774140     5      rss_feeds.npr.org_7ce5b183         3

Six of the eleven live in `rss_rss.flightcast.com_c63dc3c0` — a flightcast feed serving
`substack:` guids. Prod loops feed_targets with `output_dir = feeds/<slug>` while EVERY feed
carries the WHOLE work-list, so an untargeted feed never looks: `scraping.py:577` logs "none of
the N listed episode(s) are in this feed's corpus" and returns `[]`, by design. Six episodes can
vanish with no error anywhere.

Action: target all eight slugs (or run corpus-wide), then read the end-of-batch work-list report
— it counts against the 42 denominator and is the only line that reveals a miss.

NOT verified: which feeds the actual repair run targeted (that is in the other agent's run). If
it was corpus-wide, feed scope is not the explanation.

**Method note.** Grepping `.../metadata/*.metadata.json` misleads: it matches the FILE, but the
index keeps only the newest run per episode (`dedupe_metadata_paths_newest_run_per_episode`,
`scraping.py:384`), so a hit in an older run dir does NOT mean the id is indexable. Query the
index, not the filesystem. Selection matches on `guid` OR `episode_id` (`scraping.py:559`), so
any check should use both.

---

## Open / next

- Handed to the other agent 2026-09-22: the work-list / feed-scope answer above (F13).
- **Step 0a** `transcript_pairing_audit` dispatched (run 35700971827), awaiting approval.
  Expect 147 of 2,297 mispaired, 42 repairable, 105 needing re-ingest; **exits 1 by design**.
- The **59 hand-read demotions are still unseen** — only counted. They come from
  `speaker_migration_preview` at step 1c, AFTER the 1b repair. Handover: if a REAL PERSON is in
  that list, STOP (expected to be all show names: MLST, Conversations with Tyler, Trivium China,
  Turkey Book, Africa Tech Summit).
- No `.upgrade-ledger.json` found on prod yet — consistent with "no migrations applied", but I
  have not confirmed where the ledger is written. Worth knowing before step 2 so its
  before/after state is readable.
