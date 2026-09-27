# Handover 2026-09-27 — the #2097 post-deploy chain, and four bugs that were deleting data silently

**Read §1 and §2 before touching prod.** The post-deploy steps are order-dependent and three of them
have landmines that destroy accumulated work if run in the wrong order or with the wrong flag.

Everything below was measured on prod during 2026-09-25/27. Anything unmeasured says so in §8.

---

## 1. Where things are

| | state |
| --- | --- |
| `origin/main` | `3860811461ccf12a0911b6eeaf69d39208b4ca1a` |
| main CI | **fully green** — 10 runs, all success, including `Python application` and `Stack test` (the two the PR never exercised) |
| deploy backlog | **17 commits** ahead of the running image |
| prod image | `ghcr.io/chipi/podcast-scraper-stack-api:sha-e2dedbd` (last verified 2026-09-26; re-check before deploying) |
| prod job queue | empty — 427 jobs, all terminal |
| `.viewer/jobs.paused` | **removed** (was blocking a job for 3.5 days) |
| nightly scheduler | `nightly-ingest … enabled: false` — still off, deliberately last in the chain |

PR #2159 merged as a squash (`386081146`). Ancestry cannot confirm a squash — verify by content if you
need to: all nine changed files are byte-identical to `origin/main`.

## 2. THREE THINGS THAT WILL DESTROY WORK — do not do these

### 2.1 Do NOT run `person_web` / `org_web` with `refresh=true` (or the API's `force`)

`person_web.py:1044` — `refresh = bool(config.get("refresh", False)) or bool(getattr(delta, "forced", False))`
and `:1020` — `known = {} if refresh else _existing_person_rows(corpus_root)`.

It discards **879 derived person rows and 507 miss records** and re-fetches everything against an
upstream already returning 429s. Local task #34 used to say "re-enrich person_web with refresh=true";
it has been rewritten to say the opposite. The 281 misses do **not** need it — see §4.2.

### 2.2 Do NOT repair the search index before the deploy

Proven the hard way on 2026-09-26: dropping one episode's fingerprint and re-running the incremental
build restored **exactly 130** docs (`total_vectors` 370,969 → 371,099), and a later unrelated build
**deleted the same 130 again** (`pruned 130 stale row(s) … across 2 re-indexed episode(s)`).

The guard that prevents this is in `31467274`, on main, **not yet deployed**. Repairs are not durable
until it ships.

### 2.3 Do NOT batch `retranscript_only` before verifying ONE episode

Carried from #2100 and still outstanding: `retranscript_only`'s write + relabel half **has never been
run against a real episode**. Verify on one before any batch.

## 3. The deploy

Deploy `3860811461cc` and **pass `image_sha` explicitly**. A blank value resolves to "newest
published", which may not be this commit — `DEPLOY_GOTCHAS.md` §5 says the same thing.

## 4. Post-deploy sequence — the order is load-bearing

### 4.1 Re-repair the two index episodes

Two episodes lost their content-keyed derived rows to the prune bug:

```text
Buzzsprout-19721665                    insights 2 (disk 43), quotes 2 (disk 67), kg_topic 0 (disk 10)
9d680dae-1e1e-11f1-a58e-cbc963e1c1a2   insights 4 (disk 10), quotes 4 (disk 23)
```

Procedure (per episode): back up `search/episode_fingerprints.json`, remove that episode's single key,
then `POST /api/index/rebuild?rebuild=false`. Durable only once `31467274` is deployed.

Reaching the endpoint: it is mounted on the **tailnet operator serve** (`compose-api-1`), not the
public operator surface. Requires the `X-Operator-Key` header; the key is `APP_OPERATOR_API_KEY` in
that container's **PID 1 environment** (64 chars) — readable as host root via `/proc/<pid>/environ`,
NOT via `docker exec` (which inherits an empty value from `Config.Env`). That discrepancy cost an hour;
do not re-derive it from `docker inspect`.

### 4.2 WEB-tier enrichment passes — now they actually advance

Before the fix, coverage was hard-capped and no amount of passes could move it:

```text
person_web_raw by first letter:  a=334 b=153 c=214 d=232 e=140 f=61 g=132 h=95 i=43 j=243
                                 -> NOTHING from k to z
org_web_raw:                     digits=82  a=533  b=71   -> NOTHING from c to z
```

Run plain passes with **force OFF**: `enrich --only person_web,org_web`. Repeat until derived-row
growth stops. Expect ~200 new entities per pass. Track derived rows in
`enrichments/person_web.json` → `data.persons` and `enrichments/org_web.json` → `data.orgs`
(879 and 217 at handover).

The first pass after deploy also **backfills the reason envelopes** over the 479 already-cached
dead payloads — no separate migration needed.

### 4.3 `retranscript_only` over the 51 text-prefixed episodes — no GPU

50 Odd Lots + 1 In Moscow's Shadows. Their publisher WebVTT names each turn as cue **text**
(`Speaker 1:`) rather than a `<v>` tag, and `parse_webvtt` had no branch for it, so the label stayed
embedded in the prose: one 81 KB transcript held **280 literal `Speaker 1` strings** with
`speaker=None` on all 1,290 segments. Fixed in `d6c0de341`; the fix repairs **new parses only**, so
the stored transcripts need re-parsing.

No audio, no GPU, no re-ASR — the turn data is already in the publisher's file. **Verify on one first
(§2.3).**

Scope limit: a further **87** episodes are newline-less with *no* turn structure at all. Re-parsing
cannot help them; they stay with #2098 / #2099 / #2100 and that GPU question is untouched by this.
These 51 are **not** #2100's 51 (Explaining Brazil / The Flip / Korea Deconstructed) — the matching
count is coincidence.

### 4.4 The attribution artifacts — 26, not 27

Work-lists live on the prod corpus root:

```text
step7_relabel.txt          16   -> relabel_only
step7_enrich_edges.txt      4   -> enrich-edges --replace-speakers
step6_attribution_defect.txt 27 -> the parent list
```

Of the 7 unrouted remainder: **6 are the Odd Lots text-prefix case** (route via §4.3, no GPU — not
`rediarize_only` as previously assumed), and **1 is a false positive** —
`8e30cc48-c94f-47b3-b450-b4ca014b861f` is "Introducing: Bloomberg Money", a 30-second trailer with 634
bytes of transcript and one voice, which is correct.

### 4.5 Re-enable nightly — last

`viewer_operator.yaml` → `scheduled_jobs` → `nightly-ingest` → `enabled: true`. Last because it
reprocesses, and reprocessing is what triggered the index damage.

## 5. Epic map for continuation

### #2097 — EPIC (ops): post-deploy repair/migrate/re-derive

Its own "done when" list is the gate. Status added 2026-09-26 (2 comments):

- **done**: re-enrichment unblocked and run (5 corpus enrichers current), job queue unpaused, the
  corpus-wide index completeness audit (found 2 victims, not 1)
- **done then undone**: the index repair (§2.2)
- **open**: nightly, the 26 artifacts, the 5 vouching refusals (needed the deploy), the 5 invented
  person ids, the 22 collapse false positives (now fixed in code — see §6)
- **new action**: §4.3's 51-episode re-parse

Children: #2082 (pairing — code done, unpushed audit verification pending), #2094 (D4/D5/D6 re-check
— blocked on a trustworthy corpus), #2100 (the 49/51 split + the blocking pre-check).

### #2096 — EPIC: host/guest attribution PARTIAL → FULL

Three code items were added from this arc (comment 2026-09-26), plus a fourth:

1. `check_not_collapsed_onto_one_speaker` counted 22 interviews as damage — **fixed** in `cc40ab043`
2. the 7 generic `Speaker 1` episodes — **diagnosed**, 6 route to §4.3, 1 is a trailer
3. the "5 invented person ids" — **diagnosed and far larger**, see §7
4. malformed transcript text degrades attribution as well as chunking (GI keys on `char_start`
   against line-start `Name:` markers)

Untouched children: #2095, #2092, #2093, #2078, #2076, #2102, #2101. **#2101 is the cheapest open
item here** — a one-word fix measured at zero collateral across all 55 feeds.

### #2158 — person_web/org_web: record WHY an entity is empty (opened this arc)

Items 1-3 and 5 shipped (`60989c335`). Still open: the `not_an_org` TTL decision (currently uniform at
30 days), an ambiguity scorer *if* the §4.3 re-fetch leaves a residue, and the chunker's upstream text
defect. Carries the unexplained partial-emission mechanism (§8).

## 6. What shipped in PR #2159 (6 commits)

| commit | what it fixes |
| --- | --- |
| `60989c335` | WEB enrichment coverage ceiling — dead entities held the 200-slot budget forever, so coverage stopped at person "j" / org "b" |
| `704f895f9` | the prune deleted healthy rows on a *partial* emission (it guarded the empty case only) |
| `fa0c94f7c` | a boundaryless transcript became one 44,924-char chunk; MiniLM truncates at 512 tokens, so ~8% of a 48-min episode was searchable |
| `cc40ab043` | the collapse check counted 22 interviews as damage (exempt guest-only when a distinct host exists; host-only still reported) |
| `d6c0de341` | `parse_webvtt` dropped the speaker when the cue named it as text — `parse_srt` always handled it |
| `f2845a91` | coverage for every defensive branch the above added (0 uncovered added lines) |

All five behavioural fixes are **mutation-checked**. One test written during the arc did *not* survive
that check and was reverted along with the change it defended, rather than shipped as false coverage.

## 7. Diagnosis: the "5 invented person ids" are ~107

Of **1,459** distinct persons published in a SPEAKER role: **107 appear on no roster**, **86 are not
person-shaped**. Four classes, full detail on #2096:

- a show as its own host — `The Brazilian Report` on **40 episodes**. `check_no_show_as_speaker`
  cannot fire: it compares against the **feed title**, and the brand differs ("Explaining Brazil")
- a dead historical figure as a **guest** — `H.B. Reese` (died 1956); the same episode's other run
  correctly records him `mentioned`
- a sentence fragment as a **host** — `'Thank'`, node `person:unresolved-thank-…`, published anyway
- `World Bank` as a **host**, plus ASR mis-hearings minting duplicate identities
  (`Kaiser Guo`/`Kuo`, `Daniela Stockman`/`Stockmann`, `Mark Seidel`/`Sidel`)

**The structural point:** the roster is the entry point for three of the four classes, and every other
coherence rule validates *against* the roster — so they all agree with the error. This is the concrete
evidence behind this epic's standing rule: **do not use the roster as ground truth.**

## 8. NOT verified / unexplained — do not treat as diagnosed

- **Why a build emits a PARTIAL row set for an episode.** Two theories tested, **both wrong**: a
  half-written artifact (ruled out — it reproduces, and nothing wrote to the corpus between builds)
  and a walk visiting every run dir (ruled out — `discover_metadata_files` already applies the
  newest-run dedupe). Prod logs `duplicate row id 'episode_title:…__Buzzsprout-19721665' (keeping
  last)` while only ONE metadata record exists for that episode. **The deployed guard's new
  per-episode warning is the next evidence** — it names the episode and both counts.
- Whether the other 36 single-segment episodes share the run-together-text cause.
- The publisher `.vtt` source itself — not retained on disk, so the text-prefix conclusion rests on
  the parser's documented behaviour plus the stored-artifact shape, not on reading the source.
- Whether the 51 text-prefixed episodes actually recover once re-parsed. Nothing has run.
- Whether the ~40 single bare first names (`Joe`, `Anna`, `Ilya`) are wrong at all — many are likely
  correct; they are flagged as weak ids, not errors.
- Local `podcast-content` MCP `corpus_status` returns `{"ok": false, "note": "AttributeError"}` while
  the prod one works. Local-only, uninvestigated.

## 9. Operational notes worth not rediscovering

- **SSH to prod**: `ssh -i ~/.ssh/podcast_prod_operator root@prod-podcast` (or `deploy@`). Bare
  `ssh prod-podcast` fails with `Permission denied (publickey)` because it defaults to the wrong user.
  The key is passphrase-protected with a ~6h agent lifetime — when it drops out, `ssh-add` it again.
- **`make ci-fast` locally**: needs `PYTHON=.venv-dev/bin/python` AND `env -u NODE_OPTIONS` (a stale
  preload path breaks markdownlint). `mkdocs` lives in `.venv`, not `.venv-dev`, so run `make docs`
  with that interpreter rather than installing anything.
- **zsh does not word-split unquoted variables.** `for b in $LIST` iterates ONCE with the whole
  string. This nearly turned a branch-deletion loop into a no-op that reported success.
- **A monitoring loop that finds zero matching runs must not report green.** One did, and claimed main
  was settled while three runs were still going. Confirm any watcher's verdict with a direct query.
- **`gh run list --commit <sha>`** beats filtering with jq.

## 10. Repo hygiene done this arc

Remote branches went **24 → 10**: 15 deleted (10 merged-PR, 2 closed-but-landed, 2 superseded, 1
merged), plus one local-only. Everything remaining is `main`, an open PR, or `release/2.6` (kept
deliberately).

The seven local `backup/*` branches are **kept on purpose** until #2127 merges — they are the safety
copies of that arc, and the pre-rebase copy of it was among the branches deleted.
