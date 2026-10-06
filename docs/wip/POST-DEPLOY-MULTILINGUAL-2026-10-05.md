# Post-deploy handover — PR #2260 (multilingual, overrides, admin security)

Written 2026-10-05 for the agent that deploys PR #2260 and works the corpus afterwards. Every
step names the command it runs. Facts marked **not verified** were not checked from the machine
that wrote this (no prod SSH there); check them first rather than trusting them.

## What changes on prod when this ships

- **A feed with no usable `<language>` is refused before download.** It stops ingesting until an
  operator override sets its language. Nothing already published is touched.
- **`operator.closelistening.app` serves only a coming-soon page.** Operator work happens on the
  control-plane viewer over the tailnet (MagicDNS host).
- **Operator overrides** are written through `/api/feeds/overrides` on the control plane only,
  stored in `<corpus>/overrides.json` beside `feeds.spec.yaml`.
- **The player API trusts only `https://closelistening.app`** for cookie writes and CORS
  (native-app origins stay CORS-allowed; the app uses a Bearer token).
- **The DGX profiles hold instead of failing over** (#2178): a DGX outage stops ingest rather
  than falling back to local Whisper.
- **GI/KG fail hard** (`TranscriptBodyMissingError`) when an episode names a transcript that is
  not on disk; the episode stops instead of producing an artifact from empty text.

## Order of operations

1. **Merge PR #2260 to main.** Images publish only from `main`; one tag pins all three services.
2. **`deploy-config`** (Caddy + Alloy). This installs the closed `operator.caddy`. It is NOT part
   of `deploy-all-prod`, and it must run before the operator deploy: the operator live smoke now
   asserts the host is closed and fails against the old gated vhost.
3. **`deploy-all-prod`** (control plane, player, operator). Its smokes cover:
   - player `account.live` — includes cookie writes, so a CSRF/origin regression shows up here
     as a 403 "Cross-site request refused.";
   - operator smoke — every path on the operator host returns coming-soon.
4. **Migrations — dry run first, then decide** (below). Deploy does not run them.
5. **Overrides for feeds with no language** — before the first scheduled ingest after deploy,
   or those feeds silently stop getting new episodes (the refusal is logged, not alerted).

## Step 4 — the corpus migration (m0021)

m0021 is the only new migration in this PR. It fetches each show's RSS `<language>` (one HTTP GET
per show) and writes the normalised code onto every existing episode's `metadata.json`
(`feed.language`, `feed.language_raw`, `feed.language_source`, `episode.language`,
`episode.language_source`). Backed up and receipted per show; undoable.

Run inside the control-plane api container (`-p compose`), corpus at `/app/output`
(**path not verified on the live box**):

```bash
python -m podcast_scraper.cli upgrade status --corpus-dir /app/output
python -m podcast_scraper.cli upgrade run    --corpus-dir /app/output --dry-run
```

The dry run **fetches** and prints, per show, the language it would write, plus every show that
"declares no `<language>`" or "declares '…', which is not a usable tag". Use that output for:

- **The override list (step 5):** every show it reports with no usable language.
- **A gate before applying:** any show it would stamp with a language OTHER than `en`. Production
  content is English as far as anyone knows, so a non-`en` show is either a genuinely
  non-English feed or a publisher tag that is wrong. For a wrong tag, set the show's language
  override (step 5) FIRST: m0021 reads `overrides.json`, does not fetch a show that has a
  feed-level language override, and writes the override with `language_source: override`.
  Re-run the dry run afterwards; the report counts "show(s) set from overrides.json".

So: dry run → set overrides (step 5) → dry run again → apply.

Apply, with a snapshot on a persistent path (the default sibling path is container-ephemeral):

```bash
python -m podcast_scraper.cli upgrade run --corpus-dir /app/output --yes \
  --snapshot-dir <persistent path>
```

Then `make corpus-language-audit CORPUS_DIR=…` (read-only). It exits 1 when no episode resolved
from `rss`, which after m0021 means the backfill did not take.

## Step 5 — set overrides for feeds with no language

The endpoint is on the control plane only (tailnet). Each call needs an admin session or the
operator key (`X-Operator-Key`, `APP_OPERATOR_API_KEY`). **Not verified:** whether prod's control
plane has that key set — check `/run/secrets` / the compose env first.

```bash
# list
curl -sS -H "X-Operator-Key: $KEY" \
  "https://<control-plane MagicDNS host>/api/feeds/overrides?path=/app/output"
# set one feed's language
curl -sS -X PUT -H "X-Operator-Key: $KEY" -H 'Content-Type: application/json' \
  "https://<control-plane MagicDNS host>/api/feeds/overrides/feed?path=/app/output&url=<RSS URL>" \
  -d '{"language": "en"}'
```

The value is validated (ISO code, unknown fields refused with 422) and every change is audited in
`<app_data_dir>/audit.jsonl` with `via`/`by` and before/after. Set `en` only on feeds whose content
is known to be English — that is the point of the rule.

Reference count, NOT prod's: of the 72 feeds in `config/corpus-expansion.feeds.yaml`, 5 declare no
`<language>`. Prod's own feed list has never been measured.

## Step 6 — verify

| check | how | expected |
| --- | --- | --- |
| no-language refusals are the ones you expect | pipeline logs: `no language:` / `refused:` | only feeds without an override |
| English episodes still ingest | next run's `result: episodes=… ok=… failed=…` | no new failures |
| operator host closed | `curl -s https://operator.closelistening.app/api/health` | coming-soon HTML, not JSON |
| player cookie writes | player `account.live` smoke | green |
| GI/KG hard failures | logs: `TranscriptBodyMissingError` | none, or each one is a real missing transcript to fix |

## Things that look like regressions but are expected

- **Every transcript-cache entry written before this deploy misses once.** See the next section.
- **`pipeline_composition_version` changes once for every new episode**, English included (a
  `translation` stage was added). A reprocess query keyed on the old hash must be reissued.
- **New sidecars per diarized episode:** `<base>.turns.json`, `<base>.adfree.turns.json`,
  `<base>.anon.txt`.
- **New `metadata.json` fields:** `feed.language_raw`, `feed.language_source`,
  `episode.language`, `episode.language_source`.
- **The manifest gains `translation` (`ran=false` for English) and `turns` blocks.**
- **Run metrics gain `llm_translation_*` counters, all zero for English.**

## The transcript-cache key changes once

The transcript cache is keyed by the audio hash plus the provider name and model. On main, the
DGX profiles wrap Whisper in a failover chain, and every entry is stored under the wrapper's name,
`fallback_chain`, whichever tier actually transcribed. This PR (ccb75ad3e) keys on the provider
that produced the transcript: a lookup uses the primary tier (DGX Whisper, key name
`tailnetdgxwhispertranscription` plus its model), and a save uses the tier that ran. The
resilience strategy (`hold` or `failover`) no longer affects the key. As a result, none of prod's
existing entries match after the deploy. Re-running `prod_dgx_full` over an episode that was
already transcribed transcribes it once more on the DGX and stores it under the new key; every
run after that hits the cache. The output is the same transcript, not a different one: the real
A/B below found the DGX transcript byte-identical between main and this branch. The old entries
are not re-keyed because `fallback_chain` does not record whether DGX or a fallback tier made
them, and re-keying could serve a fallback transcript as a DGX one. For the same reason, a
transcript produced by a fallback tier is now stored under that tier's name and is never served
to a DGX lookup, so the next run with a healthy DGX transcribes it again. New episodes are
unaffected (they have no entry yet), and so are profiles that run with the transcript cache off.

## How this PR was tested before deploy

The question was whether English behaves exactly as on main. Language now flows through the whole
pipeline, so a language tag that one stage resolves differently would change English output
silently. That was checked with a tool built for it, not by reading LLM output.

**The tool:** `make pipeline-check`, code in `scripts/validate/pipeline_check/`, runbook
`docs/guides/PIPELINE_CHECK.md`, issue #2287. It lives in **PR #2288 (branch
`feat/pipeline-check`), which is not merged**: to re-run it, check out that branch. It runs two
code refs (each in its own worktree) over the same input. It records every decision the pipeline
makes on language (every lookup in a language-keyed map and every call to a function that takes a
language, with file and line). It reports a **hole** whenever an English variant (`en-US`,
`en_GB`, `English`, `eng`, an override) resolves to something other than plain `en`, and it
compares each deterministic stage's output against the base ref.

**Results** (full tables:
[#2287 comment](https://github.com/chipi/podcast_scraper/issues/2287#issuecomment-6005493868)),
candidate 1a8a493f0 vs `origin/main` 712ff7ebb:

| run | what it covered | result |
| --- | --- | --- |
| fixture mode (no DGX) | 40 English fixture episodes × 6 tag forms | 71,664 decisions, **0 holes**; hosts, speaker naming, ad removal and transcript selection identical to main in 240 of 240 cases |
| mutation check | the same check against 05ab3c8a5, before a real `en_US` host bug was fixed | **24 holes** reported in host detection (`hosts.py:851`), while that stage's output still matched main, so the tool catches what an output diff misses |
| real mode (DGX, `prod_dgx_full`) | one Japan Memo episode, full pipeline, main and branch | transcript and speaker records **byte-identical**; 4,551 decisions, **0 holes**; artifacts identical apart from the expected list below |
| LLM variation (same run) | main run twice, branch once, on one identical transcript | summaries, insights, quotes, bullets inside the band (main's own run-to-run spread, at least 10%); KG entity names differed more, and the KG input text was byte-identical on all three runs, so that variation comes from the model |
| PR CI on 1a8a493f0 | the repo's CI | 31 success, 4 skipped, 0 failed |

**What this does NOT cover:**

- **The cache-key change (ccb75ad3e) came after these runs.** It is covered by unit tests
  (`tests/unit/podcast_scraper/workflow/test_transcript_cache_key_is_the_factual_provider.py`), not
  by an A/B run. It changes which cache entry is read, not how a transcript is made.
- **One real episode only**, from one English feed. Feeds with other tag forms were exercised in
  fixture mode only.
- **LLM output quality** is not judged: only its variation against main's own run-to-run spread.
- **Search ranking** on the prod corpus was not measured (it needs the ML stack or a prod snapshot).
- **Decision points that the driven stages never reach** are counted and listed in the JSON
  report, but not checked.
- **Non-English output** was not compared: the checks guard English against main.

## Not this agent's to start (operator schedules)

- **#2187** — ASR has never run on non-English audio; needs a quiet DGX window.
- **#2251** — the translation model pick is preliminary.
- **#2255–#2259** — the non-English detector vocabularies are unmeasured.

## Known open items carried from the English-path audit (2026-10-05)

Fixed before deploy (in PR #2260):

- **m0021 honours `overrides.json`** (fd10991ab) — feed-level language overrides only;
  `metadata.json` has no `<guid>`, so episode-level overrides apply on the episode's next run.
- **Every language-tag reduction goes through `languages.primary_language`** (bd817a3d4,
  d63db3c4d) — `en_US` / `English` / `eng` read as English everywhere.

Decided, no change: keyword search reads the original-language text (`segments_nonen`) as well
as the English layer. A query's language cannot be known ("AI" is the same word in many
languages), so searching every language is the intended behaviour (operator, 2026-10-05).
