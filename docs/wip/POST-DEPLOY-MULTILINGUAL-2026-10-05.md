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

- **Transcript-cache entries written before this deploy miss once.** The cache is now keyed by
  the provider that actually transcribed (e.g. `tailnet_dgx_whisper` + model), never by the
  failover wrapper; old entries are keyed `fallback_chain`, which does not say whether DGX or a
  fallback tier made them, so they cannot be re-keyed safely. Re-running `prod_dgx_full` over an
  already-transcribed episode transcribes it once more; every entry written after the deploy is
  stable whatever the resilience strategy. New episodes and the reprocess profiles (cache off) are
  unaffected.
- **`pipeline_composition_version` changes once for every new episode**, English included (a
  `translation` stage was added). A reprocess query keyed on the old hash must be reissued.
- **New sidecars per diarized episode:** `<base>.turns.json`, `<base>.adfree.turns.json`,
  `<base>.anon.txt`.
- **New `metadata.json` fields:** `feed.language_raw`, `feed.language_source`,
  `episode.language`, `episode.language_source`.
- **The manifest gains `translation` (`ran=false` for English) and `turns` blocks.**
- **Run metrics gain `llm_translation_*` counters, all zero for English.**

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
