# Corpus integrity repair runbook (#1657)

**The ordered sequence for repairing a corpus damaged by the defects epic #1657 fixed —
which detector finds each, which repair fixes it, and which repairs fix nothing.**

For general "how do I reprocess a corpus" mode selection (`relabel_only`, `rediarize_only`,
`enrich_only`), see [CORPUS_REPROCESSING.md](CORPUS_REPROCESSING.md). This runbook is narrower:
it is the specific, gated procedure for the known damage, and it exists because the obvious
approach — "just reprocess the corpus" — was rehearsed and **does not work**.

---

## Read this first: what "reprocess the corpus" does NOT fix

`make reprocess-corpus-from-transcripts` runs with `transcribe=off` and `diarization=none` by
design. It re-runs speaker naming, summary, GI, KG. That means:

| Damage | Layer | Fixed by a reprocess? |
| --- | --- | --- |
| Placeholder GI artifacts (#19/#1657) | GI | **No** — the episode is *skipped*, see below |
| Dedupe bar too high (#27) | GI | Yes, on any GI re-derivation |
| Composite host names (#17/#1646/#1652) | naming | Yes |
| Stage ledger / provenance (#1647/#21/#30) | recorded on any run | Yes |
| **Transcript from unpreprocessed audio (#18/#558)** | **transcript** | **No — nothing we have fixes it except re-transcription** |
| Diarization turn boundaries | diarization | No (`diarization=none`) |
| Enrichment, relational edges, search index | corpus-level | No — separate passes, see step 7 |
| **Episode persisted with NO summary (#1686)** | summary | **Only via the work-list** — see below |
| **Bare person name minted as a global id (#1685)** | KG + GI | **Yes — `upgrade run` (m0007)**, see below |

Two of those need spelling out, because both were verified the hard way.

### Placeholder episodes are SKIPPED, not repaired

Rehearsed 2026-08-16 on a copy of a real corpus. Four flag combinations, none reprocessed a
placeholder episode:

```text
--skip-existing                                    -> "no transcript for ..."
--skip-existing --single-feed-uses-corpus-layout   -> transcript found, still skipped, GI:0
--reprocess-existing-only                          -> "no transcript for ..."
--reprocess-source whisper_transcription           -> force path never fired (that was #33)
```

The skip predicates key on the **presence** of a transcript/metadata file and never look at GI,
so a placeholder artifact reads as "this episode is done". Deleting the artifact does not help
either — transcript and metadata still satisfy them. This is what `gi-repair` (step 3) exists for.

### An episode with no summary has to be named explicitly

Summarisation is allowed to fail without losing the episode (#1496): the episode keeps its
transcript, GI and KG, and is written with `summary: null`. That is deliberate. What was NOT
deliberate is that nothing afterwards could find those episodes — production had 8 of them,
indistinguishable from healthy ones, discovered months later by an audit rather than by the
pipeline (#1686).

Three things changed:

1. The stage ledger now records it. Every path that ends with no summary writes
   `summarization: {outcome: "failed", reason: "<slug>"}`, where the slug says WHICH failure
   (`tokenizer_threading`, `provider_content_rejected`, `schema_invalid_after_reroll`,
   `prompt_examples_leaked`, `unspecified`). Before this the ledger's only `summarization`
   entry was `deadline_exceeded_but_completed` — a summary that was *late but real* — so it
   recorded the harmless case and stayed silent on the harmful one.
2. A transient cause is retried once, in-process, before the episode is written.
3. A summary that is genuinely lost is reported to Sentry at **error**, not warning.

```bash
# Which episodes are serving without a summary? Non-zero exit if ANY are.
make corpus-summary-audit CORPUS_DIR=/path/to/corpus

# The work-list to feed --reprocess-episode-ids.
# Writes <corpus>/summary_repair_worklist.txt (+ a .terminal sidecar).
make corpus-summary-worklist CORPUS_DIR=/path/to/corpus
```

**Retryable vs terminal.** An episode that failed on one run is retryable. One that failed on
two or more has had its requeue and did not recover — it goes to the `.terminal` sidecar and
needs a person, not another dispatch. Terminal is durable (the sidecar is read back on the next
pass) but not permanent: if the episode later gains a summary, it leaves both lists.

**The loop guard.** An episode that was dispatched and whose attempt count did NOT rise is
escalated to terminal as `no-progress`. This covers the case where the requeue dies before
writing anything: the corpus stays byte-identical, the audit re-derives the same answer, and an
automated dispatcher would otherwise retry it forever at provider expense. A hand-edited
work-list that has lost its `# attempts=N` comments escalates for the same reason — an
unprovable count is treated as no progress, not as a fresh start.

**Verified**, on a copy of the 36-episode validation corpus with one summary knocked out:
`make corpus-summary-audit` exits 1 and names the episode and cause; `make
corpus-summary-worklist` writes exactly that one id; ten consecutive work-list passes against
an unchanging corpus dispatch it once and then stay silent.

**Not verified**: whether a plain `make reprocess-corpus-from-transcripts` (no
`--reprocess-episode-ids`) also picks these episodes up. `generate_summaries=true` does bypass
the transcript and metadata skip checks in the code, so it may — but that would re-summarise the
WHOLE corpus at provider expense, which is why the work-list exists. Use the explicit list.

### A bare first name became a global, followable person

`person:jensen` is not a person. It is what happens when three different shows each say "Jensen"
with no surname: `entity_node_id()` slugs whatever string it was handed, three identical slugs
collide, and the result is one **followable** node pooling three people. `POST /interests/{token}`
accepts `person:` tokens, and derived interests mint them straight from `entities_from_kg`, so a
pooled token can enter a profile with no click at all.

Measured on production (678 episodes): **208 occurrences of 172 single-word person ids — 12
resolvable within their own episode, 0 ambiguous, 196 orphan.** Hand-checked, they are three
shapes: hollow (`person:jensen` — no insights, topics or quotes at all), pooled
(`person:sam` — carrying a stated position about *Samuel Moyn*), and locally resolvable
(`person:alex`, whose full name `person:alex-mayassi` is a co-speaker in that same episode).

The pipeline no longer mints them (`identity/bare_name_scope.py` runs during metadata generation,
before typed mentions). Existing corpora need the backfill:

```bash
# What would change, no writes:
podcast-scraper upgrade plan --corpus /path/to/corpus

# Apply (m0007 among any other pending steps):
podcast-scraper upgrade run --yes --corpus /path/to/corpus
```

**What it does.** Pairs each episode's `.gi.json` and `.kg.json` (the roster is split across both
layers — the KG alone reports `person:alex` as an orphan while the full name sits in the GI), then
per episode: exactly one full-name candidate → **heal** to the full id; two or more, or none →
**scope** to `person:unresolved-{name}-{episode}`. Scoped ids match
`is_unresolved_speaker_placeholder`, so they stop appearing as entity cards, follow targets and
derived interests.

**Verified** on a copy of the 36-episode validation corpus: 7 bare ids → 0, dry-run wrote nothing,
re-apply was a clean no-op, and a post-migration check found **0 dangling person edges and 0
duplicate node ids** — identical to the untouched baseline.

**ORDER MATTERS: deploy the new image FIRST, then migrate.** The `person:unresolved-…` branch of
`is_unresolved_speaker_placeholder` ships with the image. Running m0007 against a corpus an OLDER
image is still serving means that image does not recognise the scoped ids — every one renders as
a followable entity card (scoping preserves `name`, so it shows as "Jensen"), and its derived
interests can mint `person:unresolved-…` tokens into profiles. That is strictly worse than not
migrating. The reverse order is safe: a new image over an unmigrated corpus behaves exactly as
today, and a half-migrated corpus is fine.

**Rebuild BRIDGES too, not just enrichment and the index.** m0007 rewrites `.gi.json` and
`.kg.json` only. `*.bridge.json` carries `person:` CIL ids and the CIL read surfaces walk every
bridge file at request time (`server/cil_queries.py` — timeline, position-arc, conversation-arc),
so after migration they keep serving pre-migration bare ids that no longer exist in the KG: dead
cross-references, silently. `build_bridge` runs from metadata generation, so the practical options
are a reprocess pass over the corpus or accepting the staleness knowingly. **Decide explicitly —
this step is easy to skip because nothing errors.**

**Existing follows are not migrated.** Interests are per-user token lists in state files; m0007
never touches them. A followed `person:sam` matches zero episodes afterwards — silently, no error.
There is no tool for this; it needs a one-time sweep or a deliberate shrug.

**Check for real mononyms before running it.** The plan output counts but does not name. Grep the
corpus for single-token person ids first: scoping a genuine mononym (Grimes, Beyoncé) is the one
case where this destroys a legitimate follow target, and no allowlist exists yet.

**Afterwards, rebuild enrichment and the search index.** Ids changed, so anything derived from
them is stale. Per the corpus-index note: delete `episode_fingerprints.json` alongside
`lance_index`, or every episode is skipped and you get a silent empty index.

**Not fixed by this.** Scoping stops the harm; it does not say *who* the person was. That is
tracked separately (#1801) — the enricher would read `unresolved_persons_in_episode()`, which
reports `ambiguous` (candidates listed, a bounded choice) versus `no_candidate` (the answer is not
in the graph at all — 196 of the 208).

**Also unchanged:** the 180 shared surnames and 420 prefix-overlapping ids are a different
problem — full names disagreeing with each other, not names that identify nobody.

---

### Transcript damage survives every repair we have

`gi-repair` rebuilds insights **from** the transcript. `reprocess-corpus-from-transcripts` reuses
transcripts by design. So an episode whose transcript came from unpreprocessed, oversized audio
keeps that transcript and everything derived from it. Only **re-transcription** fixes it, and that
is the expensive stage (~$2.88 of a $3.48 14-episode run). Steps 4–6 make that a costed decision
instead of an invisible gap.

---

## The sequence

Run in order. Steps 1–2 are baselines; do not skip them, because steps 8–9 compare against them.

### 1. Baseline: GI integrity

```bash
make corpus-gi-integrity-check CORPUS_DIR=<corpus>
```

Per **episode** (scoped to corpus members — newest run per episode, using the project's own
`dedupe_metadata_paths_newest_run_per_episode` rule), asserts the declared GI actually exists,
parses, and is not a pre-#1657 placeholder. Non-zero exit on any failure.

Record the numbers. `legacy placeholders` is the work step 3 must clear.

> Do **not** use `make corpus-placeholder-check` as the exit criterion. It only asks "is the bad
> string absent?", which **passes** on a corpus whose artifacts were deleted and never
> regenerated — demonstrated 2026-08-16. It remains useful for producing the work-list only.

### 2. Baseline: unpreprocessed-audio damage

```bash
make corpus-preprocessing-audit CORPUS_DIR=<corpus>
```

Flags runs where `preprocessing_attempts >= 1 AND preprocessing_count == 0` — preprocessing was
asked for and produced nothing, so the original full-size file went to the STT provider.
Corroborated by `avg_preprocessing_wall_ms` sitting at the old flat 300 s budget.

Record the count. On a pre-#558 corpus this was **9 of 15 runs (60 %)**, all clustered at
297,064–300,845 ms. On a post-fix corpus, 0 of 14.

### 3. Repair the placeholders (cheap, safe)

```bash
podcast-scraper gi-repair --output-dir <corpus> --config <profile> --dry-run   # review first
podcast-scraper gi-repair --output-dir <corpus> --config <profile>
```

Standalone, corpus-driven, rewrites each `gi.json` **in place** — no RSS fetch, no new run dir, no
skip logic. Rebuilds from the episode's own run dir (ad-free transcript + matching segments
sidecar + summary bullets) and re-applies KG topic alignment (#585/#653) so repaired episodes
carry the same topic vocabulary as everything else.

A per-episode failure writes **nothing** for that episode, leaves the placeholder byte-identical,
and exits non-zero — so an unrepaired episode stays on step 1's red list and can never look
repaired. Audit trail lands at `<corpus>/gi_repair_report.jsonl`.

### 4. Produce the re-transcription work-list

```bash
make corpus-preprocessing-worklist CORPUS_DIR=<corpus>
# writes <corpus>/preprocessing_repair_worklist.txt
```

Episodes from **ambiguous** runs (several episodes, one run-level metric) are included
deliberately: the run demonstrably transcribed from raw audio and the metric cannot say which
episode, so over-repairing wastes money while under-repairing leaves the corpus wrong with
nothing downstream to reveal it.

### 5. DECIDE: re-transcribe or accept

**Operator decision. Nothing automates this.** Re-transcription re-runs ASR and cascades
diarization/GI/KG. ASR dominates pipeline cost.

Inputs to the decision: the step-2 count, the work-list length, and current ASR pricing. Accepting
the damage is a legitimate choice — but make it explicitly, and record it, rather than letting the
gap persist because nobody looked.

### 6. Re-transcribe the listed episodes (only if step 5 said yes)

```bash
podcast-scraper --config <profile> --feeds-spec <corpus>/feeds.spec.yaml \
  --output-dir <corpus> --skip-existing --single-feed-uses-corpus-layout \
  --no-transcript-cache \
  --reprocess-episode-ids <corpus>/preprocessing_repair_worklist.txt
```

`--no-transcript-cache` is **not optional here**, and it is the difference between this step
working and silently doing nothing. The transcript cache is keyed on the hash of the *original*
media plus `preprocessing_fingerprint(cfg)` — and that fingerprint is computed from **config**, so
it reads `pp=on|…` whether preprocessing succeeded or fell back to raw audio. Neither key
component changes between the damaged run and this repair run: the audio is the same file, and the
config always said `pp=on` — it was the *run* that failed. So without the flag this step scores a
cache hit and re-serves the exact transcript it was launched to replace, and every gate in step 8
then goes green on unrepaired data.

Three reprocess profiles (`reprocess_dgx_no_llm`, `reprocess_dgx_turbo`, `reprocess_v23_turbo`)
already set `transcript_cache_enabled: false`, so with those the flag is redundant. Pass it anyway
— it makes the command correct on its face rather than correct-if-you-picked-the-right-profile,
and `config/profiles/*.yaml` are generated (ADR-112) so the profile's value is not yours to fix.

New entries written *by* this run are unaffected: since #35, a run that asks for preprocessing and
falls back to raw audio no longer writes a cache entry at all, because the key it would write
under would misdescribe the audio. The flag covers entries written *before* that fix — which is
every entry behind the damage this step repairs.

`--reprocess-episode-ids` exists because `--reprocess-source` **cannot** express this set: on the
measured corpus all 9 damaged episodes were `whisper_transcription` and so were all 6 healthy
ones, so selecting by source would re-transcribe 6 healthy episodes to reach 9 damaged. Matching
is on **both** `episode_id` and RSS `guid`, since detectors emit whichever the artifact carries.

`--single-feed-uses-corpus-layout` is required for a single-feed corpus, or cross-run resolution
does not fire and every episode reports "no transcript".

**Scope is handled for you.** `reprocess_episode_ids` implies `reprocess_existing_only`, so the
run considers only episodes already on disk. Before that implication existed, a one-episode
work-list against one feed had preprocessed **12 unrelated episodes** before it was killed — the
run walked the feed and treated every item not on disk as new work. Across 14 production feeds
that would have been a large unbudgeted ASR bill.

Still pass `--max-episodes` if you want a hard ceiling on a first cautious run.

### 7. Corpus-level passes (these are NOT part of any reprocess)

```bash
podcast-scraper enrich-edges  --output-dir <corpus>
make index-two-tier  CORPUS_DIR=<corpus>
make topic-clusters  CORPUS_DIR=<corpus>
make enrich CORPUS=<corpus>
```

Skipping these leaves the MCP/viewer serving stale or empty relational and search results —
which is what `make corpus-completeness-check` exists to catch.

### 8. Re-run both gates — they must be green

```bash
make corpus-gi-integrity-check    CORPUS_DIR=<corpus>   # placeholders 0, healthy == episode count
make corpus-preprocessing-audit   CORPUS_DIR=<corpus>   # 0 damaged, IF step 6 was run
make corpus-completeness-check    CORPUS_DIR=<corpus>
```

Compare against the step-1/2 baselines. A count that did not move means the repair did not run —
which is exactly the failure mode this whole runbook exists to prevent.

**A green preprocessing audit is NOT sufficient on its own after step 6.** The audit's damage rule
is `preprocessing_count < preprocessing_attempts`, and it returns "not damaged" when `attempts`
is 0 — correctly, since a run that never attempted preprocessing has not damaged anything. But a
step-6 run that served every episode from the transcript cache also records `attempts: 0`, because
preprocessing lives inside the transcription path it skipped. That run is indistinguishable from
healthy to this gate while having repaired nothing.

So assert positively that work happened, against the run dirs step 6 created:

```bash
# every NEW run dir must show preprocessing actually attempted AND completed
find <corpus> -name metrics.json -newermt '-1 hour' -exec sh -c '
  echo "$1: $(jq -c "{attempts: .preprocessing_attempts, completed: .preprocessing_count,
                       transcribed: .transcribe_count}" "$1")"' _ {} \;
```

Expected on a real repair: `attempts >= 1`, `completed == attempts`, `transcribed >= 1`. All
zeros means the cache served it — go back to step 6 and check `--no-transcript-cache` is present.

### 9. Record what was decided

Note the before/after numbers and, if step 5 declined re-transcription, **which episodes remain
knowingly damaged**. Undocumented accepted damage becomes indistinguishable from an oversight the
next time someone audits.

---

## Prerequisites and gotchas

- **`DEEPSEEK_API_KEY`** — every configured failover ladder points at the `deepseek` tier. Without
  the key the ladder detects failures and recovers nothing. A startup pre-flight now warns
  (`FAILOVER LADDER BROKEN`) at every run start until it is set. Compose already forwards the
  variable; only the value is missing.
- **Profile alias must match the gateway.** `litellm_verify_served_model: true` refuses to run when
  the profile pins an alias the gateway does not advertise. This is correct behaviour — fix the
  alias, do not disable the check. Never hand-edit `config/profiles/*.yaml`: they are a generated
  view of the model registry (ADR-112); use `make profiles-materialize`.
- **Tailnet hostnames inside containers.** `litellm_api_base` may be a tailnet name the container
  cannot resolve even though the IP is reachable. Pass `--add-host <name>:<ip>`.
- **Container stdout can be dropped** through a relayed Docker socket. If a container run prints
  nothing, do not conclude it crashed — write results to a file inside and `docker cp` them out.
- **`make redo-diarization`** was inert on corpus-layout corpora until #33 (the `--reprocess-source`
  force predicate resolved metadata against the current run dir, so it never fired). If you relied
  on it before 2026-08-17, grep old run logs for `#925 forcing` — if absent, that migration
  no-opped.

## Known limits

- `metrics.json` is **run-level**. A one-episode run attributes exactly; a multi-episode run can
  only say "this run has damage". Per-episode attribution needs the `audio_preprocessing`
  stage-ledger row added in #22, which by construction exists only on runs made *after* that
  change — never on the damaged ones.
- The 60 % figure is from a **local pre-fix test corpus**, not production. Run step 2 against the
  production corpus before treating it as representative.
- `--reprocess-episode-ids` has unit-tested selection logic but has **not** been exercised end to
  end against a corpus.
- Whether the placeholder count in production is still ~112 is unverified; step 1 is what answers
  it.
