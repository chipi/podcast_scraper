# Multilingual ingest — arc notes (v1)

The one page that holds this arc together. The PRD says what the product needs, the three RFCs each
own a slice of the how, and this document is where the arc's **shape**, its **slice plan**, its
**verified code facts**, its **decisions** and its **running notes** live.

- **Arc**: multilingual ingest (source-language capture, English-normalized intelligence)
- **Scope**: **v1 only — a working pipeline and product, end to end.** Everything deferred lives in
  [MULTILINGUAL_ARC_V2](MULTILINGUAL_ARC_V2.md), with its slices and the reason it was deferred.
  Nothing was dropped.
- **Opened**: 2026-09-28
- **Status**: design — nothing implemented
- **Branch**: `feat/multilingual-ingest`
- **Documents**: [PRD-047](../prd/PRD-047-multilingual-ingest.md) · [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) · [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) · [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md)
- **Review state**: six adversarial reviews have run over these documents. §5.4 records every claim
  they found false — including errors introduced by an earlier *correction* pass — because the way
  those claims were wrong repeats and is worth recognising.

---

## 1. The arc in one page

The corpus is English-only by configuration, not by architecture. The bet is that we can ingest a
non-English show, keep **what was actually said** as the canonical record, and derive an English layer
that every existing intelligence stage reads without a per-language fork.

```text
audio (any enabled language)
  │
  ├─ transcribe + diarize IN SOURCE LANGUAGE ──────► ep1.txt            ← canonical, full timeline
  │   └─ speaker naming (on the source)              ep1.segments.json
  │                                                  ep1.turns.json     ← RFC-123
  │
  ├─ TRANSLATE every turn-bounded unit ────────────► ep1.translation.json  ← the traceability map
  │   (before summary — D-19)                        ep1.en.txt            ← derived, full timeline
  │                                                  ep1.en.segments.json  ← doubles as subtitles
  │
  ├─ ad-detect + excise ON THE ENGLISH ────────────► ep1.en.adfree.txt   ← THE ANALYSIS TRANSCRIPT
  │
  ├─ summary → GI → KG, all reading that file ─────► single-path, English
  │
  └─ full standing on every surface, same as English; provenance recorded, invisible to the user
```

Three properties make it work:

1. **The source is canonical, the English is derived.** A translation is an interpretation; the record
   is what was said. Every English span maps back to source text, source speaker and source audio time.
2. **Analysis reads one transcript.** Every transcript reader resolves through one function, so no
   stage decides for itself what "the transcript" means. That is **not** true today (§5.4 C-1) and
   making it true is part of the work, not a property the design can assume.
3. **v1 believes the translation.** A translated episode has the same standing as an English one on
   every surface — no gating, and no user-visible marker (D-36, D-37). Provenance is recorded on every
   claim so v2 can verify, label and, if it turns out to be needed, gate. The bet is explicit: pick a
   good model, and treat quality as v2's subject rather than something v1 hedges around.

## 2. Document map

| Document | Owns | Does **not** own |
| --- | --- | --- |
| [PRD-047](../prd/PRD-047-multilingual-ingest.md) | Why, for whom, what "done" means, phase gates, language policy, operator + listener surfaces | Any implementation shape |
| [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) | `turns.json` — turns and sentences as addressable units. v1 needs the artifact; its consumers are v2 | Anything language-specific |
| [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) | Language resolution, source capture, the translation stage, the artifact set, the stage order, retrieval, model selection | Trust, gating |
| [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) | Translation provenance, recorded and invisible to the user (v1); verification, labelling, gating and QE (v2) | Producing the translation |
| This document | v1 arc shape, slice plan, verified code facts, decisions, running notes | Requirements or design detail |
| [MULTILINGUAL_ARC_V2](MULTILINGUAL_ARC_V2.md) | Everything deferred out of v1, with slices and reasoning | Anything v1 ships |

## 3. Phase ladder

Phase numbering means one thing across all four documents. The demand/model validation step is
**Gate V**, not a phase, because it produces evidence rather than software.

| Phase | What | Gate to start | Visible outcome |
| --- | --- | --- | --- |
| **−1 — Instruments** | One transcript resolver with two intents (no behaviour change), and the artifact allow-list test | none; both are pure-English fixes with standalone value | A golden of which transcript each reader resolves, and a test that says what changed. Everything after is measured by these |
| **0 — English as a declared language** | Language becomes a real, parsed, resolved, validated property of the corpus we already have | none; it is a correctness fix | An audit over real data showing the corpus's actual language distribution; language on the API; no code path that substitutes a language it was not given |
| **1 — Turns artifact (RFC-123)** | `turns.json` built and written for the source variant. Nothing reads it yet; the backfill and all three consumers are v2 | none | The unit translation needs |
| **Gate V — Validate** | Demand check; model selection and its sanity check | Phase 0, because the check cannot measure non-English transcription until it exists | A go/no-go with evidence, and one pinned translation model |
| **2 — Translation (RFC-124)** | Reader routing, the stage-order change, translation, the English render, ad-free-on-English, same-language retrieval | Gate V passed | One non-English feed processed end to end, findable in its own language, behaving like any English episode |
| **3 — Surfaces** | The transcript/subtitle reading path and flag lifecycle. The badge and the language filter are v2 | Phase 2 | A listener can read the original or the English against the original audio |

**Why Phase 0 is a real phase.** Today a feed's declared language is never read, every episode is
stamped with the run configuration, and a non-English episode would be transcribed by a chain that can
fall back to `base`. Phase 0 makes language an *asserted, checked* fact — which is exactly the plumbing
translation needs. It is testable on the corpus that already exists, with no new models and no GPU, and
it is the recommended standalone ship.

## 3.1 Outcomes for Phase −1 and Phase 0, and how they are proven

Neither phase ships a feature. Both convert a question that today takes an **argument** into one
that takes a **command**. That is the outcome, and it is what the gate should be judged on — not
"the slices merged".

The per-slice acceptance criteria in §4 are *outputs*. This section is the *outcome* layer above
them, and the proof procedure for each. Outcome and proof are written together on purpose: an
outcome nobody can demonstrate is a hope.

---

### Phase −1 — "we can see what we broke"

**Outcome.** Any change to transcript resolution, or to an English artifact, is **named** — reader,
episode, field — by a command, before it reaches a review conversation.

**How we would know it failed.** The instruments never fire. An instrument that has only ever been
green is indistinguishable from no instrument, and both slices' stated criteria are satisfiable by
one that can never fail. So the phase's real gate is **fault injection**, and it must be committed
as a test rather than performed once by hand:

| Injected regression | The instrument must name |
| --- | --- |
| a reader switched from `ANALYSIS` to `TIMELINE` | that reader's row, on the episodes that have both variants |
| the ad-free fallback deleted | every row on the pre-#974 corpus |
| `load_transcript` resolving its sidecar independently of its body | the both-sidecars test, by name |
| the `gi/load.py` fix reverted | `A10`, on exactly the ad-excised episodes |
| an unexpected key added to episode metadata | that key, in the allow-list violation |
| `pipeline_composition_version` moved | the hash, and which stage set moved it |

**Evidence this is achievable, not aspirational — and that it pays.** Two instances so far.

1. The `gi/load.py` fix moved the golden on one field, on exactly the 40 ad-excised episodes and
   nothing else, established by a field-by-field diff before regenerating. The instrument working.
2. Writing the injection tests **found a blind spot in the golden itself**. The GI/KG row recorded
   only `(ref, is_adfree)`, and when nothing resolves, the loader reports the *canonical* relpath
   with `is_adfree=False` — byte-identical to a successful raw load. So deleting the ad-free
   fallback moved five other readers' rows and left that one green. The row now also records
   whether text and segments were actually loaded.

The second is the argument for the criterion: the positive golden was green, reviewed, and blind.
Only injection said so. That is the standard S0.10 has to meet too.

**Proof procedure.** For each row above: apply the regression, run the instrument, confirm the
named output, revert. Committed as negative tests (`monkeypatch` the regression in, assert the
probe output diverges from the committed golden) so the proof survives the session it was made in.

---

### Phase 0 — "English is a fact, not an assumption"

**Outcome.** Language is something the system **parsed, resolved and checked**, per episode, and an
episode in a language we cannot handle is **refused loudly** rather than mistranscribed quietly.

**The counterintuitive part: Phase 0 is measured on ENGLISH content.** It ships zero non-English
capability. Success is two things at once, and the first is the stricter:

1. **The English corpus is unchanged**, outside a declared and reviewed allow-list. If everything
   else lands and English artifacts moved, the phase failed.
2. **The five silent hazards (§5.2) are closed** — each converted from "fails quietly" to either
   "refuses loudly" or "cannot happen".

| Before | After | Proven by |
| --- | --- | --- |
| "the corpus is English", asserted | a committed audit naming every item's language **and its resolution source** | S0.4's report, in the tree |
| ~10 sites can substitute `"en"` | one — the resolver's lowest-precedence default | S0.6's lint, running in CI, with a reviewed whitelist |
| a `de`-tagged feed is transcribed as English | skipped, with a reason in logs, metrics and the manifest | S0.8, observed on a real episode |
| a tier that cannot do the language sits in the chain | out of the DGX chains; the local provider refuses non-`en` | S0.7, plus the dev-twin test |
| English artifacts drift unnoticed | an allow-list diff over a restored **prod** corpus snapshot | S0.10, run before and after the phase |

**How we would know it failed.** The audit comes back 100% `en` with every row resolving to
`profile_default`. That is the signature of a check that measured the **configuration** instead of
the corpus — the same failure shape as `capability_audit`'s "0/36 openings, defect rate 0.0%"
recorded in §5.4. Two consequences, both already in the plan and both load-bearing for this reason:
S0.4 must run **after** S0.1b, and the audit should **assert** that at least one row resolved from
`rss`, failing if none did.

**Fixtures are not evidence here.** The allow-list comparison and the audit have to run against a
restored prod corpus snapshot, not only the fixture corpora — a fixture measurement has already
caused a correct constant to be reverted once. The end-to-end skip observation (S0.8) is sized to
**one or two episodes** on the DGX; anything larger needs explicit approval first.

---

### What neither phase claims

Stated because silence on gaps reads as a claim:

- **No non-English audio is transcribed.** Phase 0 makes an unsupported language *skip*; it does
  not make a supported one work.
- **No translation quality is established.** That is Gate V, and it is evidence rather than software.
- **Nothing changes for a user.** No badge, no filter, no language control — all v2 or Phase 3.
- **The real value is unmeasurable directly.** These phases reduce the cost of being wrong in
  Phase 2. The honest proxy is how many unknowns became measured facts: five hazards, one corpus
  language distribution, and one inventory of fifteen transcript readers.

## 4. Slice plan

Each slice is sized to be **one GitHub issue**: one goal, its own tests, its own acceptance criteria,
and shippable without leaving the tree half-built. The epic is [#2169](https://github.com/chipi/podcast_scraper/issues/2169); Phase −1 and
Phase 0 are opened and linked below. Later phases are opened when Phase 0 ships.

*Depends on* is a hard ordering constraint. *Size*: **S** = one sitting; **M** = a day-ish, multiple
files, real test surface; **L** = multi-day, new subsystem or a migration. *Ship alone?* asks whether
merging only this leaves `main` correct and coherent.

---

### Phase −1 — the two slices that go before everything

Both are pure-English work with standalone value, and both are instruments the rest of the plan is
measured by. Neither depends on anything in this arc.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S2.1a** [#2170](https://github.com/chipi/podcast_scraper/issues/2170) | One transcript resolver with two intents, no behaviour change | `load_processing_transcript` has two callers while nine other readers resolve independently (§5.4 C-1). Route them all through one function that takes a **`purpose`** — `analysis` (the ad-free text GI's offsets live in) or `timeline` (the full text the player syncs to audio). Collapsing both into one precedence would desync the player, which is exactly the drift `segments_view.py` exists to prevent (D-35). Ships with **zero behaviour change** and a committed golden recording which file each reader resolves on the fixture corpus — that golden is what makes every later slice's blast radius visible. Also fixes `gi/load.py`'s raw-transcript read, a real (CLI-only) coordinate-space bug today. | — | L | Yes |
| **S0.10** [#2171](https://github.com/chipi/podcast_scraper/issues/2171) | The English-artifact allow-list test | The instrument Phase 0's acceptance is judged by, and it does not exist: serialize the metadata and manifest shapes before and after, assert `added_keys ⊆ ALLOWLIST`. Goes early because everything else should be measured against it. Must also assert that **`pipeline_composition_version` does not move** — it hashes the stages *present* (`processing_manifest.py:198`), so if a skipped `translation` stage enters `stage_names`, every English episode's composition hash changes, and that hash is what "reprocess below X" and the prod-state pin key on. Decide that a skipped stage is not present. | — | M | Yes |

---

### Phase 0 — English as a declared language

**Ships as one release.** No new models, no GPU, no user-visible chrome — the badge moved to v2 (D-11).
At the end, "this episode is in English" is something the system parses and checks rather than assumes,
and the same machinery carries any other language.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S0.1a** [#2172](https://github.com/chipi/podcast_scraper/issues/2172) | Parse and persist the feed's declared language | Extract the channel `<language>` (a `channel.find("language")` in the pattern `rss/parser.py` already uses for title, author and description) and carry it on `RssFeed` and `FeedMetadata`; persist `feed.language_raw`, `feed.language`, `feed.language_source`, plus an **episode-level** `language` and `language_source`. Add the field to `RssFeed` as a defaulted field, never positional — it is constructed at ~53 test sites. Note there are two `FeedMetadata` types (a persisted pydantic one and a positional NamedTuple); the slice must name which. | — | M | Yes |
| **S0.1b** [#2173](https://github.com/chipi/podcast_scraper/issues/2173) | Backfill language onto the existing corpus | A one-off script, **per show, not per episode**: walk the shows in the corpus, fetch each feed, read its `<language>`, normalize it, and write it onto the show plus every episode under that show. The language is a property of the feed, so one fetch backfills all of its episodes. Runs as a migration so it is versioned, re-runnable and recorded like any other corpus fix, rather than a script somebody remembers running. Needs: a feed whose URL is missing from the metadata block is reported and skipped, not guessed; the CI migration fixture has a `feed` block with no `url`, so the migration must tolerate that rather than fail; and a `--dry-run` that prints the distribution before writing. Reported per show so the output doubles as S0.4's first data. | S0.1a | M | Yes |
| **S0.2** [#2174](https://github.com/chipi/podcast_scraper/issues/2174) | Language normalization, resolution, and the per-feed override | `normalize_language_tag` — **deliberately trivial**: lowercase, take the primary subtag, keep `language_raw` as given. No `und`/`zxx`/`mul` policy, no three-letter mapping, no script parsing (D-21). Plus `resolve_episode_language(feed_entry, feed_doc, cfg)` and the `config/languages.yaml` registry (D-29) seeded with all three tiers, `en` the only enabled one. Must route the profile default through the same normalizer — `Config._normalize_language` only lowercases today, so `language: en-US` yields `en-us`, which fails `whisper_utils.py:50`'s `is_english` check: **a live bug this slice fixes**. Includes the **per-feed override** (a `language` field on `RssFeedEntry`, which is `extra="forbid"`, plus an `RSS_FEED_ENTRY_OVERRIDE_KEYS` entry) — an allowlist key and a model field are not their own slice, and the override is what corrects an odd tag at onboarding. Note `merge_feed_entry_into_config` makes the per-feed `Config` the run's `cfg`, so the override reaches every existing reader with no threading. | S0.1a | M | Yes |
| **S0.4** [#2175](https://github.com/chipi/podcast_scraper/issues/2175) | Corpus language audit over real data | Read-only CLI in the existing `check_corpus` pattern: walk every feed and episode, resolve a language, report the distribution and every item not resolving to `en`, each with its resolution source. Commit the report — **S0.6 depends on it being clean.** Runs after S0.1b, or it can only re-read the run config and reports a guaranteed 100% `en`, which proves nothing. | S0.1b, S0.2 | S | Yes |
| **S0.5** [#2176](https://github.com/chipi/podcast_scraper/issues/2176) | Expose language on the app API | Episode-level `language` (additive) on the episode list and detail responses; `AppPodcastItem.language` starts serving the normalized feed tag instead of the run config; **`CorpusFeedItem` gains the field too** — the operator viewer's shows library consumes that one and has no language data without it. Contract tests for present / absent / legacy values. | S0.1a, S0.2 | S | Yes |
| **S0.6** [#2177](https://github.com/chipi/podcast_scraper/issues/2177) | One language reader: thread it, delete the substitutions | The transcription call site passes the episode's resolved language; the DGX provider stops **reporting** `"en"` for an auto-detected transcript; `ml_provider`'s `self.cfg.language or "en"` goes; `sniff_gate.py`'s four sites and `metadata_generation.py:957/:3832` are covered. **Hard dependency on S0.4's committed clean report**: after this slice a feed whose RSS says `de` is actually transcribed as German, so it must not land while the audit is unknown. Acceptance is a lint rule with an explicit whitelist — "exactly one reader" is false as stated, because the cloud providers and `ner.py` legitimately read `cfg.language` — and the lint must be wired into CI, which its cited precedent is not. | S0.2, S0.4 | M | Yes |
| **S0.7** [#2178](https://github.com/chipi/podcast_scraper/issues/2178) | Drop the local Whisper tier from the DGX profiles' chains | The DGX Whisper is one multilingual model; you pass the language code. Remove `whisper` from `prod_dgx_full.yaml:111` → `[tailnet_dgx_whisper]`, and from `dev_dgx_full.yaml:126` (or the dev-twin-tracks-prod test fails) and `eval_default.yaml:172`. The provider **stays** — it is the primary transcriber in eight local/dev/airgapped profiles — and gains a one-line guard refusing non-`en`. **This replaces the per-episode model-selection work entirely**: prod then has no language-driven model selection at all (D-22). | S0.6 | S | Yes |
| **S0.8** [#2179](https://github.com/chipi/podcast_scraper/issues/2179) | Unsupported-language skip, and its visibility | `skipped_unsupported_language` for a language that is not `enabled`, expressed with the **existing** status vocabulary — `status="skipped"` plus a reason, no `Literal` change (D-23) — together with the manifest `language` block, log and metric surfacing, and a runbook entry. The skip and the ability to see it are one change: shipping the skip blind would add a silent failure to the phase whose whole purpose is removing them. Must land **after** the override (folded into S0.2) and the audit, or a mis-tagged English feed stops ingesting with no remedy. | S0.2, S0.4 | M | Yes |

**Phase 0 acceptance** (a release checklist, not an issue): the audit reports the corpus's real language
distribution with every item's resolution source named; language is on the API; no stage can receive a
null language and substitute one; the local Whisper tier is out of the DGX profiles' fallback chains and
refuses non-English; the new failure modes are visible; and English artifacts are unchanged outside the
declared allow-list.

---

### Phase 1 — Turns artifact (RFC-123)

v1 needs the artifact to exist, because translation units are sentence groups inside a turn. Its three
consumers are **v2** ([V2-D](MULTILINGUAL_ARC_V2.md#6-turns-consumers)).

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S1.1** | ~~`build_turns`: turns and sentences from a segments sidecar~~ | **DONE 2026-09-29.** `providers/ml/diarization/turns.py`: turns are the screenplay lines by construction, backchannels are flagged not merged, sentences carry `segment_exact`/`segment_interpolated`, and `TurnInvariantError` raises rather than returning a plausible structure — the failure guarded against is a quote attributed to the WRONG speaker with every stage reporting success. 23 tests drive the real formatter, not hand-written offsets. Nothing reads it. | — | M | closed |
| **S1.2** | ~~Write `turns.json` in the pipeline~~ | **DONE 2026-09-29.** `workflow/turns_artifact.py` + `episode_processor._produce_transcript_sidecars` (which replaces `_maybe_produce_adfree` at all five write sites). One artifact per VARIANT, each anchored to its own text — the ad-free variant drops whole turns and renumbers from `t0000`, so a cross-variant join by `turn_id` would point at the wrong turn. Turns are written even when `save_adfree_transcript` is off: the two are independently gated on purpose. A non-screenplay transcript gets no artifact and says why (`not_a_diarized_screenplay`), which is RFC-123's `turns: unavailable`. The manifest gains `stages.turns` WITHOUT moving `pipeline_composition_version` (asserted). An invariant failure is counted and logged, not fatal — a stated v1 deviation that must harden at S1.4. Backfill stays in v2. | S1.1 | M | closed |

---

### Gate V — Validate (evidence, not software)

| # | Issue title | Goal | Depends on | Size |
| --- | --- | --- | --- | --- |
| **V.1** | ~~Verify translation model availability and licences~~ | **DONE 2026-09-28 — §6.1.** | — | closed |
| **V.5** | ~~Choose the pilot language by measurement~~ | **CLOSED by D-29.** The language roadmap is decided and the pilot is Spanish or Italian. The one residual check is confirming the chosen model covers tier 1 — trivially true for everything except **Catalan**, which needs the gated TranslateGemma card opened or MiLMMT-46 chosen. Folded into V.3. | — | closed |
| **V.2** | Demand check with the beta cohort | Needs the instrument written first: question wording, cohort size, how "≥30%" is computed. | — | S |
| **V.3** | ~~Model selection and the quality gate~~ | **DONE 2026-09-29 — [ADR-156](../adr/ADR-156-translation-model-and-serving.md).** `google/translategemma-12b-it` is deployed co-resident on `:8005` and translates the V.6a fixture correctly: speaker labels survive verbatim, and the English render carries **two** `_AD_PATTERNS` hits against **zero** on the Spanish source. Two plan changes fell out of it — S2.3 must use `/v1/completions` (the chat route is unusable) and S2.4 must decide about titles (the model renamed the show). **The quality gate itself is NOT done**: throughput measured 4.3 tok/s under a ~96% loaded box, which is contention, not capacity, and a bake-off needs a quiet DGX. | S0.10, V.6 | closed |
| **V.4** | ~~Gate V decision record~~ | **DONE — [ADR-156](../adr/ADR-156-translation-model-and-serving.md)**: the model and pinned revision, the co-resident serving decision and why it departs from the single-owner rule, the `/v1/completions` contract, the memory sizing (and the boot that failed to find it), the evidence, and the Gemma licence position. **The language decision is NOT in it** — that is V.2's, and it is yours. | V.2, V.3 | closed |
| **V.6a** | ~~Non-English fixtures: observe the hazards~~ | **DONE 2026-09-29 (#2186), transcripts only.** All three transcript-observable hazards measured against an English control — and **two of the three predictions were WRONG**, both in the same direction (§5.2). Ad excision confirmed (0 patterns vs 6). The sniff gate *over*-counts (98 vs 65). Naming finds **both** real names but ships **four phantom people** past `_looks_like_person`. English NLP on non-English text is confidently wrong, not blind — which is why S2.14 now exists. Audio (V.6b, #2187) is deferred; the two hazards it would show are closed by construction in S0.6/S0.7. | — | closed |

---

### Phase 2 — Translation (RFC-124)

Behind `multilingual_ingest`. **The flag gates the pipeline; deciding when a translated episode becomes
visible is deciding when to add the feed to the production feed list** (D-16 withdrawn). So the
labelling and gating slices must be in before that feed is added.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S2.1b** | ~~Add the English branch to the resolver~~ | **DONE 2026-09-29.** ANALYSIS: `.en.adfree.txt` → `.adfree.txt` → `.txt`; TIMELINE: `.en.txt` → `.txt` → `.adfree.txt`. A pure prepend — the 80-episode golden regenerated with 80 pure INSERTIONS and zero modified lines, so no resolved path moved — which is why the resolver needs no flag check. ANALYSIS deliberately does NOT fall back `.en.adfree` → `.en.txt`: that would put ad text into the space GI's offsets index. Canonicalization now strips the suffix STACK (`ep1.en.adfree.txt` → `ep1.txt`); stripping one would leave `ep1.en` and resolve nothing. The player's default language, which this surfaced, was decided on the spot rather than deferred: **D-38, English**. | S2.1a | S | closed |
| **S2.2** | ~~Give translation a stage slot, in one seam~~ | **DONE 2026-09-29.** `CANONICAL_STAGE_ORDER` gains `translation` between naming and summary; `workflow/translation_stage.py` decides and records at ONE seam inside `generate_episode_metadata`, which every transcript-producing path converges on. The flag `multilingual_ingest` is added (default off). Outcome vocabulary: `skipped` / `pending` / `translated` / `failed`; a non-English episode with the flag on records **`pending`** — an unpaid debt, not a failure and not a success. **Every episode in every language records the block** (D-39), an English one with `ran=False`, so the composition hash moves ONCE for the whole corpus and is identical across languages. I briefly shipped the opposite — withholding the block for English episodes to hold the hash still — which made the hash a function of the episode's LANGUAGE rather than of the code; reverted the same day, see §9. Translation's wall time is CREDITED back to the metadata deadline via a new `deadline_credit`, rather than guessing an allowance ahead of S2.10. | S2.1 | M | closed |
| **S2.3** | Translation units and the vLLM client | Turn-bounded unit packing from the source variant; the `dgx_vllm_translate` client with per-episode batching, bounded concurrency and per-unit retry. | S1.2, V.4 | M | Yes — flag-off is a no-op |
| **S2.4** | `translation.json` and the English render | The unit map; the English screenplay and `.en.segments.json` rendered through the existing formatter, one pseudo-segment per unit **carrying `unit_id`**; `translation_pending` and failed-unit semantics; a stub-translator integration test. **Plus the TITLE decision** — the deployed model translated `Sesiones de Sendero` → `Trail Sessions` (§6.1), so whether the show and episode titles go through the translator has to be chosen deliberately: drifting into it renames every show. Note §5.4 C-6 wants the title translated so NER candidate discovery works, which is an argument for translating the EPISODE title and not the SHOW name. | S2.3 | M | Yes |
| **S2.5** | Ad-free base on English, and span→unit resolution | Build `.en.adfree.*` with the existing machinery. `resolve_units_for_span` resolves through `.en.adfree.segments.json` and `unit_id` — **not** the ad-map, which cannot invert the ad-free transform (§5.4 C-5, measured). Uses **overlap**, not containment, so a span touching a label prefix or inter-turn whitespace still resolves. Writes the English artifact set atomically and refuses on a provenance mismatch — including `excerpt != text[char_start:char_end]`, which catches a re-translation that a file hash alone would not. | S2.4 | L | Yes |
| **S2.6** | Speaker labels bypass the translator | Naming stays **before** translation, on the source (§5.4 C-6), and the label is carried onto the English line **verbatim** — never sent through the translation model, which would rename the same person inconsistently across units. **No transliteration, no alias minting** (D-24): every tier-1 language is Latin script and names are usually the identical string across them, so there is nothing to convert. Folded into S2.4's render rather than being its own slice. | S2.4 | S | Yes |
| **S2.7** | Language-aware reprocess and invalidation | `_produce_transcript_sidecars` (renamed from `_maybe_produce_adfree` in S1.2) has **five** call sites including the transcript-cache hit; on a non-English episode each would write the identity ad-free artifact this design says never exists, and strand `.en.*`. Make it language-aware — **and note it now also writes `turns.json`**, so the same five sites decide the source variant's turns; the English variant's turns are S2.5's, built with the rest of the `.en.*` set. Give every path that changes the source an explicit invalidation of `.en.*`, `translation.json` and the cached prompt prefix; add the per-episode reprocess command. Note `rederive_only` must **not** re-translate — that is the cheap repair path. | S2.5 | M | Yes |
| **S2.8** | Segments API `?lang=` and translation status | Additive `language`, `machine_translated` and `translation_model` on `SegmentsResponse`, and `translation_status` on episode detail. **The DEFAULT is settled (D-38): English.** `?lang=` selects the source language as an explicit alternative; it does not decide the default, which S2.1b already implements in the `timeline` precedence. These are what the transcript control (S3.1) reads. **The user-visible "Translated from X" label is v2** (D-36) — so this slice is API surface only, not chrome. | S2.4 | S | Yes |
| **S2.9** | Same-language retrieval: a keyword-only table for non-English | Index both layers so a query in either language reaches the same episode. **Non-English chunks go in their own table with no vector column** (D-14, option B), consulted only for the keyword leg — a chunk with no vector cannot appear in a semantic result, which a row tag plus a filter could not guarantee. Leaves the existing `segments` table untouched, which should avoid the schema bump, the stale index and the full rebuild — **confirm that in the slice**, along with the read path tolerating the table's absence on older indexes. Also: chunk ids carry language, insight→segment linking filters on language, and a non-English query drops the dense leg via script detection or it returns English noise. Measure keyword recall through the English tokenizer before calling this done. **Script detection does not discriminate for tier 1** — *inflacion* and *inflation* are the same script — so the dense-leg switch needs a stop-word heuristic or an accepted dilution, measured either way. | S2.5 | L | Yes |
| **S2.10** | Cost and capacity measurement | Translation GPU time and storage delta per episode, and the bake-off's own cost. RFC-124 OQ3's wall-time cap cannot be set without it. Note for model choice: a 27B translator and the served 30B model do not co-reside in the DGX's memory while a 12B does. | S2.3 | S | Yes |
| **S2.11** | Translation provenance on every claim | The `translation` block (`translated`, `source_language`, `unit_ids`, `en_sha256`) written into node `properties` via `resolve_units_for_span`, by **every** writer of `gi.json` — the artifact builder, `add_spoken_by_edges(replace=True)` and `gi/repair.py`. | S2.5 | M | Yes |
| **S2.14** | English-only NLP cannot run on non-English text | A guard at the naming and sniff-gate entry points: refuse (or skip with a reason) when the text's resolved language is not `en`. **Defence in depth, and the failure it catches is silent.** If translation ran correctly nothing non-English ever reaches these stages — but a skipped translation, a stage run out of order, or a reprocess with the wrong flag would feed Spanish to English NER, and the measured result (§5.2) is not an absence but *four phantom people* surviving `_looks_like_person`, each minted as a person node with a `SPOKEN_BY` edge and position claims. A missing name is visible; a phantom person is not. Same shape as S0.7's English-only transcription guard, and for the same reason. | S2.4 | S | Yes |
| **S2.13** | Phase 2 gate | One non-English feed from audio to insights, findable in its own language, no English regression outside the allow-list, and the translated episode behaving on every surface exactly as a native-English one does. | S2.1b–S2.11 | M | This is the ship |

---

### Phase 3 — Surfaces

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S3.1** | Transcript language control | **The transcript defaults to English** — the rest of the app (summary, insights, everything) is English, so a source-language transcript by default would be the inconsistent choice. A small control in the transcripts panel switches to the original (D-25). Two things to get right: it must serve the **full-timeline** `ep1.en.txt`, not the ad-free analysis base, or it desyncs from the audio wherever an ad was cut; and it needs a small language affordance in the panel — the badge *component*, whose use as metadata decoration stays v2 (D-26). Backend is already in place — `?lang=` lands with S2.8. | S2.8 | M | Yes |
| **S3.2** | Feature-flag lifecycle and rollback | What `multilingual_ingest` gates at each phase, its removal criterion, the per-phase rollback procedure, and what happens to a language that is **disabled** after episodes exist in it. | S2.13 | S | Yes |

---

### Critical path

```text
{S2.1a, S0.10} ═══ PHASE −1: the instruments ═══╗
                                                ▼
S0.1a → S0.1b → S0.4 ──┐
S0.1a → S0.2 ──────────┼→ S0.6 → S0.7 → S0.8 ═══ PHASE 0 SHIPS ═══╗
S0.1a → S0.5 ──────────┘                                          ║
                                                                  ▼
S1.1 → S1.2 ═══ PHASE 1 ═══╗       V.6 → V.3 ──┬── V.4 ═══ GATE V ═══╗
                           ║  V.2 ─────────────┘                     ║
                           ▼                                         ▼
  S2.1b → S2.2 → S2.3 → S2.4 → S2.5 → {S2.6, S2.7, S2.8, S2.9, S2.10,
                                       S2.11} → S2.13 ═══ PHASE 2 ═══╗
                                                                           ▼
                                                                  {S3.1, S3.2}
```

Phase 0 is close to serial through `S0.1a → S0.2 → S0.6`, and V.6 — the non-English fixture, run end to
end locally — is the cheapest thing in the whole plan that can invalidate a design assumption, so it
should happen on day one rather than when Gate V formally starts. V.2's demand instrument is the only
human-shaped item and has the longest lead time; nothing blocks on it, so start it early.

## 5. Code facts this arc rests on

### 5.1 Findings that survived review

| Finding | Evidence | Consequence |
| --- | --- | --- |
| **There is no stance-extraction stage, and deliberately none.** Stances are GI insights; Positions are read-time queries. | `server/cil_queries.py:636` `position_arc`; `enrichment/profile_sets.py:144-146`; ADR-108's 2026-07-08 update retiring `stance_timeline`. A repo-wide word search for `stance` finds only comments. | A future gate would be a **read-path filter**, which is why v2 can add one retroactively over the existing corpus without re-extraction. No stance-extraction dependency exists. |
| **`vector_embedding_model` is genuinely wired to the search index.** | `search/indexer.py:577` → `build_two_tier_index`; the query side reads the model recorded in the index (`hybrid_search.py:238-244`). | A future encoder swap is coherent. v1 changes no embedding model (D-14). |
| **`GiArtifact` forbids extra top-level keys**; `EvidenceSpan` / `SupportingQuote` / `SegmentsResponse` are plain models. | `gi/contracts.py:130`, `:14`, `:25`; `server/schemas.py:24`. | Additive data lives in node `properties`; API fields are safe to add. |
| **The ad-free identity hazard is real.** | `gi/ad_regions.py:409-418`; `adfree_transcript.py:104-106`, `:129-137`; `load_processing_transcript:237-250` sets `is_adfree=True` on file existence alone. | Translation precedes ad detection (D-3). Caveat: with no segments `build_adfree_artifacts` returns `None`, so the identity artifact only appears for episodes with a segments sidecar. |
| **`adfree_transcript_relpath` composes `ep1.en.txt` → `ep1.en.adfree.txt`** unchanged. | `adfree_transcript.py:52-55`. | No new path helper needed. |
| **`formatting.py:81`'s passthrough tuple can carry `unit_id`** through to both sidecars. | `formatting.py:81`; `episode_processor.py:825`; `adfree_transcript.py:180` dump segment dicts unchanged. | The span→unit chain (C-5's fix) is implementable. |
| **The per-feed override reaches the whole run.** | `feeds_spec.py:250-291` `merge_feed_entry_into_config` does `cfg.model_copy(update=...)`; `service.py:143`. | An override needs no threading — but see C-4 for the multi-feed singleton hazard. |

### 5.2 The silent hazards for non-English audio

Five. Each fails quietly rather than erroring. Hazards 4 and 5 were measured on the Spanish
fixture on 2026-09-29 (V.6a, #2186) — **4 confirmed, 5 falsified and rewritten below.** 1, 2 and 3
are transcription-tier and need audio (V.6b, #2187); they are closed by construction in S0.6/S0.7
rather than by observation.

**The speaker-naming claim was WRONG, and the truth is worse. MEASURED 2026-09-29 (V.6a).**

§5.4 says non-English naming *fails*: voices stay `SPEAKER_NN`, no `SPOKEN_BY` edge, zero
position-bearing insights. Measured through `detect_hosts_from_transcript_intro` with a real
spaCy pipeline, against corpus-shaped transcripts in both languages:

| | candidates | real people found | precision | noise surviving the filters |
| --- | --- | --- | --- | --- |
| English (control) | 3 | **2/2** | 67% | **none** |
| Spanish | 11 | **2/2** | **18%** | **4** |

**Recall is perfect in both.** `Maya` and `Liam Verbeek` are proper nouns, and a proper noun does
not change across Latin-script languages — the same reason D-24 skips transliteration. The names
are found.

**Precision collapses, and the downstream filters make it worse rather than better.**
`gi/speakers._looks_like_person` requires ≥2 tokens, which is exactly what Spanish function-word
phrases satisfy. On English it cleans perfectly (`Strava` dropped). On Spanish **four noise phrases
survive it**:

```text
'banco en'   'de la'   'la construcción de senderos'   'más impacto en'
```

So the failure mode is not an absent name — it is a **phantom person**.
`la construcción de senderos` would be minted as a person node, carry a `SPOKEN_BY` edge, and
attach position claims. That is strictly harder to detect than silence, because every stage
reports success.

This is the third §5.2 prediction that measurement corrected rather than confirmed, and all three
failed the same way: **English NLP on non-English text is confidently wrong, not blind.** Hazard 5
(the sniff gate) over-counts 98 vs 65 rather than reading ~0; naming over-produces 11 vs 3. A
threshold or filter tuned on English behaviour is being fed inflated noise in both cases.

It also makes the naming-symmetry decision load-bearing rather than an optimisation: naming must
run on the **English** text after translation, because that is what stops a
`la construcción de senderos` person node from existing at all.

1. **A tier that cannot transcribe the language is in the chain.** The local `whisper` tier's chain runs
   down to `tiny` for non-English and prod's default is `base.en`. Removed from the DGX profiles
   entirely, with the provider guarded against non-`en`. → **S0.7**
2. **No per-episode language path exists.** Providers read run-global config when the argument is
   absent; the call site passes exactly that global; `sniff_gate.py` threads it four more times.
   → **S0.6**
3. **The DGX provider misreports the language.** `whisper_provider.py:197` is the **returned result
   dict**; the request at `:429-430` *omits* `language` when it is `None`, so the server auto-detects
   and returns the detected value — which the client discards and overwrites with `"en"`. A provenance
   lie, not a forced English transcription. → **S0.6**
4. **Ad excision silently no-ops on non-English text. MEASURED 2026-09-29 — confirmed.** On the
   Spanish fixture (V.6a, #2186), paired against English `p01_e01` through the same functions:

   | | patterns firing | cut ranges | chars cut | identity copy |
   | --- | --- | --- | --- | --- |
   | English (control) | 6 | 3 | 999 (12.5%) | No |
   | **Spanish** | **0** | 0 | 0 | **Yes** |

   The mechanism is specific: `PREROLL_THRESHOLD = 3`, and every domain pattern needs an English verb
   before the URL (`visit` / `go to` / `check out` / `learn more at`) or ` slash ` after it. There is
   **no bare-domain pattern**, so `strava.com/podcast`, `stripe.com/podcast` and `linear.com` all
   appear verbatim in the Spanish text and nothing fires — `Visita stripe.com` does not even match
   `\bvisit\s+`, because after `visit` comes `a`, not whitespace. Ad-free base is an identity copy
   marked `is_adfree: True` with all three sponsor reads intact. → **S2.5**
5. **The sniff gate's entity count becomes meaningless on non-English — NOT zero. CORRECTED
   2026-09-29; the original claim here was wrong.** This said "on non-English that count is ~0".
   Measured, Spanish scores **higher** than English:

   | | entities counted | distinct |
   | --- | --- | --- |
   | English (control) | 65 | 10 |
   | **Spanish** | **98** | **54** |

   So the gate would keep the cheap transcript *more* readily, not less. The real defect is worse
   than the predicted one: English NER on Spanish hallucinates entities out of ordinary capitalised
   words and sentence fragments — `'Claro'`, `'Cuestiona'`, `'El marco'`, `'Las'`, `'Qué'`,
   `'Nos vemos'`, `'aplícala durante'`, `'Cambié de opinión sobre los senderos hechos'` — against
   English's 10 real ones (`Maya`, `Liam Verbeek`, `Cascadia Alliance`, `Strava`, `Shimano`,
   `Linear`). It also counted the diarization label `SPEAKER_01` as an entity. The gate is not
   blind, it is **confidently wrong**, and a threshold tuned on real English counts is being
   compared against inflated noise. Still off in every profile, one config line from live. → **S0.6**

### 5.3 Surfaces the arc touches

| Surface | Component / file | Used for |
| --- | --- | --- |
| Consumer episode list + toolbar | `views/CatalogView.vue:43-58` — single-select filter via `ListToolbar`, plus a show selector and sort | v2 filter |
| Episode + show items | `EpisodeRow/Tile/Card.vue`, `ShowRow/ShowTile.vue`, `PodcastView.vue` | v2 badge |
| Operator shows library | `library/{ShowsBrowse,ShowsView,ShowDetailView}.vue`, `LibraryFilterBar.vue` | S0.5 field, v2 badge/filter |
| Claim serialization | `AppInsight`/`AppQuote`, `gi/contracts.py` `InsightSummary`/`SupportingQuote`, `hybrid_search._to_search_result`, `server/og/build.py`, snapshot exports | v2 labelling — every one drops node properties today, and snapshots are copies a read-time change cannot reach |
| Position surfaces | `cil_queries.py` (nine entry points), `search/relational_queries.py:170 positions_of`, `enrichment/enrichers/topic_consensus.py` | v2 only — nothing in v1 filters these |

### 5.4 Claims that were WRONG, and the pattern behind them

Two rounds of review found false claims; the second found errors introduced by the *first* round's
corrections. **The recurring mistake is asserting completeness or behaviour from a partial search —
verifying a reader and assuming its writer, or enumerating what turned up and calling the list
complete.** Recorded so the next pass recognises it rather than repeating it.

- **C-1 — "`load_processing_transcript` is the single resolver all NLP consumers use" is a docstring,
  not a fact.** Two callers; nine or more independent resolvers (S2.1). **D-4's "consumers change
  nothing" was false.** A follow-on correction was itself wrong: `gi/load.py` reading the raw `.txt` is
  reached only from `gi inspect` / `gi show-insight`, so it is a CLI bug, not a live pipeline path.
- **C-2 — the RSS `<language>` tag is never parsed.** `feed.language` is the run config written back
  out, so `_feed_language` reads config and `AppPodcastItem.language` serves `"en"` by construction.
  The badge would have shown config, the audit would have been a tautology, and `en-US → en` had no
  input. One nuance: "no episode-level language field at all" was overstated — `TranscriptInfo.language`
  exists, also from config.
- **C-3 — the DGX hazard was misdescribed** (§5.2 hazard 3).
- **C-4 — model selection happens once at provider init**, and in a multi-feed batch the ML singleton is
  held across feeds, so feed 1's model persists and feed 2's language is silently ignored.
- **C-5 — the ad-map cannot invert the ad-free transform.** Measured twice. The mechanism is *not* "one
  label per excised range": the error appears even when a range removes a whole line cleanly, because
  newly adjacent same-speaker turns coalesce and their labels vanish. Conclusion stands; the fix
  resolves through segments and `unit_id`, using **overlap** rather than containment.
- **C-6 — naming cannot move after translation.** Citation corrected: `ml_provider.py:1061` is inside
  `detect_speakers`, not `transcribe_with_segments`; the load-bearing evidence is that labels are baked
  into the `.txt` at write time (`episode_processor.py:2869-2872`).
- **C-7 — D-15's containment was right by accident.** `gi_embedding_model` is read by nothing in GI; GI,
  the bridge, CIL identity, KG topic clustering, `hybrid_search` and `query_router` all hardcode MiniLM,
  and `insight_clusters.json` would not follow a config change. Moot for v1, load-bearing whenever a
  swap happens — v2 doc §7.
- **C-8 — RFC-125's code mechanics were wrong in three places.** `surfaceable` is set by
  `_apply_voice_flags`, not `_apply_route_and_tag`; `position_arc`'s predicate is SPOKEN_BY-supported
  quote ∩ `ABOUT` ∩ `insight_type == "claim"` and never reads `surfaceable` or `speaker_id` — and that
  type filter is a *default* a caller can drop; there are **no KG evidence spans** at all.
- **C-9 — the Positions surface list was wrong twice**, at two entries and then at five. It is nine, one
  of which is write-time. The enumeration belongs to v2's gate rather than to v1.
- **C-10 — Appendix A was incomplete.** Every number matched the source, but French (8.3), Arabic,
  Azerbaijani and Maori were dropped and then "everything unlisted is above 40%" was asserted.
- **C-11 — two overstatements in opposite directions from one `?`.** "Greek is covered by every eligible
  MT model" and "Serbian by none".
- **C-12 — smaller ones.** The `ner.py` gate selects the *default* NER model and is not a gate on NER;
  the cloud providers never substitute `"en"`, so there was nothing to remove there; 15 bakeoff
  profiles, not 14; "the index build runs in Docker" was cited to this document, which never said it;
  "the player forbids hard-coded user-facing strings" is a rule I asserted and could not be found.

## 6. Model and ASR evidence (verified 2026-09-28)

### 6.1 Translation model shortlist

| Model | HF id | Licence | `el` | `sr` | Verdict |
| --- | --- | --- | --- | --- | --- |
| TranslateGemma 27B / 12B / 4B | `google/translategemma-{27b,12b,4b}-it` | `gemma` — commercial use, no territorial carve-out | ? | ? | **Eligible.** 55 languages claimed; the list is nowhere public and **the model card is gated**, so a login settles it, or pick MiLMMT-46 which lists Catalan |
| MiLMMT-46-12B v1.0 | `xiaomi-research/MiLMMT-46-12B-v1.0` | `gemma` | ✅ | ❌ | Eligible; Serbian absent from its 46. Sizes 1B/4B/12B — no 27B |
| LMT-60-8B | `NiuTrans/LMT-60-8B` | **apache-2.0** | ✅ | ❌ | Eligible; Serbian absent from its 60. Self-described Chinese-English-centric |
| Qwen3-30B-A3B | already served | apache-2.0 | ? | **?** | Baseline / verification model; tier-1 coverage not separately checked |
| ~~Hunyuan-MT / HY-MT~~ | Tencent | Territory **excludes the EU**, UK, South Korea | — | — | **Excluded**, confirmed |
| ~~NLLB-200~~ | `facebook/nllb-200-3.3B` | **cc-by-nc-4.0** | ✅ | ✅ | **Excluded — non-commercial.** The only shortlisted model covering Serbian |

TranslateGemma's report confirms it was optimised with "an ensemble of reward models, including
MetricX-QE and AutoMQM" — an argument for QE later, and a caution that QE from the same lineage would
partly mark its own homework.

#### DEPLOYED AND MEASURED 2026-09-29 — `google/translategemma-12b-it`

Running on the DGX at `:8005`, **co-resident with the 30B summary model on `:8003`** rather than
swapped against it, because an episode needs translation and summarization in the same pipeline
pass. `gpu-mode-swap.sh prod` brings both up; the tailnet ACL grants `:8005`. Revision pinned to
`d1b225e1caa1…` (ADR-155).

**The model id in this table was wrong in a way worth recording:** the real id is
`google/translategemma-12b-it` — *one word*, no hyphen after "translate". `translate-gemma-…`
404s. It is `gated: manual`, so the terms need accepting by the token's owner; before that, the
HF API returns 200 on metadata and **403 on the files**, which presents as a boot hang rather
than an auth error.

**Memory — the fraction is of TOTAL, and covers weights PLUS KV cache.** Measured:

| | GiB |
| --- | --- |
| total usable (GB10) | 121.7 |
| prod-vllm's ACTUAL footprint | 29.1 |
| other compute apps (whisper/diarize/speaches) | 5.1 |
| **TranslateGemma-12B weights** | **23.3** |

The first boot used `--gpu-memory-utilization=0.20` (= 24.3 GiB) and vLLM refused:
`No available memory for the cache blocks`. ~1 GiB left for cache is an impossible config, not a
tuning miss. `0.32` → 38.9 GiB budget → **72,086 tokens of KV cache, 8.80× concurrency** at
`max-model-len` 8192.

The reasoning error worth keeping: prod-vllm is *allowed* 0.75 but **holds 29 GiB**. The fraction
is a ceiling a stack may claim, not what it occupies — which is what makes co-residency possible,
and is only knowable by measuring.

**It translates well.** es→en on the V.6a fixture:

> **in** — `Maya: Bienvenidos de nuevo a Sesiones de Sendero. … Maya: Este episodio es patrocinado
> por Strava. Comienza en strava.com/podcast.`
>
> **out** — `Maya: Welcome back to Trail Sessions. … Maya: This episode is sponsored by Strava.
> Visit strava.com/podcast to get started.`

Three consequences for the plan, each now evidence rather than assumption:

1. **Speaker labels survive verbatim** (`Maya:`, `Liam Verbeek:`). S2.6's design — carry the label
   onto the English line, never through the translator — is compatible with how the model behaves.
2. **The English render is ad-detectable.** That output contains `sponsored by` **and**
   `visit strava.com` — two `_AD_PATTERNS` hits, against **zero** on the Spanish source (§5.2
   hazard 4). Ad-detection-after-translation is demonstrated end to end, not predicted.
3. **It translates the SHOW TITLE** (`Sesiones de Sendero` → `Trail Sessions`). S2.4 has to decide
   deliberately whether titles go through the translator; drifting into it would rename shows.

**API contract — the chat route is unusable.** `/v1/chat/completions` rejects even the exact
structured content the model's own `chat_template.jinja` documents
(`content=[{type, source_lang_code, target_lang_code, text}]`): vLLM transforms the content list
before the template sees it, and the template's `content | length != 1` guard then fires.
**S2.3 must use `/v1/completions`** with the prompt rendered by the caller:

```text
<start_of_turn>user
You are a professional {source_lang} ({src_code}) to {target_lang} ({tgt_code}) translator. Your
goal is to accurately convey the meaning and nuance of the original text.

{text}<end_of_turn>
<start_of_turn>model
```

The language-name map the template uses (`es` → `Spanish`, and ~hundreds more including regional
subtags) is inside `chat_template.jinja` in the model snapshot — the client needs the same mapping.

**Throughput: 4.3 tok/s** (74 completion tokens in 17.1 s) — measured while the box was at ~96%
GPU under a production load, so this is *contention*, not capacity. It is still the only real
number available, and at ~200 units per episode translation is a material wall-time cost. **S2.10
needs a quiet box** for a figure worth planning against.

**Licence, re-confirmed against the accepted terms.** §4.3: *"Google claims no rights in Outputs
you generate using Gemma."* §1.5: *"For clarity, Outputs are not deemed Model Derivatives."* So
§3.1's distribution obligations (the `Notice` file, passing the agreement on, propagating §3.2)
bind redistribution of **weights**, which we never do — not publication of translated text. One
clause to keep in view: §1.2 counts "making Gemma or its functionality available as a hosted
service via API" as Distribution, which would matter only if the translator itself were exposed
to users.

### 6.2 ASR evidence — Whisper FLEURS WER

**Source**: Whisper paper, Appendix D.2.4, **Table 13 "WER (%) on Fleurs"**, the **`large-v2`** row.
Every value below was verified against the table.

**Caveats:** these are not large-v3 or turbo numbers (`large-v3` appears nowhere in the paper) and our
DGX model is `large-v3-turbo`, so treat this as a **conservative prior**; FLEURS is read speech and real
podcasts are worse; and bigger is not monotonically better — Serbian regressed from `large` 29.2 to
`large-v2` 33.9. Clusters A and B are complete; C is complete to 40%.

**Under 5%** — Spanish 3.0 · Italian 4.0 · English 4.2 · Portuguese 4.3 · German 4.5

**5–10%** — Japanese 5.3 · Polish 5.4 · Russian 5.6 · Dutch 6.7 · Indonesian 7.1 · Catalan 7.3 ·
**French 8.3** · Turkish 8.4 · Swedish 8.5 · Ukrainian 8.6 · Malay 8.7 · Norwegian 9.5 · Finnish 9.7

**Over 10%** — Vietnamese 10.3 · Thai 11.5 · Slovak 11.7 · **Greek 12.5** · Czech 13.3 ·
**Croatian 13.4** · Danish 13.8 · Tagalog 13.8 · Korean 14.3 · Romanian 14.4 · Bulgarian 14.6 ·
Chinese 14.7 · Galician 15.4 · Bosnian 15.7 · Arabic 16.0 · Macedonian 16.5 · Hungarian 17.0 ·
Tamil 17.5 · Hindi 21.5 · Estonian 21.9 · Urdu 22.6 · Latvian 23.1 · Slovenian 23.1 ·
Azerbaijani 23.4 · Hebrew 27.1 · Lithuanian 28.1 · Persian 32.9 · Welsh 33.0 · **Serbian 33.9** ·
Afrikaans 36.7 · Kannada 37.0 · Kazakh 37.7 · Icelandic 38.2 · Marathi 38.3 · Maori 38.5 ·
Swahili 39.3 — then Armenian 44.6 and the remaining low-resource languages. Javanese is `nan`.

### 6.3 The language roadmap (D-29)

Three tiers, market-ordered. The registry ships with all of them listed and only `en` enabled.

**Tier 1 — the focus.** Dutch 6.7 · German 4.5 · Italian 4.0 · Spanish 3.0 · Catalan 7.3 · French 8.3 ·
Portuguese 4.3 · Swedish 8.5 · Norwegian 9.5

Every one under 10%, five under 5% — English itself is 4.2, so Spanish, Italian, Portuguese and German
transcribe *better* than English does on this benchmark. There is no marginal language in the set. And
**all of tier 1 is Latin script**, which is why no transliteration or aliasing is built (D-24), why the
person-name guard works unmodified, and why word-based unit packing is fine.

**Tier 2 — Eastern Europe.** Russian 5.6 · Serbian 33.9 · Bulgarian 14.6 · Romanian 14.4

**Tier 3 — Asia and the Middle East.** Korean 14.3 · Japanese 5.3 · Chinese 14.7 · Arabic 16.0 (MSA;
dialects worse)

**The tiers are not difficulty-ordered, and it is worth knowing where they diverge.** Russian (5.6) and
Japanese (5.3) are technically easier than Norwegian (9.5) and would be tier-1 grade if wanted sooner.
Serbian (33.9) is the hardest language in all three tiers by more than double the next one. Tiers 2 and 3
also introduce **non-Latin scripts**, which is the trigger for the deferred transliteration and aliasing
work (v2 doc §5) — and tier 3 additionally breaks two v1 assumptions: word-count unit packing is
meaningless for Japanese, Korean and Chinese, and the ≥2-token person-name guard fails on a single-token
CJK name. Both must be addressed before a tier-3 language is enabled.

**Two different language choices, and only one of them is technical.**

- **The exercise language is Spanish**, for building and debugging the pipeline **locally on fixtures**
  before any real feed is touched. Chosen for convenience, not merit: the existing fixture generator
  (`tests/fixtures/scripts/transcripts_to_mp3.py`) drives macOS `say`, which ships Spanish voices; the WER
  is the lowest in tier 1; and every candidate model covers it. This is a test harness, not a product
  decision.
- **The first production language is the operator's call, on content value.** A low error rate makes a
  language *easy to process*, not *worth having in the corpus* — the question is whether there are shows
  that genuinely add source divergence, and that is judged by listening, not by a benchmark. Tier 1 sets
  the technical floor; which member of it goes first does not follow from the numbers.

**Fixtures come first, and that is a sequencing commitment, not a nicety.** The whole pipeline is worked
out locally against generated non-English fixtures — transcription, naming, turns, translation, ad removal
on English, summary, GI — and only then pointed at a real feed. It means every silent hazard in §5.2 is
observed on content we control before a single production episode is at stake.

**Model coverage across tier 1 is uniform except Catalan** — MiLMMT-46-12B lists it, LMT-60-8B does not,
TranslateGemma's list is behind a gated card. If Catalan is genuinely in the initial set, that narrows
the choice (D-27).

## 7. Decisions taken

| # | Decision | Rationale | Date |
| --- | --- | --- | --- |
| D-1 | Translate once, to English; analyze English | Every downstream layer stays single-path | 2026-09-28 |
| D-2 | Source is canonical, English is derived | The record is what was said | 2026-09-28 |
| D-3 | **Translation precedes ad detection**; the ad-free base is built on English | `_AD_PATTERNS` is English, so the alternative is an identity ad-free base feeding sponsor reads to GI. Also collapses two translation passes into one | 2026-09-28 |
| D-4 | **Revised.** Analysis reads English through **one** resolver, and routing every reader to it is *part of the work* | The original claimed the resolver already existed as such; that was its docstring (C-1). Slices S2.1, S2.2 | revised 2026-09-28 |
| D-5 | **Withdrawn.** There is no Positions gate in v1 | Superseded by D-37. A gate whose only possible v1 behaviour is "hide every translated claim forever" is not caution, it is spending GPU to produce data nobody can see. The gate belongs with the verifier that can release it, so both are v2. The nine-surface enumeration and the two-predicate design are preserved in the v2 notes because they will be needed there | withdrawn 2026-09-28 |
| D-6 | **QE is out of v1** | Calibration needs a translated corpus that does not exist until v1 runs. (The human-bottleneck half of the original rationale dissolved with D-20.) v2 doc §4 | 2026-09-28 |
| D-7 | Defer, don't substitute, on translation-model availability | A model swap invalidates the evidence the language was enabled on | 2026-09-28 |
| D-8 | Speaker labels bypass the translator, and **naming stays before translation** | Naming is baked into the `.txt`; moving it later means a relabel that merges turns and invalidates the unit map (C-6) | revised 2026-09-28 |
| D-9 | The language override is a feeds-spec key | Needs a model field **and** an allowlist entry, and it reaches the whole run via `model_copy` | revised 2026-09-28 |
| D-10 | **Phase 0 ships before Gate V**, on its own | A correctness fix on the existing corpus, and the bake-off depends on it | 2026-09-28 |
| D-11 | **Revised.** The badge and the filter are **v2**, shipping together | A language chip was deliberately deleted in #2115 because the corpus is monolingual; that reasoning expires exactly when a second language arrives. v1 delivers the data, not the chrome. Operator confirmed the reversal is intended | revised 2026-09-28 |
| D-12 | The language filter is its own control, not an option inside the played/downloaded filter | The dimensions are orthogonal | 2026-09-28 |
| D-13 | **Withdrawn.** Serbian is not demoted; superseded by the D-29 roadmap, where it sits in tier 2 | Demoting on a Cyrillic-referenced large-v2 number plus two language lists skipped three checks costing about a day (§6.3) | withdrawn 2026-09-28 |
| D-14 | **Same-language retrieval only, via a separate keyword-only table.** Non-English chunks live in their own table with no vector column; no embedding model changes | A vector-less row **cannot** surface in a semantic result, which a row tag plus a filter cannot guarantee — and a zero vector would actively outrank most real results. It also leaves the existing table untouched, so it should avoid the stale-index outage a column addition forces. Cross-lingual semantics is v2 | revised 2026-09-28 |
| D-15 | **Withdrawn.** No embedding-model change in v1, so its blast radius is moot | Superseded by D-14; findings preserved in v2 doc §7 | withdrawn 2026-09-28 |
| D-16 | **Withdrawn.** The flag gates the pipeline; visibility is controlled by **when the feed is added to the production feed list** | There is no per-episode serving gate: ~32 modules walk the corpus independently and the indexer walks metadata directly, so a catalog filter would not stop search, CIL, MCP or digest. A separate corpus root was considered and rejected — one corpus, no split. Which feed is in the production feed list is config, not code | withdrawn 2026-09-28 |
| D-17 | **Withdrawn.** Nothing about labelling or gating ships in Phase 2 | Superseded by D-36 and D-37 | withdrawn 2026-09-28 |
| D-18 | Decisions here graduate to **ADRs** as they are implemented | The engineering process puts decisions in ADRs; this many living only in an arc note is process drift. V.4 is the first | 2026-09-28 |
| D-19 | **Translation runs after transcription and diarization, before summary** — in **one** seam, inside `generate_episode_metadata` | Summary output feeds GI topic labels and KG topics, so translating later gives English insights on Greek topics and fragments cross-episode identity. One seam covers ASR, cache hits, direct downloads, publisher transcripts and every reprocess cascade; "after transcription" names a seam that does not exist for publisher-transcript episodes | refined 2026-09-28 |
| D-20 | **A native-speaker reviewer is replaced by an LLM judge**, validated by fault injection | The reviewer was never identified, had no protocol, and gated Phases 2–4. Operator decision. The protocol and its three trust rules are in v2 doc §9; they apply to V.3 | 2026-09-28 |
| D-21 | **The normalizer is trivial, and odd input is an onboarding task** | Lowercase, primary subtag, keep `language_raw`. No `und`/`zxx`/`mul` policy, no three-letter mapping, no script parsing. Feeds are onboarded manually a couple of episodes at a time, so a human inspects every one — a defensive code branch for input that process will never pass is machinery for nothing. The registry already skips anything not `enabled`, and the per-feed override (S0.3) is where a wrong tag gets corrected | 2026-09-28 |
| D-22 | **The local Whisper tier is removed from the DGX profiles' chains** | The DGX Whisper is one multilingual model; you pass the code. The local tier defaults to `base.en` and cannot usefully transcribe anything else. Without it, prod has **no language-driven model selection at all** — which deletes the per-episode model-resolution work outright. The provider stays for the eight local/dev/airgapped profiles where it is primary, with a guard refusing non-`en` | 2026-09-28 |
| D-23 | **No change to the episode status `Literal`** | `status="skipped"` plus a reason for an unsupported language, `status="failed"` plus a reason for a transcription failure. A `skip_reason` pattern already exists. `translation_pending` is a translation-stage outcome in the ledger, not an episode status | 2026-09-28 |
| D-24 | **No transliteration and no alias minting. Speaker labels are carried verbatim** | Every tier-1 language is Latin script and a person's name is usually the identical string across them, so there is nothing to convert. Labels still bypass the translation model, which would rename the same person inconsistently between units. Revisit only when a non-Latin-script tier is enabled (v2 doc §5) | 2026-09-28 |
| D-25 | **The transcript defaults to English, with a control to switch to the original** | The rest of the app — summary, insights, search — is English, so a source-language transcript by default is the inconsistent choice. The control is in the transcripts panel and ships in **v1**. It must serve the full-timeline English, not the ad-free analysis base, or it desyncs from the audio where ads were cut | 2026-09-28 |
| D-26 | **`LanguageBadge` is built in v1 as the transcript control; badges as metadata decoration stay v2** | Splitting the component from its placement keeps the #2115 reasoning intact: a badge on every show in a monolingual corpus was noise, but a control where there is an actual language choice to make is functional | 2026-09-28 |
| D-27 | **One translation model, chosen for breadth and for running reliably; the comparison is v2** | v1 needs a model that is good enough across the tier-1 languages, not the proven best of five on six axes. Serbian is explicitly **not** a selection criterion. Catalan is the only tier-1 language where candidates differ — MiLMMT-46-12B covers it, LMT-60-8B does not — so if Catalan is in the initial set that narrows the choice | 2026-09-28 |
| D-28 | **Gemma's terms impose nothing on outputs** — verified, and an earlier claim here was wrong | §3.3 of the terms: *"Google claims no rights in Outputs you generate."* No attribution, no notice, no pass-on for generated text; pass-on applies only to redistributing the model or a derivative, which we do not do. So licence is **not** a differentiator between the candidates — pick on quality and whether it runs. GDPR is not a multilingual question either: the corpus already holds attributed statements by named people, translation adds no new category, and it needs its own legal review rather than a line in a design doc | 2026-09-28 |
| D-29 | **Language roadmap in three tiers**, seeded into the registry with only `en` enabled | **Tier 1** (the focus): Dutch, German, Italian, Spanish, Catalan, French, Portuguese, Swedish, Norwegian — every one under 10% FLEURS WER, five under 5%, and **all Latin script**, which is why D-24 holds. **Tier 2**: Russian, Serbian, Bulgarian, Romanian. **Tier 3**: Korean, Japanese, Chinese, Arabic. The tiers are market-ordered, not difficulty-ordered — Russian (5.6) and Japanese (5.3) are technically easier than Norwegian (9.5), and Serbian (33.9) is the hardest language in all three tiers by more than double | 2026-09-28 |
| D-30 | **Keyword recall through the English tokenizer is accepted, measured on the pilot** | The search index applies English stemming, stop-words and accent folding to every language. For tier 1 that costs some recall — verb forms will not collapse the way English ones do — while accent folding helps. No pre-set threshold: look at how search behaves on the pilot feed. A per-language index table is only worth building if a heavily inflected tier-2 language is enabled | 2026-09-28 |
| D-31 | **Translation runs as its own service, and the first pick is TranslateGemma-12B** | A dedicated vLLM beside the existing Whisper and diarization services, intended as **reusable translation infrastructure beyond this project** — which is why the "extra service" is an asset rather than a dependency. The size choice is not a memory constraint (measured: 74.6 GiB available on the DGX with everything loaded, and the served LLM is FP4 at ~15–18 GB, not bf16 — an earlier claim that a 27B would not co-reside was **wrong**). It is a quality-per-throughput choice: on WMT24++ the 12B scores MetricX 3.60 / Comet22 83.5 against the 27B's 3.09 / 84.4, so the step up is 0.51 / 0.9 while the 4B→12B step is 1.72 / 3.4 — steep diminishing returns. TranslateGemma-**12B also beats base Gemma-3-27B** (4.04), so the fine-tune is worth more than the size. 12B leaves ~50 GB headroom for the shared-service ambition and is faster on a per-unit workload of ~400 units per episode. Upgrade paths kept open: a 27B, or an FP8 27B at roughly a bf16 12B's footprint | 2026-09-28 |
| D-32 | **Units are the translation context; sentences are the alignment atom** | A ~120-word block cannot also be a subtitle cue or an ad-excision atom. The ad-free builder drops any segment overlapping an excised range, so unit-sized segments would discard up to ~45 s of real speech per ad boundary, and a 120-word cue is a paragraph. Numbered sentences in, numbered output of equal length, retry then fall back. **This is the one thing v1 cannot cheaply reverse** — changing granularity later re-translates the corpus | 2026-09-28 |
| D-33 | **Units carry a content key, and translations are remembered** | A translation memory keyed by `(source_language, model@revision, src_text)` plus a content-hash key beside the ordinal `unit_id`. Turns out a rename does **not** merge turns — coalescing is by equal *adjacent* labels, so only prefix lengths and offsets change, never unit text. So with the memory, a relabel or re-render costs **zero GPU**. Without it every naming repair on a translated show pays for a full re-translation, and naming repair is the most common repair in this corpus | 2026-09-28 |
| D-34 | **Speaker naming uses the ad-detection trick: translate first, then run the existing English cue matchers on the English text** | The naming cue matchers and NER are English (`roster.py:1472-1494`, `hosts.py:1506-1520`, `detection.py:60-68`) and the one language-agnostic layer is **closed-list** (`resolution.py:202-204`), so a non-English feed's voices stay `SPEAKER_01` — which means no SPOKEN_BY edge, which means `position_arc` matches nothing and **a translated episode yields zero position-bearing insights**. Rather than maintaining per-language cue lists, diarize to anonymous labels, translate, run the English matchers on the English transcript, map names back through the turn, and re-render both transcripts. Cheap only because of D-33. Also translate the title and description so NER candidate discovery works. What stays before translation is **diarization**, not naming — this revises D-8 | 2026-09-28 |
| D-35 | **The transcript resolver carries a `purpose`, not one precedence** | `analysis` wants `.en.adfree.txt`; `timeline` (the player, the viewer transcript route, the segments view) wants the **full-timeline** text. Collapsing both into a single precedence would desync the player from the audio — the exact drift `segments_view.py` exists to prevent. One resolver, two intents, and a written table of which reader has which | 2026-09-28 |
| D-36 | **No user-visible "translated from X" marker in v1** | Operator decision: the listener should see one simple thing with no doubt attached. The chip is genuinely useful and it is v2 work, alongside verification and quality. The honest consequence, stated rather than buried: **in v1 a listener cannot tell a translated quote from a native-English one.** It matters most for quotes, since those can be passed on as somebody's words — which is why v2 prioritises it. Provenance is still written on every claim (S2.11), so v2 adds the label without reprocessing | 2026-09-28 |
| D-37 | **A translated episode has full standing on every surface, exactly as an English one** | We believe the translation. No gating anywhere, no per-surface split, no fail-closed behaviour. Quality is what v2 is *about* — scores, verification, and the label — and hiding output is not a substitute for it. The bet is explicit and mitigated by choosing a good model and by the Gate V check, not by withholding | 2026-09-28 |
| D-38 | **English is the DEFAULT transcript language on every surface, including the player** | Operator decision 2026-09-29, taken when S2.1b surfaced it rather than deferred to S2.8. When `.en.*` exists, both resolver purposes prefer it: `analysis` reads `.en.adfree.txt`, and `timeline` — the player's own precedence — reads `.en.txt`. So a Spanish episode plays Spanish audio with English text by default, which is the same single-path promise as D-1 carried through to what the listener sees. It follows D-37: a translated episode has full standing, and serving its source language by default would be a per-surface split in everything but name. Consequence, stated not buried: **with D-36 there is no marker, so the default view of a translated episode is English text the listener cannot tell is translated.** The source language is never destroyed — `.txt` and `.segments.json` stay canonical — so S2.8's `?lang=` exposes it as a choice rather than needing a reprocess | 2026-09-29 |
| D-39 | **One pipeline for every language. Stages are fixed; each stage evolves internally to handle more languages** | Operator, 2026-09-29. ASR → diarization → naming → translation → summary → GI → KG runs for every episode whatever its language. There is no per-language stage set and no "English stages": a stage that finds nothing to do on a given episode RECORDS that it found nothing, and `pipeline_composition_version` therefore describes the CODE, never the content. The multilingual work is consequently not about branching the graph — it is about the hard-coded English assumptions inside individual stages, which §5.2 measured: ad detection is English regexes, naming is English NER that mints phantom people rather than finding none, the sniff gate over-counts. Those are what S2.14 and the per-stage work address. **Corrects a defect shipped and reverted the same day**, where an English episode was given no translation block so its composition hash would not move — making two episodes off the same commit hash differently because one was Spanish | 2026-09-29 |

| D-40 | **PROPOSED 2026-09-29, pending confirmation. No stage overwrites another stage's output: the pre-naming render is kept as `<base>.anon.txt`** | With naming moving after translation (D-34), the sequence is diarize → write anonymous → translate → name → re-render. The naming re-render would destroy the anonymous transcript, which is the only human-readable view of what the pipeline saw BEFORE it decided who was speaking — the first thing anyone debugging a naming failure wants. It is reconstructible from `.segments.json`'s `speaker` field (the frozen voice id, which naming never touches — only `speaker_label` is updated), so this is cheapness and clarity rather than recoverability. Written once at diarization time, never rewritten. **`.txt` keeps exactly today's meaning** — the canonical NAMED source transcript — so none of the ~30 modules that read it change, and it still exists between diarization and naming, which resume and the audit depend on. The same applies in every language: an English episode gets `.anon.txt` too (D-39). **Separately and more importantly: S2.11's `en_sha256` must hash the LABEL-FREE translation units in `translation.json`, not the named `.en.txt` render.** Hashing the render would invalidate the provenance on every claim whenever naming is re-run, for a reason that has nothing to do with the translation — and translation units are already label-free by construction, since labels bypass the translator (D-24) | proposed 2026-09-29 |

## 8. Open decisions

**None.** The last one — whether the Positions gate should be fail-closed everywhere or split by surface — dissolved when the gate left v1 entirely (D-5 withdrawn, D-37). Everything else was closed on 2026-09-28; see D-21 … D-37 and the withdrawn items D-13, D-15, D-16, D-17.

Two questions dissolved rather than being answered: the non-diarized-episode category does not exist (diarization is a mandatory core stage with a strict validator), and there is no local-transcription-tier question once that tier leaves the DGX profiles' chains.

Record new open questions here as they appear; do not let a settled decision drift back into this list.

## 9. Running notes

**2026-09-29 — a per-language stage record, shipped and reverted.** S2.2 first withheld the `translation` manifest block from English episodes so that `pipeline_composition_version` would not move on the 678 already on disk. Two things were wrong. The hash exists to say which pipeline shape produced an episode, so it moving when a stage is added is the hash working, not a cost — the real cost is one-time and operational, a reprocess query reissued once. And withholding made the hash depend on the episode's LANGUAGE rather than the code: measured, `en -> pc-dba934bc` and `es -> pc-cf90c1cf` off the same commit. The justification was also circular: it cited S0.10's `test_declaring_the_stage_but_never_recording_it_is_the_safe_shape` as a later measurement overriding the plan, but that test was written in the same arc by the same author and encoded an opinion, not a measurement. Corrected to D-39; the guard now asserts the invariant (`test_the_hash_does_not_depend_on_the_episodes_language`) and `_ENGLISH_STAGES` was renamed `_PIPELINE_STAGES`, since the name is what made a per-language stage set look reasonable.

**2026-09-28 — arc opened.** PRD-047 and RFC-123/124/125 landed, reworked against the code rather than
accepted as drafted. QE cut from v1. Phase 0 reframed as "English as a declared language". The
demand/bake-off step renamed **Gate V**. Slice plan added.

**2026-09-28 — evidence pass.** V.1 closed with primary sources (§6): the MT shortlist verified,
Hunyuan's EU carve-out and NLLB's non-commercial licence confirmed, the Whisper FLEURS table extracted
from the paper. Multilingual retrieval brought into scope on the operator's call. DGX capacity removed
as a deployment-time question the operator owns.

**2026-09-28 — first adversarial review; three load-bearing claims false.** C-1, C-2, C-5. New slice for
RSS parsing; D-4 revised; the span chain rebuilt on segments and `unit_id`. Serbian demotion withdrawn
(D-13). Retrieval split. Trust markers and the gate pulled into Phase 2 (D-17).

**2026-09-28 — second adversarial review; the corrections had their own errors.** Found: the Positions
surface list was still incomplete (nine, one of them write-time); `gi/load.py` is CLI-only, not a live
pipeline path; the reader inventory had a bogus entry and was short; `position_arc`'s type filter is a
caller-overridable default; the trigger and gate populations would have diverged, excluding translated
non-claim insights permanently; the marker does not survive four serialization boundaries or snapshots
at all; two chunk sets would break the transcript-lift path; and **D-14's narrowing did not remove the
reindex** — a column addition bumps the index schema version and takes search offline until a full
rebuild. Also: the badge reverses a deliberate deletion (#2115); the feeds-spec override needs a model
field and carries a multi-feed singleton hazard; `Config._normalize_language` only lowercases, so
`en-US` already fails the `is_english` check today.

**2026-09-28 — v1 trimmed, v2 document opened.** The operator's test — *"v1 is a working pipeline and
product end to end; v2 is translation quality and fine edges"* — moved these to
[MULTILINGUAL_ARC_V2](MULTILINGUAL_ARC_V2.md): QE, source verification and the operator worklist, the
badge and the language filter, all three turns consumers, and cross-lingual semantic retrieval. The
native-speaker reviewer was replaced by an **LLM judge validated by fault injection** (D-20), removing
the arc's only human dependency. **D-16 withdrawn** — no serving gate and no separate corpus root, one
corpus; visibility is controlled by when a feed is added to the production feed list. **D-14 settled on
option B** — a separate keyword-only table, which makes a non-English chunk structurally incapable of
appearing in a semantic result and should avoid the schema bump. Nothing implemented; no issues opened.

**2026-09-28 — the trust apparatus left v1 entirely, and v1 got much smaller.** Operator decision
(D-36, D-37): **we believe the translation.** A translated episode has the same standing as an English one
on every surface — no gate, no filter — and **no user-visible marker either**. Provenance is still written
on every claim, invisible to the user, so v2 adds verification, labelling and gating without reprocessing.

This corrected a genuine incoherence rather than a preference: v1 had a read-time gate whose only possible
behaviour was "hide every translated claim, permanently", because nothing in v1 could verify and therefore
nothing could release. That meant paying for translation and extraction to produce data nobody could see.
A gate belongs with its verifier. **D-5 and D-17 withdrawn**; S2.12 deleted; S2.8 shrinks from an L
labelling slice to an S API slice; §8 is empty again because the last open question — fail-closed
everywhere versus split by surface — dissolved with the gate. The nine-surface enumeration and the
two-predicate design are preserved in the v2 notes, where they will be needed.

The honest cost is recorded rather than buried: **in v1 a listener cannot tell a translated quote from a
native-English one.** That matters most for quotes, which get passed on as somebody's words, and it is why
the label is the first thing v2 adds after verification.

**2026-09-28 — final architectural review; the plan changed shape.** A single architect-framed review
(rather than another fault hunt) returned **ready with conditions**, endorsed the spine, and found three
things cheap now and a corpus re-translation later. **(1) Speaker naming on non-English was unassessed and
decides whether the product works at all**: the cue matchers and NER are English and the one
language-agnostic layer is closed-list, so a non-English feed's voices stay `SPEAKER_01` → no SPOKEN_BY →
`position_arc` matches nothing → **zero position-bearing insights**. Fixed by applying the ad-detection
trick to naming (D-34, the operator's own suggestion): diarize anonymously, translate, run the existing
English cue matchers on the English text, map names back, re-render. **(2) The ~120-word unit was the
wrong atom** for subtitle cues and ad excision — sentences are now the alignment atom (D-32), and this is
the one thing v1 cannot cheaply reverse. **(3) "One resolver" as written would desync the player** — it
now carries a `purpose` (D-35). Plus the translation memory (D-33), which is what makes D-34's relabel
free and stops every future naming repair paying for a re-translation.

Plan re-shaped: a new **Phase −1** puts the resolver refactor and the allow-list test *before* everything,
since both are pure-English instruments the rest is measured by. S0.3 folded into S0.2, S0.9 into S0.8,
S1.3's backfill moved to v2, the two `.en.*` turns variants dropped (no v1 reader), S2.9 re-sized to L,
Corrected: `topic_consensus` *is* reachable by the property predicate (it
loads `gi.json` itself); script detection does not discriminate for tier 1, since *inflación* and
*inflation* share a script. Also closed: **TranslateGemma-12B** is the first pick (D-31), and the
27B-won't-fit claim was **wrong** — measured 74.6 GiB available on the DGX with everything loaded, and the
served LLM is FP4 not bf16. The ad-excision gap became its own defect, **issue #2168**, rather than arc
scope. One open decision remains, and it is a product judgement (§8).

**2026-09-28 — every open decision closed; §8 was empty at this point.** D-21 … D-30. Highlights: the normalizer is
deliberately trivial because feeds are onboarded manually, so odd input is an onboarding task and not a
code branch (D-21); the local Whisper tier leaves the DGX profiles' chains, which deletes the
per-episode model-selection work outright (D-22); **no transliteration and no alias minting** — every
tier-1 language is Latin script (D-24); the transcript **defaults to English** with a control to switch
to the original, in v1, which pulls the badge *component* into v1 while badges-as-decoration stay v2
(D-25, D-26); one translation model chosen for breadth, comparison deferred (D-27); and a **three-tier
language roadmap** (D-29) whose tier 1 — Dutch, German, Italian, Spanish, Catalan, French, Portuguese,
Swedish, Norwegian — is entirely under 10% WER and entirely Latin script. Pilot is Spanish or Italian.
Two questions dissolved rather than being answered: diarization is already a mandatory core stage with a
strict validator, so there is no non-diarized category; and there is no local-tier question once the tier
is gone. Gemma's terms were read: **nothing is owed on outputs**, so licence is not a differentiator
between candidate models — an earlier claim of mine to the contrary was wrong. Serbian is explicitly not
a selection criterion and sits in tier 2.

**2026-09-28 — the backfill is specified.** S0.1b is a one-off migration
that walks **shows**, fetches each feed's `<language>` once, and writes it onto the show and every
episode under it — the language belongs to the feed, so one fetch covers all of its episodes. Versioned
and re-runnable as a migration rather than a script someone remembers, with a `--dry-run` that prints the
distribution first and per-show reporting that feeds S0.4. Re-sized M from L.

<!-- Append new entries above this line. Decisions go in §7 with a D-number; facts about the code go in
     §5; claims found false go in §5.4; anything deferred goes in MULTILINGUAL_ARC_V2.md, never deleted. -->
