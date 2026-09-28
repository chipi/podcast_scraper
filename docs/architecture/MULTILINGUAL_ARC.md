# Multilingual ingest — arc notes

The one page that holds this arc together. The PRD says what the product needs, the three RFCs each
own a slice of the how, and this document is where the arc's **shape**, its **slice plan**, its
**verified code facts**, its **decisions** and its **running notes** live — so none of that has to be
re-derived from four documents or rediscovered from the code.

- **Arc**: multilingual ingest (source-language capture, English-normalized intelligence)
- **Opened**: 2026-09-28
- **Status**: design — nothing implemented
- **Branch**: `feat/multilingual-ingest`
- **Documents**: [PRD-047](../prd/PRD-047-multilingual-ingest.md) · [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) · [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) · [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md)
- **Review state**: §5 and §4 were rewritten on 2026-09-28 after an adversarial review found three
  load-bearing code claims false. §5.4 records what was wrong and why, because the *way* they were
  wrong is a reusable warning.

---

## 1. The arc in one page

The corpus is English-only by configuration, not by architecture. The bet is that we can ingest a
Greek show, keep **what was actually said** as the canonical record, and derive an English layer that
every existing intelligence stage reads without a per-language fork.

```text
audio (any enabled language)
  │
  ├─ transcribe + diarize IN SOURCE LANGUAGE ──────► ep1.txt            ← canonical, full timeline
  │                                                  ep1.segments.json
  │                                                  ep1.turns.json     ← RFC-123
  │
  ├─ TRANSLATE every turn-bounded unit ────────────► ep1.translation.json  ← the traceability map
  │   (BEFORE summary — D-19)           ep1.en.txt            ← derived, full timeline
  │                                                  ep1.en.segments.json  ← doubles as subtitles
  │
  ├─ ad-detect + excise ON THE ENGLISH ────────────► ep1.en.adfree.txt   ← THE ANALYSIS TRANSCRIPT
  │
  ├─ summary → GI → KG, all reading that file ─────► single-path, English
  │
  └─ Positions (read-time CIL arc) ────────────────► gated on source verification
```

Three properties make it work:

1. **The source is canonical, the English is derived.** A translation is an interpretation; the
   record is what was said. Every English span maps back to source text, source speaker and source
   audio time.
2. **Analysis reads one transcript.** Every transcript reader resolves through one function, so no
   stage decides for itself what "the transcript" means. That is not true today (§5.4 C-1) and making
   it true is part of the work, not an assumption the design rests on.
3. **The product never presents a translation as verbatim speech.** Labelled everywhere, traceable
   on tap, and — for anything that feeds a Position timeline — verified against the source first.

## 2. Document map

| Document | Owns | Does **not** own |
| --- | --- | --- |
| [PRD-047](../prd/PRD-047-multilingual-ingest.md) | Why, for whom, what "done" means, phase gates, language policy, operator + listener surfaces | Any implementation shape |
| [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md) | `turns.json` — turns and sentences as addressable units. **Ships alone, on English value.** | Anything language-specific |
| [RFC-124](../rfc/RFC-124-multilingual-transcription-and-translation.md) | Language resolution, source capture, the translation stage, the artifact set, the pipeline order, retrieval, the model bake-off | Trust, confidence, gating |
| [RFC-125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) | Source verification of claims, the read-time Positions gate, the review worklist. QE deferred to v2. | Producing the translation |
| This document | Arc shape, slice plan, verified code facts, decisions, running notes | Requirements or design detail |

## 3. Phase ladder

Phase numbering means one thing across all four documents. The demand/bake-off validation step is
**Gate V**, not a phase, because it produces evidence rather than software.

| Phase | What | Gate to start | Visible outcome |
| --- | --- | --- | --- |
| **0 — English as a declared language** | Language becomes a real, parsed, resolved, validated, *displayed* property of the corpus we already have | none; it is a correctness fix | An `EN` badge sourced from the feed's own tag; an audit over real data; no code path that substitutes a language it was not given |
| **1 — Turns (RFC-123)** | `turns.json` + backfill, then GI attribution, search chunking, sentence cues | none; English value stands alone | Better quote attribution and speaker-true search on today's corpus |
| **Gate V — Validate** | Demand interviews; the Serbian investigation; the per-language bake-off and its quality gate | Phase 0 for the bake-off; **V.5 needs nothing and starts now** | A go/no-go with evidence, and a pilot language chosen on measurement |
| **2 — Translation (RFC-124)** | Source capture, translation stage, English render, ad-free-on-English, routed readers, retrieval | Gate V passed | One gated non-English feed fully processed, findable in both languages, and labelled |
| **3 — Trust (RFC-125)** | Source verification and the operator worklist | Phase 2 producing artifacts | Translated claims can enter Position timelines — and cannot before |
| **4 — Surfaces** | Language toggle, original-text reveal, language filters | Phase 2 | A listener reads the original or the translation, and filters by language |

**Why Phase 0 is a real phase.** Today a feed's declared language is never read, every episode is
stamped with the run config, and a non-English episode would be transcribed by a chain that can fall
back to `base`. Phase 0 makes language an *asserted, checked, visible* fact — which is also exactly
the plumbing translation needs. It is testable end to end on the existing corpus, with no new models
and no GPU, and it is the recommended standalone ship.

## 4. Slice plan

Each slice is sized to be **one GitHub issue**: one goal, its own tests, its own acceptance criteria,
and shippable without leaving the tree half-built. Nothing is opened yet.

*Depends on* is a hard ordering constraint. *Size*: **S** = one sitting; **M** = a day-ish, multiple
files, real test surface; **L** = multi-day, new subsystem or a migration. *Ship alone?* asks whether
merging only this leaves `main` correct and coherent.

**This plan was re-sliced on 2026-09-28** after review. Slice ids changed; no issues had been opened,
so renumbering was free. Sizes that the review showed to be wrong were raised rather than defended.

---

### Phase 0 — English as a declared language

**Ships as one release.** No new models, no GPU. At the end, "this episode is in English" is something
the system parses, checks and displays rather than assumes — and the same machinery carries any other
language.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S0.1** | Parse and persist the feed's declared language | The core plumbing, and it does not exist today (§5.4 C-2). Extract the channel `<language>` in `rss/parser.py`; add it to `RssFeed` and `FeedMetadata`; persist `feed.language_raw`, `feed.language`, `feed.language_source`; add an **episode-level** `language` + `language_source`. This is a `*.metadata.json` shape change, so it carries a migration under `upgrade/migrations/`, a `corpus_format_version` bump, a reader-support bump and a `CORPUS_UPGRADE.md` row, plus a backfill over the existing corpus. | — | L | Yes |
| **S0.2** | Language-tag normalization and per-episode resolution | `normalize_language_tag` (`el-GR`→`el`, `sr-Latn-RS`→`sr`, `pt_BR`→`pt`, junk→`None`) and `resolve_episode_language(feed_entry, feed_doc, cfg)`; `config/languages.yaml` registry with `en` the only enabled language. `_feed_language` calls the normalizer so catalog and pipeline cannot disagree. | S0.1 | M | Yes |
| **S0.3** | Per-feed language override in the feeds spec | Cheaper than it looked: `rss/feeds_spec.py` already accepts a mapping per feed entry (`:26-56`), gated by `RSS_FEED_ENTRY_OVERRIDE_KEYS` with `extra="forbid"` (`:77`), and the allowlist requires the key to exist on `Config` — `language` already does. So this is: add `language` to the allowlist, thread it into resolution as highest precedence, and display the resolved value + its source in the shows library. **Moved into Phase 0** because it is the only remedy for a mis-tagged feed, and S0.8's skip path is dangerous without it. | S0.2 | S | Yes |
| **S0.4** | Corpus language audit over real data | Read-only CLI: walk every feed and episode, resolve a language, report the distribution and an explicit list of anything not resolving to `en`, with each item's resolution source. Runs against the corpus **after** S0.1 has persisted real feed tags — before that it can only re-read the run config and would prove nothing. Commit the report. | S0.1, S0.2 | S | Yes |
| **S0.5** | Expose language on the app API | Episode-level `language` (additive) on the episode list and detail responses. Show-level `AppPodcastItem.language` exists but currently serves `cfg.language` (§5.4 C-2) — it starts serving the normalized feed tag. Contract tests for present / absent / pre-migration values. | S0.1, S0.2 | S | Yes |
| **S0.6** | Language badge on every show and episode | `LanguageBadge.vue` (compact square, uppercase ISO code; no generic badge primitive exists in the player, so this is the primitive). Renders on `EpisodeRow`, `EpisodeTile`, `EpisodeCard`, `ShowRow`, `ShowTile`, `PodcastView`, plus the operator viewer's show surfaces. Includes `t()` entries (the player forbids hard-coded user-facing strings), a language display-name source so the aria-label reads "Greek" not "EL", contrast check, Playwright specs in both apps, and the `E2E_SURFACE_MAP.md` + UXS updates the engineering process requires. Omits the badge rather than guessing when language is unknown. | S0.5, open decision 5 | L | Yes |
| **S0.7** | One language reader: thread the resolved language, delete the substitutions | The transcription call site passes the episode's resolved language; the DGX provider stops **reporting** `"en"` for an auto-detected transcript (§5.4 C-3); `ml_provider`'s `self.cfg.language or "en"` goes; `sniff_gate.py`'s four sites are covered; `metadata_generation.py:957/:3832` write the resolved value. Acceptance is a **lint rule**, not a list: `cfg.language` may be read in exactly one place, the resolver's lowest-precedence default. | S0.2 | M | Yes |
| **S0.8** | Select the transcription model from the resolved language, with a non-English floor | Bigger than it looked (§5.4 C-4): `MLProvider` resolves its Whisper model **once at init** from run-global language (`ml_provider.py:566-568`), so per-episode selection needs per-call model resolution. Adds `min_model_for_non_english`, truncates the fallback chain, and fails `deferred_quality_floor` rather than transcribing with `base`/`tiny`. Byte-identical for English. Decide explicitly whether the local tier serves non-English at all (simplest: it does not). | S0.7 | L | Yes |
| **S0.9** | Unsupported-language skip and the language-ID sanity check | Neither was in any slice. `skipped_unsupported_language` for a language that is not `enabled`; the language-ID check on a 30 s window from the middle third, recording `language_mismatch` and flagging for the operator without ever rerouting. Must land **after** S0.3, or a mis-tagged English feed silently stops ingesting with no remedy. | S0.3, S0.4 | M | Yes |
| **S0.10** | Phase 0 observability | How the operator learns something went wrong: a manifest `language` block, and log/metric surfacing for `skipped_unsupported_language`, `deferred_quality_floor` and `language_mismatch`, plus a runbook entry. Without this, Phase 0's new failure modes are silent — the exact property the phase exists to remove. | S0.8, S0.9 | S | Yes |
| **S0.11** | Phase 0 release gate | One full run with language explicitly declared and asserted at every stage; the S0.4 audit re-run clean; a regression test that an episode resolving to `el` can never ship `language: en`; and **byte-identical** English artifacts — defined as transcript / ad-free / GI artifacts identical, with metadata and manifest diffs restricted to an allow-list of added keys (S0.1 and S0.7 deliberately add fields, so unqualified byte-identity is not achievable and was never the right claim). | S0.1–S0.10 | M | This **is** the ship |

**Phase 0 acceptance.** The audit reports the real language distribution of the corpus from parsed
feed tags, with every item's resolution source named; every show and episode renders a badge sourced
from that; no stage can receive a null language and substitute one; a non-English language selects a
multilingual model at or above the floor or fails loudly; the new failure modes are visible; and
English artifacts are unchanged outside the declared allow-list.

---

### Phase 1 — Turns (RFC-123)

Independent of everything multilingual. Pays for itself on the English corpus and is the unit
translation later needs.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S1.1** | `build_turns`: turns and sentences from a segments sidecar | The pure builder plus the §2.4 invariants, property tests and 5 committed goldens. Nothing reads it, nothing is wired. Split out of the old single L slice. | RFC-123 OQ2 decided | M | Yes |
| **S1.2** | Wire `turns.json` into the pipeline for both variants | Emit per variant inside `adfree_transcript` where the ad-free segments are produced, the manifest `turns` block, and the `turns: unavailable` legacy flag. Requires deciding the non-diarized case first: `_derive_offsets_by_find` **skips** segments it cannot locate, which violates "every segment in exactly one turn" — so either non-diarized episodes get no turns, or the invariant is relaxed. Count how many production episodes are non-diarized before choosing. | S1.1 | M | Yes |
| **S1.3** | Backfill `turns.json` across the existing corpus | `podcast-scraper turns backfill --corpus <dir>`, no GPU, idempotent, coverage report. | S1.2 | S | Yes |
| **S1.4** | Switch GI speaker attribution to turn lookup | Char-span binary search into turns, regex kept as fallback. Gated on replaying the 7,101-quote #2062 set and reporting the full transition matrix; zero name→different-name transitions required to flip the default. The existing replay measured direction, not correctness, so the new one reports the matrix rather than a single number. | S1.3 | M | Yes |
| **S1.5** | Turn-bounded search chunking with per-chunk speakers | Chunk within turn boundaries; `SegmentDocument` and the LanceDB segment schema gain `speaker_ids` and `turn_ids`; reindex via the RFC-118 delta path; retrieval eval reporting recall@k and speaker-precision. **Coordinate with S2.9** — both change the same LanceDB schema, and doing them in one schema change is the point. | S1.3 | L | Yes |
| **S1.6** | Player sentence-granularity cues | Optional `?granularity=sentence` serving sentence cues from `turns.json`. Default stays raw segments. | S1.3 | S | Yes |

---

### Gate V — Validate (evidence, not software)

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **V.1** | ~~Verify translation model availability and licences~~ | **DONE 2026-09-28 — §6.1.** Eligible: TranslateGemma 4B/12B/27B (`gemma`), MiLMMT-46-12B (`gemma`), LMT-60-8B (`apache-2.0`), Qwen3-30B baseline. Excluded: Hunyuan/HY-MT (EU carve-out), NLLB (`cc-by-nc-4.0`). TranslateGemma's language list could not be enumerated — its model card is gated. | — | S | closed |
| **V.5** | Settle whether Serbian is supportable — **start now** | No dependency on any phase. Three cheap actions first: (a) rescore FLEURS `sr` with `large-v3-turbo` on the DGX, transliterating both hypothesis and reference to Latin — if WER collapses toward Croatian's 13.4 the ASR objection is gone; (b) probe Qwen3-30B-A3B for Serbian, since it is already served and Apache-2.0 and §6.1 marked its column `—` rather than checking; (c) log in to the gated TranslateGemma card and settle its 55. Only then the Croatian-proxy question. Outcome: a pilot language chosen on measurement. | — | M | n/a |
| **V.2** | Demand check with the beta cohort | What share of their listening is non-English and which shows. Needs the instrument written first: question wording, cohort size, how "≥30%" is computed. | — | S | n/a |
| **V.3** | Per-language bake-off harness and gate report | Bake-off profiles per candidate, scoring script, reviewer CSV export, run on 2–3 episodes of the **pilot language chosen by V.5** plus one Spanish or Italian control. Reports all six RFC-124 §7 measurements including ad-detection survival. Blocked on a named native reviewer — the arc's only human-gated resource. | S0.11, V.5 | L | n/a |
| **V.4** | Gate V decision record | An **ADR** recording which languages passed, with numbers, and the chosen model + pinned revision per language. | V.2, V.3, V.5 | S | n/a |
| **V.6** | Source a test fixture in the pilot language | A short CC-licensed or TTS episode with an injected sponsor read, for RFC-124's integration test. None exists, and none of the design is testable end to end without it. Do not hand-build audio. | V.5 | S | n/a |

---

### Phase 2 — Translation (RFC-124)

Behind `multilingual_ingest`. **The flag gates serving as well as the pipeline** — a translated
episode is not served until its trust and labelling slices are in (D-16).

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S2.1** | Route every transcript reader through one resolver | The work D-4 assumed away (§5.4 C-1). `load_processing_transcript` has two callers; the summary stage, the faithfulness check, transcript NER, `search/indexer.py:96-111`, `gi/repair.py:161-171` and `gi/load.py:19-37` each resolve independently — and `gi/load.py` reads the **raw** `.txt`, a coordinate-space inconsistency that exists today, independent of this feature. Route them all; add the `.en.txt` branch to the precedence. **Ship this in Phase 1 or earlier if convenient — it is a standalone bug fix.** | — | L | Yes |
| **S2.2** | Move translation ahead of summary in the stage order | `CANONICAL_STAGE_ORDER` is `asr, diarization, naming, summary, gi, kg` with no translation slot, and summary runs in the same job immediately after transcription. Summary output feeds GI topic labels and KG topics, so translating after it would give English insights with Greek topics. Adds the stage slot and a `translate_only` reprocess mode for retries (chosen over building a per-episode pending queue). | S2.1 | M | Yes |
| **S2.3** | Translation units and the vLLM client | Turn-bounded unit packing from the source variant; the `dgx_vllm_translate` client with per-episode batching, bounded concurrency and per-unit retry. | S1.2, V.4 | M | Yes — flag-off is a no-op |
| **S2.4** | `translation.json` and the English render | The unit map; the English screenplay and `.en.segments.json` rendered through the existing formatter, one pseudo-segment per unit **carrying `unit_id`** (see S2.5); `translation_pending` and failed-unit semantics. Stub-translator integration test. | S2.3 | M | Yes |
| **S2.5** | Ad-free base on English, and span→unit resolution | Build `.en.adfree.*` with the existing `build_adfree_artifacts` (`adfree_transcript_relpath` already composes `ep1.en.txt` → `ep1.en.adfree.txt`). `resolve_units_for_span` resolves **through `.en.adfree.segments.json` and `unit_id`, not through the ad-map** — the ad-map cannot invert the ad-free transform on the diarized branch, because that branch re-renders survivors and re-emits the `Label:` prefix and its trailing space (§5.4 C-5, measured). Writes the whole English artifact set atomically with an `en_sha256` provenance check the resolver enforces. | S2.4 | L | Yes |
| **S2.6** | Speaker identity across scripts | Naming stays **before** translation, on the source, with English feed metadata as context — naming runs inside `transcribe_with_segments` and its labels are baked into the `.txt`, so naming afterwards would mean a relabel that merges turns and invalidates `translation.json` (§5.4 C-6). Labels bypass the translator; CIL alias lookup then deterministic transliteration with a `_looks_like_person` guard and source-script fallback. | S2.4 | M | Yes |
| **S2.7** | Language-aware reprocess and invalidation | Every reprocess stage (`relabel_only`, `rediarize_only`, ASR, direct-download) calls `_maybe_produce_adfree` on the **source** text, which on a non-English episode writes the identity ad-free artifact §5.2 hazard 4 describes and strands `.en.*` as stale. Make it language-aware, and give every path that changes the source an explicit invalidation of `.en.*`, `translation.json`, verification records and the episode's cached prompt prefix. Includes the per-episode delete/reprocess command none of the design had. | S2.5 | M | Yes |
| **S2.8** | Segments API `?lang=`, translation status, and the translated-content chip | Additive `language` / `machine_translated` / `translation_model` on `SegmentsResponse`; `translation_status` on episode detail; the "Translated from <Language>" chip wherever translated content renders. **The chip is a Phase 2 exit criterion**, not Phase 4 — otherwise translated quotes render unlabelled between S2.12 and Phase 4 (D-17). | S2.4 | M | Yes |
| **S2.9** | Same-language retrieval: index both layers | Tier-1 chunks built from **both** the source transcript and the English analysis transcript, each tagged `language`, both keyed to the same episode — so a Greek query reaches the Greek chunks and an English query reaches the English ones, and both land on the same episode. Chunk **ids must carry language** (`chunk:{scope}:{i}` collides otherwise, and LanceDB merges on id), and insight→segment linking is by time so it must filter on language or an English insight will link a Greek chunk. **Non-English chunks are keyword-only — excluded from the dense stage** (MiniLM vectors for Greek are noise), and **no embedding model changes** (D-14). Requires one measurement: LanceDB's FTS index is created English-only (one tokenizer per index, with stemming and accent folding), so quantify Greek keyword recall before calling this done. | S2.5, S1.5 | M | Yes |
| **S2.10** | Cost and capacity measurement | Translation GPU time per episode, storage delta per translated episode, and the bake-off's own cost. RFC-124 OQ3 proposes a wall-time cap as a gate criterion and it cannot be set without this. Note for model choice, not for capacity: a 27B translator and the 30B Qwen do not co-reside in the DGX's memory while a 12B does — that is a constraint on the bake-off winner. | S2.3 | S | Yes |
| **S2.11** | Trust markers on translated claims (was S3.1) | **Pulled into Phase 2.** The `translation` block (`translated`, `source_language`, `unit_ids`) written into node `properties` via `resolve_units_for_span`. Without it, translated insights carry no marker, so RFC-125's read-time filter is a no-op on exactly those nodes and the gate is not fail-closed (D-17). | S2.5 | M | Yes |
| **S2.12** | Phase 2 gate: one non-English feed end-to-end | A gated feed from audio to insights, findable by a query in either language, every quote resolving to source text and source audio, every translated claim labelled and marked, and no English regression. | S2.1–S2.11, S3.1 | M | This is the ship |

---

### Phase 3 — Trust (RFC-125, verification-only)

QE is not here — D-6. Note S2.11 (the marker) moved to Phase 2.

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S3.1** | Gate Positions on verification (read-time filter) | Include a translated insight only when its outcome is `verified`. Applies to **every position-bearing read path**, not two: `position_arc`, `topic_conversation_arc` (which reuses `topic_timeline`), `topic_timeline_merged`, `person_profile`'s insights-by-topic, and `topic_perspective_leaders` — one predicate helper at insight selection, plus a written list of which surfaces count as Positions. Ships **before** the verification pass: with no records in existence it excludes every translated claim, which is the desired default. | S2.11 | M | Yes |
| **S3.2** | Source-grounded entailment verification | Source-language cited units plus one turn of context each side, the English claim, a JSON `supports / contradicts / insufficient` verdict from the served Qwen, the record on the node, and the cross-model re-translation fallback on `insufficient` (which must trigger S2.7's invalidation). Batched per episode. | S3.1 | L | Yes |
| **S3.3** | Operator review worklist | JSONL export plus a minimal operator view of contradicted and unverified claims, source and translation side by side, with verify / reject / re-translate actions and operator provenance. | S3.2 | M | Yes |

---

### Phase 4 — Surfaces

| # | Issue title | Goal | Depends on | Size | Ship alone? |
| --- | --- | --- | --- | --- | --- |
| **S4.1** | Filter shows and episodes by language | A language control in the consumer episode toolbar (`CatalogView`, whose filter is single-select via `ListToolbar` today), on show browse, and in the operator library filter bar. Its **own** control, not an option inside the played/downloaded filter, so "Greek and unplayed" is expressible. Rendered only when the corpus holds more than one language. | S0.5, S2.12 | M | Yes |
| **S4.2** | Transcript language toggle and original-text reveal | Read the source transcript or the English subtitles against the original audio; tapping a translated quote reveals the source sentence and plays the source span. UXS first. | S2.8 | L | Yes |
| **S4.3** | Feature-flag lifecycle and phase rollback | What `multilingual_ingest` gates at each phase, its removal criterion, and the rollback procedure per phase — including what happens to a language that is **disabled** after episodes exist in it (still served? still in arcs? reprocessed?), which no document currently answers. | S2.12 | S | Yes |

---

### Deferred to v2

| # | Issue title | Why deferred |
| --- | --- | --- |
| **V2.1** | Per-unit quality estimation and calibrated bands | D-6. Needs a native reviewer per language to fit thresholds — a human bottleneck — and a translated corpus to calibrate against, which does not exist until Phase 2 runs. |
| **V2.2** | Word-level anchors via forced alignment | Per-language aligner checkpoints; `turns.json` is already forward-compatible (`timing: word_aligned`). |

### Explicitly not in this arc

**Cross-lingual semantic retrieval** — an English query finding Greek content *by meaning*, and vice
versa. It needs a multilingual embedding model, which re-embeds all existing English episodes, changes
vector dimensionality (a migration and a reader-support bump), forces a corpus-wide reindex with a
cutover and rollback, and drags along `search/insight_clusters.json`, `kg/topic_clustering`,
`search/query_router` and `search/quality_metrics.py`'s hardcoded `_ZERO_VECTOR_DIM = 384` — all
gated on a prod-derived eval set that does not exist yet.

The operator's decision (2026-09-28): **not now.** Same-language retrieval (S2.9) delivers the actual
requirement — Greek words find the Greek episode, English words find the English layer. Matching by
meaning across languages is a complication this product does not need yet. It becomes genuinely useful
once the **application itself is internationalized** and a listener can search in whatever language
they prefer; revisit it then, as part of that work rather than as part of this arc.

---

### Critical path

```text
S0.1 → S0.2 → {S0.3 → S0.9, S0.4, S0.5 → S0.6, S0.7 → S0.8} → S0.10 → S0.11 ══ PHASE 0 SHIPS ══╗
                                                                                                ▼
S1.1 → S1.2 → S1.3 → {S1.4, S1.5, S1.6} ══ PHASE 1 ══╗      V.5 (starts NOW) ──┐                ║
S2.1 (standalone bug fix, any time) ─────────────────╫──────────────────────────┤               ║
                                                     ║   V.2 ──┬─ V.3 ← V.6 ────┴─ V.4 ══ GATE V ═╣
                                                     ▼         ▼                                ▼
        S2.2 → S2.3 → S2.4 → S2.5 → {S2.6, S2.7, S2.8, S2.9←S1.5, S2.10, S2.11 → S3.1} → S2.12 ══ PHASE 2
                                                                                        ▼
                                                                    S3.2 → S3.3 ══ PHASE 3
                                                                                        ▼
                                                                       {S4.1, S4.2, S4.3}
```

Phase 0 is close to serial: `S0.1 → S0.2` gates everything, and the migration in S0.1 is the long
pole. For one operator the two human-gated items — a native reviewer (V.3) and the demand instrument
(V.2) — have the longest lead time and should be started first even though they are not blocking.

## 5. Code facts this arc rests on

Verified against the source. §5.4 records the claims an adversarial review found **false** on
2026-09-28, and is the most useful part of this section.

### 5.1 Load-bearing findings that survived review

| Finding | Evidence | Consequence |
| --- | --- | --- |
| **There is no stance-extraction stage, and deliberately none.** Stances are GI insights; Positions are a read-time query. | `server/cil_queries.py:636` `position_arc`, `:842` `topic_conversation_arc`; `enrichment/profile_sets.py:144-146`; ADR-108 update 2026-07-08 retiring `stance_timeline`. A repo-wide word search for `stance` finds only comments. | The trust gate is a **read-path filter** — retroactive over the existing corpus, fail-closed on a missing record. No stance-extraction dependency exists. |
| **`vector_embedding_model` is genuinely wired to the search index** and is not vestigial. | `search/indexer.py:577` passes `cfg.vector_embedding_model` into `build_two_tier_index`; the query side reads the model recorded in the index (`search/hybrid_search.py:238-244`). | A swap plus reindex is coherent for the segment and insight tiers. The seam D-15 needs is real — see §5.4 C-7 for what D-15 got wrong about *why*. |
| **`GiArtifact` forbids extra top-level keys**; `EvidenceSpan` / `SupportingQuote` / `SegmentsResponse` are plain models. | `gi/contracts.py:130` `ConfigDict(extra="forbid")`; `:14`, `:25`; `server/schemas.py:24`. | Everything additive lives in node `properties`; API fields are safe to add. |
| **The ad-free identity hazard is real** (§5.2 hazard 4). | `gi/ad_regions.py:409-418`; `workflow/adfree_transcript.py:104-106`, `:129-137`; `load_processing_transcript:237-250` sets `is_adfree=True` on file existence alone. | Translation must precede ad detection (D-3). One caveat: `build_adfree_artifacts` returns `None` when there are no segments, so the identity artifact only appears for episodes that have a segments sidecar. |
| **`adfree_transcript_relpath` composes `ep1.en.txt` → `ep1.en.adfree.txt`** unchanged. | `workflow/adfree_transcript.py:52-55` (`splitext` + `.adfree` + ext). | The English ad-free artifacts need no new path helper. |
| **The per-feed override is nearly free.** | `rss/feeds_spec.py:26-56` already accepts a mapping per entry; `:77` `extra="forbid"` with `RSS_FEED_ENTRY_OVERRIDE_KEYS`, whose keys must exist on `Config` — `language` does. | S0.3 is adding a key to an allowlist, not designing a config surface. |

### 5.2 The silent hazards for non-English audio

Five, not four. Each fails quietly rather than loudly.

1. **Silent degradation to an unusable model.** `normalize_whisper_model_name` drops `.en` for
   non-English then builds a chain down to `tiny`; prod's local fallback default is `base.en`
   (`config_constants.py:332`), so for Greek the chain is `["base", "tiny"]`. The DGX path never
   passes through the normalizer — `dgx_whisper_model` is an HF id — so this hazard lives entirely in
   the fallback tier. → **S0.8**
2. **No per-episode language path exists.** Providers read run-global config when the argument is
   absent (`ml_provider.py:827/:885`), the transcription call site passes exactly that global
   (`episode_processor.py:2296`), and `sniff_gate.py:116,130,144,177` threads it four more times.
   → **S0.7**
3. **The DGX provider misreports the language** (corrected — see §5.4 C-3). → **S0.7**
4. **Ad excision silently no-ops on non-English text.** `gi/filters.py:35-77` `_AD_PATTERNS` all
   require an English token, so a Greek transcript yields no ranges and `build_adfree_artifacts`
   emits an **identity** ad-free base with `is_adfree: True` and the sponsor reads intact in the
   analysis text. Not absolute: a Greek episode whose host reads an English URL ("visit X dot com")
   *would* match. → **S2.5**, measured by **V.3**
5. **The sniff gate keeps the cheap transcript on non-English audio.** `sniff_gate.py:63-70` decides
   whether the small-model transcript is good enough by counting entities with spaCy
   `en_core_web_sm`. On Greek that count is ~0, so the gate keeps the sniff transcript. Off in every
   profile today (no `dgx_whisper_sniff_model` set), but one config line from live. → **S0.7**

### 5.3 Surfaces the arc touches

| Surface | Component / file | Used for |
| --- | --- | --- |
| Consumer episode list + toolbar | `web/learning-player/src/views/CatalogView.vue:43-58` — single-select filter via `ListToolbar` (all / unplayed / played / insights, plus `downloaded` only when native), a `show` selector and a sort | S4.1 |
| Consumer episode items | `EpisodeRow.vue`, `EpisodeTile.vue`, `EpisodeCard.vue` | S0.6 badge |
| Consumer show items | `ShowRow.vue`, `ShowTile.vue`, `PodcastView.vue`, `BrowseView.vue` | S0.6 badge, S4.1 filter |
| Reusable chip control | `TypeFilterBar.vue` — generic multi-select kind chips, built for reuse | S4.1 |
| Operator shows library | `library/{ShowsBrowse,ShowsView,ShowDetailView}.vue`, `LibraryFilterBar.vue`, `chips/LibraryFeedChip.vue` | S0.3 display, S0.6, S4.1 |
| i18n | `web/learning-player/src/i18n/` — the player forbids hard-coded user-facing strings; `gi-kg-viewer` has no i18n layer | S0.6, S4.1 |

No generic badge primitive exists in the player, so `LanguageBadge.vue` is the primitive.

### 5.4 Claims that were WRONG, and the pattern behind them

Recorded because the failure mode is reusable, not to dwell on it. **Three of these came from
verifying a reader and assuming its writer; one from citing a line without checking which code path
it sat on.**

- **C-1 — "`load_processing_transcript` is the single resolver all NLP consumers use" is a docstring,
  not a fact.** Two callers (`metadata_generation.py:4835` GI, `:5082` KG). Independent resolution
  lives in `search/indexer.py:96-111`, `gi/repair.py:161-171`, `gi/load.py:19-37` (which reads the
  **raw** `.txt` — a live coordinate-space inconsistency today), the summary stage at
  `metadata_generation.py:2926-2928` and the faithfulness check at `:2401-2405`. **D-4's "one
  function changes, consumers change nothing" was false** → slice S2.1, and the ordering problem in
  S2.2.
- **C-2 — the RSS `<language>` tag is never parsed.** A sweep of `rss/` finds nothing;
  `models/entities.py:16-44` `RssFeed` has no language field. `feed.language` is written by
  `metadata_generation.py:957` as `language=cfg.language`, so `_feed_language`
  (`corpus_catalog.py:159`) reads back the run config and `AppPodcastItem.language`
  (`schemas.py:1719`, served at `app_episodes.py:156`) serves `"en"` by construction. **So "the feed
  tag is already read" and "it carries the raw RSS tag" were both false**, the `en-US → en`
  normalization had no input, and an audit over today's metadata would have been a tautology →
  slice S0.1.
- **C-3 — the DGX hazard was misdescribed.** `whisper_provider.py:197` is in the **returned result
  dict**, not the request. The request (`:429-430`) *omits* `language` when it is `None`, so the
  server auto-detects and the provider then **reports** `"en"`. The hazard is a provenance lie, not a
  forced English transcription.
- **C-4 — model selection happens once at init.** `ml_provider.py:566-568` resolves the Whisper
  model at load from `self.cfg.language`, so per-episode selection is a provider change, not a
  parameter change → S0.8 re-sized to L.
- **C-5 — the ad-map cannot invert the ad-free transform.** Measured: on the diarized branch
  `build_adfree_artifacts` drops overlapping segments and **re-renders** the survivors, re-emitting
  `Label:` prefix plus its trailing space, so `_shift_for` is off by a label per excised range and by a whole segment when
  one straddles a boundary. RFC-124's span → ad-map → unit chain was wrong at hop 1; resolution goes
  through `.en.adfree.segments.json` and a `unit_id` carried on the pseudo-segment instead → S2.5.
- **C-6 — speaker naming cannot run after translation as written.** Naming happens inside
  `transcribe_with_segments` (`ml_provider.py:1061`) and its labels are baked into the `.txt`, from
  which source turns and `translation.json` derive. Naming afterwards means a relabel, and a relabel
  that merges two `SPEAKER_xx` into one name **shifts every turn id** and invalidates the unit map
  → S2.6 keeps naming before translation.
- **C-7 — D-15's containment was right by accident.** `gi_embedding_model` is read by **nothing** in
  GI; GI hardcodes MiniLM (`gi/about_edges.py:28`, `gi/chunked_extraction.py:49`), as do
  `builders/bridge_builder.py:121`, `identity/resolver.py:50`, `kg/topic_clustering.py:43`,
  `search/hybrid_search.py:39` and `search/query_router.py:92`. "Four places" was off by six, and
  `search/insight_clusters.json` is a **search** artifact built with the hardcoded default, so it
  would not follow `vector_embedding_model` — "exactly two things change" was false within the
  search subsystem itself → S4.3 carries the real list.
- **C-8 — RFC-125's code mechanics were wrong in three places.** `_apply_route_and_tag` does not set
  `surfaceable=False` (`_apply_voice_flags` does, `gi/pipeline.py:1100-1124`). `position_arc`'s row
  predicate is SPOKEN_BY-supported quote ∩ `ABOUT` edge ∩ `insight_type == "claim"`
  (`cil_queries.py:650-682`) — it never reads `surfaceable` or `speaker_id`, so "the trigger and the
  gate cannot drift apart" was false. And **there are no KG evidence spans at all**, so "the KG
  evidence writer does the same" referred to nothing. Also, the GI artifact builder is not the only
  writer of those nodes: `add_spoken_by_edges(replace=True)` and `gi/repair.py` rewrite `gi.json`
  in place.
- **C-9 — Appendix A was incomplete and its closing claim false.** Every number listed matched
  Table 13, but **French (8.3, Cluster B)** was omitted along with Arabic 16.0, Azerbaijani 23.4 and
  Maori 38.5 — so "the remaining low-resource languages above 40%" was untrue, and the arc's own
  version appended Kannada 37.0 directly beneath it.
- **C-10 — two overstatements in opposite directions from one `?`.** "Greek is covered by every
  eligible MT model" (TranslateGemma's `el` is unconfirmed) and "Serbian by none of them" (the same
  table says "only via TranslateGemma's unconfirmed 55").
- **C-11 — smaller ones.** The `ner.py:133` gate applies only when `cfg.ner_model` is unset — it
  selects the default model, so a configured model runs on any language. The cloud providers fall
  through to `None`/auto and never substitute `"en"`, so there was nothing there to remove. There are
  15 `bakeoff_*.yaml` profiles, not 14. And "the index build runs in Docker" was cited to this
  document, which never said it — that was session memory wearing a citation.

## 6. Model and ASR evidence (verified 2026-09-28)

This closes slice **V.1**. Where something was **not** verified it says so.

### 6.1 Translation model shortlist

| Model | HF id | Params | Licence | `el` | `sr` | `hr` | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TranslateGemma 27B / 12B / 4B | `google/translategemma-{27b,12b,4b}-it` | 27B / 12B / 4B | `gemma` — commercial use permitted, no territorial carve-out | ? | ? | ? | **Eligible.** 55 languages claimed; the list is not enumerated on the card, the launch blog or the abstract, and **the model card is gated**, so coverage is unconfirmed. V.5(c) settles it with a login |
| MiLMMT-46-12B v1.0 | `xiaomi-research/MiLMMT-46-12B-v1.0` | 12.19B | `gemma` | ✅ | ❌ | ✅ | Eligible, no Serbian in its 46 |
| LMT-60-8B | `NiuTrans/LMT-60-8B` | 8.19B | **apache-2.0** | ✅ | ❌ | ✅ | Eligible, no Serbian in its 60 (language tags on the card). Self-described Chinese-English-centric |
| Qwen3-30B-A3B | already served | 30B MoE | apache-2.0 | ? | **?** | ? | Baseline and the RFC-125 verification model. Its `sr` coverage was **never checked** — V.5(b) |
| ~~Hunyuan-MT / HY-MT~~ | Tencent | — | Community Licence: "Territory" **excludes the EU**, UK, South Korea | — | — | — | **Excluded**, confirmed |
| ~~NLLB-200~~ | `facebook/nllb-200-3.3B` | 3.3B | **cc-by-nc-4.0** | ✅ | ✅ | ✅ | **Excluded — non-commercial.** The only shortlisted model that covers Serbian |

- **TranslateGemma** released 2026-01-15, gemma3, report arXiv:2601.09012 — which confirms RFC-125's
  provenance note in its own words: "we optimize translation quality using an ensemble of reward
  models, including MetricX-QE and AutoMQM". That is an argument for QE in v2 *and* a caution that QE
  would partly mark its own homework.
- **MiLMMT-46** (arXiv:2608.10812) trained on 143B tokens across 46 languages; claims to beat
  TranslateGemma and HY-MT-1.5. Sizes 1B/4B/12B — **no 27B**.
- **Not verified**: the QE candidates (MetricX-24, COMETKiwi, xCOMET). They are v2 and off the v1
  path.

### 6.2 ASR evidence — Whisper FLEURS WER

**Source**: Whisper paper, Appendix D.2.4, **Table 13 "WER (%) on Fleurs"**, the **`large-v2`** row —
the strongest model in that table.

**Caveats, all load-bearing:**

1. **Not large-v3 or turbo.** `large-v3` appears nowhere in the paper. Our DGX model is
   `faster-whisper-large-v3-turbo-ct2`, generally better per language, so treat this as a
   **conservative prior**. The per-language v3 figures are in the `openai/whisper` repo's
   `language-breakdown.svg`, a figure that has not been transcribed here.
2. **FLEURS is read speech.** Real podcast rates are higher — crosstalk, music, informal register.
3. **Bigger is not monotonically better.** Serbian regressed from `large` 29.2 to `large-v2` 33.9.
4. This list is **complete for every language under 40%**; above that only a representative sample is
   shown. (An earlier version silently dropped French, Arabic, Azerbaijani and Maori and then claimed
   everything unlisted was above 40% — §5.4 C-9.)

**Cluster A — under 5%**
Spanish 3.0 · Italian 4.0 · English 4.2 · Portuguese 4.3 · German 4.5

**Cluster B — 5% to 10%**
Japanese 5.3 · Polish 5.4 · Russian 5.6 · Dutch 6.7 · Indonesian 7.1 · Catalan 7.3 · **French 8.3** ·
Turkish 8.4 · Swedish 8.5 · Ukrainian 8.6 · Malay 8.7 · Norwegian 9.5 · Finnish 9.7

**Cluster C — over 10%**
Vietnamese 10.3 · Thai 11.5 · Slovak 11.7 · **Greek 12.5** · Czech 13.3 · **Croatian 13.4** ·
Danish 13.8 · Tagalog 13.8 · Korean 14.3 · Romanian 14.4 · Bulgarian 14.6 · Chinese 14.7 ·
Arabic 16.0 · Galician 15.4 · Bosnian 15.7 · Macedonian 16.5 · Hungarian 17.0 · Tamil 17.5 ·
Hindi 21.5 · Estonian 21.9 · Urdu 22.6 · Azerbaijani 23.4 · Latvian 23.1 · Slovenian 23.1 ·
Hebrew 27.1 · Lithuanian 28.1 · Persian 32.9 · Welsh 33.0 · **Serbian 33.9** · Afrikaans 36.7 ·
Kannada 37.0 · Kazakh 37.7 · Icelandic 38.2 · Marathi 38.3 · Maori 38.5 · Swahili 39.3 — then
Armenian 44.6 and the remaining low-resource languages. Javanese is `nan` in the source.

### 6.3 The Serbian question — open, not settled

| | Greek | Serbian |
| --- | --- | --- |
| Whisper FLEURS WER (large-v2, Cyrillic for `sr`) | **12.5** | **33.9** |
| In MiLMMT-46 / LMT-60 | ✅ / ✅ | ❌ / ❌ |
| In an eligible MT model | TranslateGemma unconfirmed; two others yes | **unknown** — TranslateGemma unconfirmed, Qwen3 unchecked |
| Covered by NLLB | yes | yes — but non-commercial |

Serbian looks like the weakest link on both axes. An earlier version of this document **demoted it
outright**; that was premature, because three cheap checks were skipped:

1. **The script hypothesis is testable in about an hour.** Croatian sits at 13.4 and is mutually
   intelligible with Serbian; FLEURS `sr` is **Cyrillic** and `hr` is **Latin**. Rescoring FLEURS
   `sr` with `large-v3-turbo`, transliterating hypothesis and reference to Latin, would show directly
   whether the penalty is orthographic rather than linguistic.
2. **Qwen3-30B-A3B was never checked** for Serbian, and it is already served and Apache-2.0.
3. **TranslateGemma's card is gated**, not unknowable — a login settles its 55.

This also reclassifies the script question: "normalize Serbian to Latin or keep Cyrillic?" was filed
as a **display** decision, and on this evidence it is a **capability** decision affecting WER, MT
coverage and CIL identity at once.

**V.5 runs now** (§4, Gate V) and the pilot language is chosen from its result. There is a second
argument for weighing the outcome carefully: Gate V's quality metrics need a native reviewer, and a
pilot language the operator can review directly removes the only human from that critical path.

## 7. Decisions taken

| # | Decision | Rationale | Date |
| --- | --- | --- | --- |
| D-1 | Translate once, to English; analyze English | Every downstream layer stays single-path; prompts, evals and thresholds already exist for English | 2026-09-28 |
| D-2 | Source is canonical, English is derived | Grounding and provenance mean the record is what was said | 2026-09-28 |
| D-3 | **Translation precedes ad detection**; the ad-free base is built on English | `_AD_PATTERNS` is English, so the alternative is an identity ad-free base feeding sponsor reads to GI. Also collapses two translation passes into one, since subtitles need the full timeline anyway | 2026-09-28 |
| D-4 | **Revised.** Analysis reads the English through **one** resolver — and routing every transcript reader through it is *part of the work*, not a property the code already has | The original decision claimed `load_processing_transcript` was already the single resolver and that consumers would not change. That was a docstring, not a fact (§5.4 C-1): the summary stage, the search indexer, `gi/repair` and `gi/load` each resolve independently, and `gi/load` reads the raw transcript. The goal stands; the cheapness does not. Slices S2.1 and S2.2 | revised 2026-09-28 |
| D-5 | The Positions gate is a **read-time** filter | Positions are already a read-time query; this makes the gate retroactive and fail-closed. Corrected in scope: it applies to **five** read paths, not two (§5.4 C-8 and S3.1) | 2026-09-28 |
| D-6 | **QE is cut from v1.** Verification-only | QE's per-language calibration needs a native reviewer to fit thresholds — a human bottleneck — and a translated corpus that does not exist yet. Source-grounded entailment needs no calibration and catches the inversion we fear | 2026-09-28 |
| D-7 | Defer, don't substitute, on translation-model availability | A model swap invalidates the bake-off evidence the language was enabled on | 2026-09-28 |
| D-8 | Speaker labels bypass the translator, and **naming stays before translation** | Identity is CIL's job; a label failing `_looks_like_person` un-attributes a whole turn. Naming after translation would mean a relabel that merges turns and invalidates the unit map (§5.4 C-6) | revised 2026-09-28 |
| D-9 | The language override lives in feed config | The shows library has no backend, and `feeds_spec` already accepts per-entry keys — this is an allowlist addition | 2026-09-28 |
| D-10 | **Phase 0 ships before Gate V**, on its own | It is a correctness fix on the existing corpus, and the bake-off depends on it | 2026-09-28 |
| D-11 | **Revised.** The badge ships in Phase 0 **on a parsed feed tag**; the filter ships once a second language exists | The original rationale — "a badge over a monolingual corpus is a verified fact" — was wrong: without S0.1 the badge would display the run config, not the corpus (§5.4 C-2) | revised 2026-09-28 |
| D-12 | The language filter is its own control, not an option inside the played/downloaded filter | The dimensions are orthogonal; merging them makes "Greek and unplayed" unexpressible | 2026-09-28 |
| D-13 | **Withdrawn.** Serbian is not demoted; the pilot language is chosen by measurement (V.5) | Demoting on a Cyrillic-referenced large-v2 number plus two language lists skipped three checks that cost about a day in total (§6.3) | withdrawn 2026-09-28 |
| D-14 | **Same-language retrieval only. Index both layers; change no embedding model.** Greek words find the Greek episode, English words find the English layer, both land on the same episode. Non-English chunks are keyword-only. | This is the whole requirement, and it is additive: no encoder swap, no dimensionality migration, no corpus-wide reindex, no risk to English search, no dependency on an eval gate that does not exist. Cross-lingual *semantic* matching is a separate capability that belongs with internationalizing the application — see "Explicitly not in this arc" | revised 2026-09-28 |
| D-15 | **Withdrawn.** There is no embedding-model change in this arc, so its blast radius is moot | Superseded by D-14. The findings that motivated it are kept in §5.4 C-7 because they will matter whenever a swap is eventually done | withdrawn 2026-09-28 |
| D-16 | `multilingual_ingest` gates **serving**, not only the pipeline | Otherwise a translated episode becomes visible the moment the pipeline can produce one, before its trust and labelling slices exist | 2026-09-28 |
| D-17 | **Trust markers and the Positions gate ship in Phase 2, before any translated episode is served** | The marker is what the read-time filter keys on, so a translated insight written without it slips through a gate that is only fail-closed when the marker exists. The original plan wrote markers in Phase 3 and served translated episodes in Phase 2 — which would have shipped exactly the failure the PRD calls worst | 2026-09-28 |
| D-18 | Decisions in this table graduate to **ADRs** as they are implemented | The repo's engineering process puts decisions in ADRs; this many decisions living only in an arc note is process drift by its own definition. V.4 is the first | 2026-09-28 |
| D-19 | **Translation runs directly after transcription and diarization, before summary.** Fixed design, not an open question | Summary output feeds GI's topic labels and KG's topics, so translating after it yields English insights hanging off Greek topics — and topics are how episodes connect across the corpus, so identity would fragment by language. Operator decision: load-bearing, accepted, everything shapes around it. `CANONICAL_STAGE_ORDER` gains the slot (S2.2) | 2026-09-28 |

## 8. Open decisions

1. **Is Serbian supportable, and which language is the pilot?** V.5 answers it. Until then no
   language other than English is promised anywhere.
2. **Serbian script is a capability decision, not a display one** (§6.3). Transcribe Cyrillic as-is,
   force Latin output, or romanize post-hoc?
3. **Source-language ad-free variant** — keep one, derived by mapping English ad ranges back? It has
   a reader/player use, no analysis use.
4. **Code-switching** (English passages inside a Serbian episode): pass the unit through
   untranslated, or accept the damage?
5. **Badge scope** — every surface showing a show or episode, or only where metadata chips already
   appear? Blocks S0.6; it is the UXS question.
6. **Ad-survival threshold.** No prior exists for what fraction of sponsor reads survive translation
   into pattern-matchable English. The first language measured sets it.
7. **Greek keyword recall through an English FTS tokenizer.** LanceDB builds one English tokenizer per
   index (stemming, stop-words, accent folding), and Greek is heavily inflected. S2.9 must measure how
   much recall that costs before it is called done; if it is bad, the options are a per-language FTS
   table or accepting it and saying so.
8. **Legal.** Translations are derived works of copyrighted audio, on the same footing as today's
   transcripts but not yet written down; the Gemma Terms carry downstream obligations for outputs
   served to users; and GDPR erasure must enumerate `.en.*`, `translation.json`, verification records
   and the worklist export.
9. **Non-diarized episodes** — how many are in the corpus, and do they get turns at all? Blocks S1.2.

## 9. Running notes

**2026-09-28 — arc opened.** PRD-047 and RFC-123/124/125 landed, reworked against the code rather
than accepted as drafted: the stance-stage dependency does not exist, ad detection is English-keyed
and reorders the pipeline, and `analysis_transcript_ref` looked like it collapsed into an existing
resolver. QE cut from v1 (D-6). Phase 0 reframed as "English as a declared language" (D-10, D-11),
and the demand/bake-off step renamed **Gate V**. Slice plan added. No issues opened; no code written.

**2026-09-28 — evidence pass.** V.1 closed with primary sources (§6): the MT shortlist verified,
Hunyuan's EU carve-out and NLLB's non-commercial licence confirmed, and the Whisper FLEURS table
pulled from the paper. Multilingual retrieval moved into v1 on the operator's call. PRD Appendix A
replaced with the sourced clusters. DGX capacity removed from the documents as a deployment-time
question the operator owns.

**2026-09-28 — adversarial review; three load-bearing claims were false.** Three reviews attacked the
design, the claims and the plan. Outcome recorded in §5.4: **the RSS `<language>` tag is never parsed**
(so the badge, the audit and the normalization step all had no input — new slice S0.1 with a
migration); **`load_processing_transcript` is not the single resolver** (D-4 revised; the summary
stage, search indexer, `gi/repair` and `gi/load` each resolve independently, and `gi/load` reads the
raw transcript today); and **the ad-map cannot invert the ad-free transform** (measured — the span
chain now resolves through segments and `unit_id`). Also corrected: the DGX provider *misreports*
rather than forces English; model selection happens at provider init, not per call; naming cannot move
after translation; D-15's containment was right for the wrong reason; RFC-125's code mechanics were
wrong in three places; Appendix A had dropped French; and two `?` cells had been hardened into
"every" and "none". Plan changes: Serbian demotion **withdrawn** (D-13) with V.5 starting now;
retrieval **split** (D-14) so the risky encoder swap waits for a real eval set; trust markers and the
Positions gate **pulled into Phase 2** (D-17) because the original ordering would have served
unlabelled, ungated translated claims; the flag now gates serving (D-16); and slices were added for
the skip path, the language-ID check, observability, reprocess invalidation, cost measurement, the
test fixture, flag lifecycle and rollback — none of which had one. Re-sliced and renumbered
throughout. Still nothing implemented.

**2026-09-28 — two operator decisions closed, and the arc got smaller.** **D-19**: translation runs
directly after transcription and before summary — accepted as load-bearing design rather than left as
a slice-level detail, and everything shapes around it. **D-14 narrowed**: same-language retrieval
only. Greek words find the Greek episode, English words find the English layer; no matching by meaning
across languages. That removes the embedding swap, the dimensionality migration, the corpus-wide
reindex, the cutover-and-rollback runbook and the prod-eval-set dependency from this arc entirely —
one slice instead of two, M instead of L, and the largest single risk to existing English search is
gone. Cross-lingual semantic search is recorded under "Explicitly not in this arc": it belongs with
internationalizing the application, when a listener can search in whatever language they choose.

<!-- Append new entries above this line, newest last. Keep each to a few lines: what changed, what it
     cost, what it invalidated. Decisions go in §7 with a D-number; facts about the code go in §5;
     claims found false go in §5.4. -->
