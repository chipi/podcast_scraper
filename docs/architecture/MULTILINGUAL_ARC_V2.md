# Multilingual ingest — v2 arc notes (continuation)

The continuation of [MULTILINGUAL_ARC](MULTILINGUAL_ARC.md). That document scopes **v1: a working
pipeline and product, end to end.** This one holds **v2: making it good** — translation quality, the
trust machinery that lets translated claims count, the surfaces that make language visible, and the
retrieval work that needs a different embedding model.

- **Status**: parked. Nothing here is scheduled, and no PRD or RFC has been written for it yet.
- **Opened**: 2026-09-28, when v1 was trimmed
- **Relationship to v1**: strictly downstream. Every item here was in v1's plan and was deliberately
  moved out, with the reason recorded. Nothing was dropped.

---

## 1. Why this document exists

The operator's framing, 2026-09-28:

> *"My focus on version one is a working pipeline and product end to end, and version two would be
> translation quality, fine edges and all of that."*

Applied as a test on every slice: **does the pipeline work end to end without this, and is this about
quality rather than function?** Everything that failed that test moved here.

The point of a separate document rather than a "deferred" table is that a deferred table gets skimmed
and then quietly becomes a graveyard. This is a pile with the reasoning attached, so that when v2 is
scheduled the work starts from a plan rather than from archaeology.

**When v2 is scheduled**, each theme below gets a real PRD and/or RFC. Do not write those now. Where a
design already exists in an RFC, this document points at it instead of copying it, so there is one
source of truth per design.

## 2. What is in v2, and why it left v1

| Theme | Slices | Why it is not v1 |
| --- | --- | --- |
| [Claim verification](#3-claim-verification) | V2-A.1, V2-A.2 | ~~v1 ships with translated claims simply absent from Position surfaces.~~ **Corrected 2026-09-30: v1 ships them PRESENT.** This row described the pre-D-5-withdrawal plan. D-5 was withdrawn 2026-09-28 and **D-37** replaced it — a translated episode has full standing on every surface, exactly as an English one, because a gate with no verifier able to release it just hides output permanently. So v1 believes the translation, and what v2 adds is the machinery to *check* that belief (V2-A.1) and the gate that can then act on it (V2-A.3) — not permission for translated claims to appear |
| [Quality estimation](#4-quality-estimation-and-calibrated-bands) | V2-B.1 | Needs a translated corpus to calibrate against, which does not exist until v1 has run |
| [Language visibility](#5-language-visibility-badge-and-filter) | V2-C.1, V2-C.2 | A badge and a filter over a single-language corpus are dead controls. A language chip was **deliberately deleted** for exactly that reason in #2115 |
| [Turns consumers](#6-turns-consumers) | V2-D.1, V2-D.2, V2-D.3 | Independent English-corpus improvements. v1 needs `turns.json` to *exist*, not to be consumed |
| [Cross-lingual retrieval](#7-cross-lingual-semantic-retrieval) | V2-E.1 | Needs a multilingual embedding model, which re-embeds the whole corpus and risks English search quality |
| [Word-level anchors](#8-word-level-anchors) | V2-F.1 | Per-language aligner checkpoints; the artifact is already forward-compatible |
| [Also deferred](#10-also-deferred-decided-2026-09-28) | V2-C.3, V2-H.1 … H.3 | The model comparison; transliteration and identity aliasing; the language-ID check; the tier-3 prerequisites |

## 3. Claim verification

**Design already written**: `docs/rfc/RFC-125-translation-confidence-and-claim-verification.md` §2
(methods and outcomes), §3 (the read-time gate and its nine surfaces), §4 (surfaces), §5 (monitoring).
Read it there; this section holds the slices and the deferral reasoning.

**What v1 does instead.** v1 writes the translation provenance block on every claim — `translated`,
`source_language`, `unit_ids`, `en_sha256` — and then **believes the translation**. A translated episode
has the same standing as an English one on every surface, with no gate and no user-visible marker
(v1 D-36, D-37). The provenance is recorded precisely so everything in this section can be added later
without reprocessing: verification writes an outcome onto claims that already carry their unit
references, the gate reads that outcome, and the label reads the same block.

**Why the gate came here rather than shipping in v1.** A gate is only meaningful if something can
release what it holds. In v1 nothing verifies, so a gate's only possible behaviour would be "hide every
translated claim, permanently" — paying for translation and GI to produce data nobody can see. That is
not caution, it is waste. The gate belongs with the verifier.

| # | Slice | Goal | Size |
| --- | --- | --- | --- |
| **V2-A.1** | Source-grounded entailment verification | Take an English claim, pull the source-language sentences it was extracted from plus one turn of context each side, and ask a model whether that source actually supports the claim: `supports` / `contradicts` / `insufficient`, constrained to JSON, with the record written to the node. Includes the cross-model re-translation fallback on `insufficient`, and that fallback triggering the invalidation path v1 builds. Batched per episode. | L |
| **V2-A.2** | Operator review worklist | A JSONL export plus a minimal operator view of contradicted and unverified claims — source and translation side by side, audio timestamps, and verify / reject / re-translate actions with operator provenance recorded. Empty until V2-A.1 produces flags, which is why it moved with it. | M |
| **V2-A.3** | The read-time Positions gate | Withhold a translated claim from position-bearing surfaces until it is `verified`. **This moved out of v1 with the verifier** (v1 D-5 withdrawn, D-37): a gate with nothing able to release it just hides output permanently, so it belongs here. Two predicates, not one — an **edge** predicate for what gets verified (SPOKEN_BY-supported ∩ `ABOUT` ∩ `insight_type == "claim"`, and note that type filter is a caller-overridable default) and a **property** predicate for what renders — plus a test that the first is a superset of the second, or translated non-claim insights would be excluded permanently rather than pending. Applies to **nine** surfaces: `position_arc`, `topic_conversation_arc` (via `topic_timeline`), `topic_timeline_merged`, `person_profile`'s insights-by-topic, `topic_perspective_leaders`, `topic_perspectives` (which feeds the consumer app *and* OG share images), `app_gi_view`'s episode stance display, `search/relational_queries.positions_of` and neighbours (which read `CorpusGraph`, not `gi.json`), and `enrichment/enrichers/topic_consensus.py` — write-time, but it loads `gi.json` itself, so the same property predicate applies at its run time. Whether the gate is fail-closed everywhere or split between surfaces that *assert* a position and those that merely *describe* is a product judgement to make then, not now. | L |
| **V2-A.4** | Label translated content wherever it renders | The "Translated from <Language>" marker, deferred from v1 (D-36) so the listener sees one simple thing. Not one chip: the marker has to survive every boundary that drops node properties — `AppInsight` / `AppQuote`, the MCP `InsightSummary` / `SupportingQuote` contracts, `hybrid_search._to_search_result` (Lance rows carry no `translated` column, so a translated quote is served as verbatim speech in search, digest and trending), `server/og/build.py` share cards, and **snapshots**, since favourites and captures are copies that a read-time change can never reach. Matters most for **quotes**, which can be passed on as somebody's words — that is why it is the first thing in this section after verification. | L |

**Sequencing note.** V2-A.1 is the single highest-value item in v2, because it is what turns a Greek
episode from "readable" into "counts towards the corpus's primary signal". If v2 is scheduled in
pieces, this goes first.

## 4. Quality estimation and calibrated bands

**Design already written**: `docs/rfc/RFC-125` §7, which exists specifically as the v2 seam and
explains what QE adds, what it costs, and how it attaches without reshaping anything v1 writes.

**Why it is not v1**, in the operator's words: *"I would not put QE back in version one, I would keep
it in version two."* The original reasoning had two legs and one has since dissolved:

- **Still true**: calibrating a band means fitting thresholds against labelled units, and there is no
  translated corpus to sample until v1 has run. QE cannot precede the thing it grades.
- **No longer true**: the calibration labels needed a native speaker. With the LLM-as-judge approach
  (§9) they do not. So QE is deferred on sequencing alone, not on a human bottleneck.

| # | Slice | Goal | Size |
| --- | --- | --- | --- |
| **V2-B.1** | Per-unit QE with per-language calibrated bands | Score every `(src_text, en_text)` pair with a reference-free QE model; map scores to green / amber / red per (language, translation model revision, QE model revision); propagate the **minimum** band over the units a claim cites; use `band = red` as a verification trigger for the non-position claims v1 leaves unchecked; show a confidence marker for amber and red only. Requires licence verification for the QE checkpoint — MetricX-24 and the Unbabel COMETKiwi / xCOMET family are the candidates and several carry non-commercial terms. | L |

**A caution to carry forward.** TranslateGemma's own technical report states it was optimised with
"an ensemble of reward models, including MetricX-QE and AutoMQM". If the translation model and the QE
model come from the same lineage, QE is partly marking its own homework. Pick a QE model from a
different family, or treat agreement between them as weak evidence.

## 5. Language visibility — badge and filter

**Requirements already written**: PRD-047 FR7. FR7.1 (the API field) is v1; FR7.2–FR7.5 are here.

**Why it is not v1, and this one has history.** A language chip on the show page existed and was
removed in **#2115 (2026-09-18)**, with the reason recorded in `PodcastView.test.ts:229-233`:

> *"The language badge is deliberately gone with it: every show in the corpus is English, so it was a
> constant that cost a wrap in a 144px column."*

That reasoning is correct while the corpus is monolingual and stops applying the moment it is not.
Reinstating the badge in v1 would re-add precisely what was deleted, to display a value that — until
v1's parsing work lands — is the run configuration rather than the feed's own tag. So badge and filter
ship together, when there is a second language to distinguish. The operator confirmed the reversal is
intended: *"yes, we reverse what we did some time ago because we were only in one language."*

**The component itself is built in v1**, because the transcript language control needs it (v1 D-25,
D-26). What is deferred is *placement as metadata decoration* — on episode rows, tiles, cards, show rows
and the shows library. So V2-C.1 is a rendering slice, not a component slice.

| # | Slice | Goal | Size |
| --- | --- | --- | --- |
| **V2-C.1** | `LanguageBadge` across both apps | A compact squared chip with the uppercase code, on `EpisodeRow`, `EpisodeTile`, `EpisodeCard`, `ShowRow`, `ShowTile`, `PodcastView` and the operator shows library. Includes `t()` strings, a language display-name source so the aria-label reads "Greek" not "EL", a contrast check, Playwright specs in both apps, the `E2E_SURFACE_MAP.md` updates, and a UXS — the existing UXS-011 / 012 / 015 have no language content. Omits the badge rather than guessing when language is unknown. Note `ShowRow` already carries a documented spec gap. | L |
| **V2-C.2** | Filter shows and episodes by language | A language control in the consumer episode toolbar (single-select via `ListToolbar` today), on show browse, and in the operator library filter bar, reusing `TypeFilterBar`. Its **own** control, not an option inside the played/downloaded filter, so "Greek and unplayed" stays expressible. Renders only when the corpus holds more than one language. | M |

## 6. Turns consumers

**Design already written**: `docs/rfc/RFC-123-speaker-turns-artifact.md` §4 (consumer migration) and
its Rollout steps 3–5.

**Why it is not v1.** v1 needs `turns.json` to exist, because translation units are defined as
sentence groups inside a turn. It does not need anything to *read* turns. All three consumers are
independent English-corpus improvements with standalone value — which also means **any of them can be
pulled forward** if English quality becomes the priority, without touching the multilingual work.

| # | Slice | Goal | Size |
| --- | --- | --- | --- |
| **V2-D.1** | GI speaker attribution by turn lookup | Replace regex-over-transcript-text with a char-span binary search into turns; keep regex as the fallback when `turns.json` is absent. Gated on replaying the 7,101-quote #2062 set and reporting the full transition matrix — zero name→different-name transitions required to flip the default. | M |
| **V2-D.2** | Turn-bounded search chunking | Chunk within turn boundaries so a window cannot span a host's question and a guest's answer; `SegmentDocument` and the LanceDB segment schema gain `speaker_ids` and `turn_ids`; retrieval eval reporting recall@k and speaker-precision. Carries a schema bump and therefore a reindex — **coordinate with any other schema change** so one rebuild covers both. | L |
| **V2-D.3** | Player sentence-granularity cues | Optional `?granularity=sentence` on the segments contract, serving sentence cues from `turns.json` so subtitles and tap-a-quote use complete sentences instead of raw 5–15 s Whisper fragments. Default stays raw segments. | S |

## 7. Cross-lingual semantic retrieval

**Analysis already written**: RFC-124 §6.2 explains what v1 does and why it stops there.

**What v1 delivers.** Both layers are indexed — the source-language transcript and the English
analysis transcript — each chunk tagged with its language. A Greek query reaches the Greek chunks
lexically; an English query reaches the English chunks. Both land on the same episode. Crucially, a
Greek episode **does** participate in semantic search, through its English translation layer.

**The one hole v1 leaves**: a Greek query that needs *meaning* rather than words. Searching Greek for
"inflation" finds passages containing that word, not a passage discussing rising prices without it.
Greek keyword recall is also somewhat weakened by the index's English tokenizer (stemming, stop-words,
accent folding).

**Why closing it is v2.** It requires a multilingual embedding model, and that means: re-embedding all
existing English episodes; a vector-dimensionality change, which is a stored-schema change, so the
index goes stale and search answers "no index" until a full rebuild completes; and a real risk that
**English retrieval gets worse**, because a multilingual model of a given size is typically weaker on
English-only tasks than an English specialist. For a corpus that is overwhelmingly English, that is a
large global downside for a small local upside — measured by an eval of 25 queries on a synthetic
fixture with no CI gate, which this repo's own rules say is not evidence. The swap also ripples into
`search/insight_clusters.json`, `kg/topic_clustering`, `search/query_router` and
`search/quality_metrics.py`'s hardcoded `_ZERO_VECTOR_DIM = 384`.

| # | Slice | Goal | Size |
| --- | --- | --- | --- |
| **V2-E.1** | Multilingual embeddings and cross-lingual search | Swap `vector_embedding_model` for a multilingual encoder; re-enable non-English chunks on the dense stage; bring along the artifacts and constants built on the hardcoded MiniLM default; dimensionality migration with a reader-support bump; full reindex with build-then-flip cutover and a rollback snapshot. **Gated on a prod-derived eval set built from `search/query_log.py`**, so an English regression is detectable. | L |

**When this becomes valuable**, in the operator's framing: *"that would be super useful later once we
internationalize the application and then people can search in any language they want, but we're still
not there."* So the trigger is the application being internationalized, not a date. If a full reindex is
being run for some other reason at that point, fold this into it rather than scheduling a second outage.

## 8. Word-level anchors

| # | Slice | Goal | Size |
| --- | --- | --- | --- |
| **V2-F.1** | Forced alignment for word-level timing | Sentences gain `words: [{char_start, char_end, start_ms, end_ms}]` and `timing` becomes `word_aligned`. Needs per-language wav2vec2 aligner checkpoints, pinned per ADR-155. `turns.json` is already forward-compatible — additive fields only, no schema break. | L |

## 9. The LLM-as-judge substitution (applies to both v1 and v2)

Recorded here because it is the mechanism several v2 items depend on, and because it replaced a
dependency that had no owner.

**What changed.** v1's quality gate originally required a **native-speaker reviewer** per candidate
language to judge whether translations preserved meaning. That reviewer was never identified, had no
protocol, and gated the entire arc — Phases 2 through 4 all sat behind it. The operator's decision,
2026-09-28: *"I'm not going to talk to people… use LLM as a judge, some good one, I'm fine with the
public one."*

**Protocol, wherever a judge is used.** Per unit the judge receives the source text, the English text,
and one turn of source context each side, and returns structured JSON: `meaning_preserving` ∈
{yes, minor, no}, `error_type` ∈ {negation, hedge, name, omission, register, other}, and a one-line
rationale. Temperature 0. Model id, version and prompt version recorded on every judgement, and the
judgements committed as an eval artifact so a gate decision stays auditable.

**Three rules that make it trustworthy:**

1. **Different family from the translator.** TranslateGemma was trained with MetricX-QE and AutoMQM as
   reward models, so a same-lineage judge may share its blind spots.
2. **Blind and shuffled.** The judge never sees which candidate produced which translation.
3. **Validated by fault injection, not by assertion.** LLM judges are biased toward fluency — and the
   failure mode that matters here *is* fluent, because a dropped "not" produces perfectly natural
   English. So before a judge's verdicts count for anything, inject the faults we actually fear into
   known-good translations — delete a negation, flip a hedge to a certainty, drop a clause, swap a
   name, invert a stance — and measure the per-fault-type detection rate. That number is the judge's
   own gate, it needs no human, and it targets the real risk far better than asking someone whether 60
   sentences read well.

For gate-critical calls, run two judges from different families and treat **disagreement** as a flag
rather than averaging it — the cheap substitute for inter-rater reliability.

**What this does not solve.** The ASR quality check requires hearing the audio; a text judge cannot do
it. Options are an audio-capable model, or cross-ASR agreement (transcribe with two models, treat
divergence as a proxy) plus the FLEURS prior. Cross-ASR agreement is fully automatable and weaker than
a native ear on real podcast audio — say so rather than implying parity.

**Honest cost.** A native speaker catches sarcasm, register, dialect and culturally-loaded meaning a
judge may miss. This is a trade: a large signal of *measurable* quality in place of a small signal of
*assumed* quality that was never actually obtainable.

**Scope boundary.** Use the public judge for **evaluation and calibration** — bounded episodes,
auditable, content exposure limited to a handful of episodes. Keep any **runtime** per-claim
verification on the DGX, where it is private and pinnable, rather than sending every claim from every
translated episode to a public API.

## 10. Also deferred, decided 2026-09-28

| Item | Slice | Why it is not v1 |
| --- | --- | --- |
| **The model comparison** | **V2-H.1** | v1 picks one model for breadth across tier 1 and checks that it runs and produces sane output. The five-candidate × six-measurement comparison with calibrated thresholds is exactly "make translations better" — and D-7 means a model swap invalidates the earlier evidence anyway, so v2 re-measures from scratch regardless (v1 D-27) |
| **Transliteration and identity aliasing** | **V2-C.3** | Not built at all in v1, because every tier-1 language is Latin script and a person's name is usually the identical string across them. The trigger is enabling a **non-Latin-script** language — tier 2 (Cyrillic) or tier 3 (CJK, Arabic). Then: canonical-name lookup, deterministic transliteration, and source-script forms written as aliases so one person does not become two (v1 D-24, D-29) |
| **Language-ID sanity check** | **V2-H.2** | A wrong feed tag is caught by the corpus audit and corrected with the per-feed override, which v1 builds |
| **Tier-3 prerequisites** | **V2-H.3** | Two v1 assumptions break for Japanese, Korean and Chinese and must be fixed before any of them is enabled: translation units are packed by **word count**, which is meaningless without spaces; and the speaker-name guard requires ≥2 tokens, so a single-token CJK name would silently un-attribute every quote in its turn |

## 11. Running notes

**2026-09-28 — document created.** v1 was trimmed against the operator's "v1 works, v2 works better"
test. Moved here with his confirmation: QE; source verification and the operator worklist; the badge
and the language filter; all three turns consumers; cross-lingual semantic retrieval. Already-parked
items folded in: word-level anchors. The LLM-as-judge substitution
(§9) is recorded here because it removed the arc's only human dependency and several v2 items rest on
it. Three further trim candidates are parked in §10 awaiting a decision. Nothing here is scheduled and
no PRD or RFC has been written for it.

**2026-09-28 — the v1 trim finished.** §10 added: the model comparison, transliteration and identity
aliasing (triggered by the first non-Latin-script language), the language-ID check, and the two tier-3
prerequisites — word-based unit packing and the ≥2-token name guard, both of which break for CJK. v1's
tier 1 is entirely Latin script, which is why the aliasing work has no v1 consumer at all.

**2026-10-03 — the parallel-translation grid exists, and it already produced four measurements
V.3b (#2251) can use.** The ten non-English `e02`/`e03` fixtures were rewritten as translations of
`p01_e02` / `p01_e03` and then run through `fixture_validation_dgx` on the DGX: 10 episodes, 430
units, **0 failures**, `google/translategemma-12b-it` at temperature 0, prompt sha
`b0cabece…` — the SAME model and prompt as the five committed `e01` captures, so all fifteen are
directly comparable. 28 minutes wall for the ten including summary/GI/KG; translation is ~2.5-3
min of each episode. No ASR (`transcribe_missing: false`; the `.vtt` twin is the source).

WHAT THE GRID IS. For each of three conversations, `p01` holds the authored English ORIGINAL and
five feeds hold it translated into es/it/fr/de/pt — and the pipeline then translates each back to
English. So every machine-English render has a known-correct reference sitting beside it, and the
five renders of one conversation are mutually comparable. Unit packing came out at **43 units for
all ten episodes**, so comparison is 1:1 at unit level, not just document level.

1. ROUND-TRIP VARIANCE IS DRIVEN BY IDIOM, NOT BY LANGUAGE. `e03`'s title is descriptive and four
   of five languages converge on it ("The Mechanics of a Quiet and Fast Bike/Bicycle", `de` exact).
   `e02`'s contains an idiom — "Without the Hype" — and the five diverge sharply:

       source (p01_e02): Enduro Skills Without the Hype
       es  Enduro Skills: No Hype, Just Performance.   <- ADDS content absent from the source
       it  Enduro Technique: Quiet Excellence.         <- reframes
       fr  Enduro Riding Without the Hype              <- near-exact
       de  Enduro Technology, No Hype.                 <- MISTRANSLATION: Technik -> Technology
       pt  Enduro Technique: No Exaggerations          <- literal rendering of the idiom

   `de` is simultaneously the worst (`e02`) and the best (`e03`, exact), which is the caution
   against ranking a model or a language on one episode.

2. THE de MISTRANSLATION IS ONLY VISIBLE BECAUSE THE ORIGINAL EXISTS. "Enduro Technology" reads
   as a plausible title; it is wrong because the conversation is about riding technique, which is
   what `p01_e02` says. This is the argument for the parallel-original design over
   back-translation scoring alone.

3. TRANSLATION VARIANCE MOVES GI COUNTS BY ~20% ON IDENTICAL SOURCE CONTENT. Same conversation,
   same GI stage, same model — only the English render differs:

       e02   es 36   it 37   fr 44   de 39   pt 44
       e03   es 36   it 42   fr 37   de 42   pt 40

   36-44 insights for one conversation. Downstream counts are therefore not a stable measure of an
   episode unless the translation is held fixed — worth knowing before any GI metric is compared
   across languages.

4. AD EXCISION ON THE ENGLISH RENDER DIVERGES 2x FOR de/pt ON ONE EPISODE. Share of the canonical
   English body removed to build the ad-free base:

       e03 (one native ad)        es  8.4%   it  8.8%   fr  8.3%   de  8.5%   pt  8.5%
       e02 (four disclosed reads) es 11.0%   it 11.3%   fr 11.0%   de 22.7%   pt 23.1%

   `e03` is tight across all five. On `e02` German and Portuguese remove roughly twice what the
   others do — about 730 extra characters, a turn or two of real conversation. NOTE THE MECHANISM
   IS NOT THE OBVIOUS ONE: excision runs on the ENGLISH body with ENGLISH patterns, so this is the
   translation's word choice changing what the English ad patterns match, NOT a gap in the de/pt ad
   vocabulary (#2258/#2259 are about detecting ads in the SOURCE language and are not implicated).
   Unverified: which spans are being cut. The text diff is the first thing to look at.

DETERMINISM, as far as one observation goes: `p10_e03`'s title came out identical across two
independent runs at temperature 0. Not a determinism claim — one episode, one field.

WHAT THIS DOES NOT MEASURE. Nothing here scores translation QUALITY — there is no judge, no
reference-based metric, no human rating. These are artifacts of a working pipeline, useful as the
instrument V.3b points at. The five `e01` captures are from 2026-10-01 and the ten from
2026-10-03 on the same model and prompt, so the grid is internally consistent; a model change
invalidates all fifteen together, which is the point.

### 2026-10-03 — the naming vocabulary is per-language, and what that did NOT fix

WHAT MOVED. Every English-only naming collection we knew about is now a per-language map in
`src/podcast_scraper/speaker_detectors/naming_vocabulary.py` (43 maps, six rows each), read through
`hosts._NAMING_VOCABULARY_MAPS` (56 entries) and advertised by the fail-closed intersection
`NAMING_VOCABULARY_LANGUAGES`. Four modules were converted: `hosts.py`, `roster.py`,
`resolution.py`, `gi/speakers.py`. English is unchanged — 18 of 23 converted patterns in `hosts.py`
rebuild byte-identically, and the five that differ do so only by `(?:x)` around a single literal or
a positional group becoming a named one, proven equivalent on 5,296 matches (full match plus every
capture group) over 592 KB of fixture and test text.

THREE DEFECTS THE CONVERSION FOUND. All three were live, none failed a test, and each is a
different shape of the same mistake — a hand-copy that looks complete:

1. `CUE_FIRST_BODY` had eight English alternatives; the extracted row had six. The two lost were
   the progressive ("I'm speaking with X") and the inverted ("with us today is X") — whole
   introduction shapes, not variants.
2. `_NONPERSON_AUTHOR_MARKERS` had 45 English markers; the extracted row had 36. The nine lost
   included `committees?`, `councils?`, `foundations?` and `plus` — i.e. three of the subsets whose
   comments record the measurement that put them there (4,307 roster entries / 13,642 Person nodes
   for the institution tokens; 17,949 person names for `plus`).
3. The German `ARTICLE_BEFORE` row gained `von` by translation. `von` IS German's genitive — and it
   is also German's by-agent marker, the word every German host statement puts directly in front of
   the host. The guard built to stop "…Council of the Americas Online" (#2075) therefore ate the
   host of every German feed. Measured: four languages named their host, German named nobody.

Two more were mine, caught by A/B rather than by reasoning: `roster._NAME_SUFFIXES` is
GENERATIONAL (`jr`, `sr`, roman numerals incl. `v`) and is NOT `hosts.NAME_SUFFIXES`, which is
CREDENTIALS (`md`, `phd`, `esq`) and has no `v`; and `_INTRO_AFFILIATION_TOKENS` lost 13 of its 30
tokens, specifically the two-letter ones (`by`, `we`, `me`, `us`, `go`, `he`, `do`, `if`) that exist
so a short real surname (Ng, Wu, Li, Xu) is not read as a function word. Both reverted to the live
sets.

THE PAYOFF, as a measurement. Recall, `hosts_from_feed_statement` on a description that states its
host in the first sentence — before, every non-English row returned the empty set:

| | before | after |
| --- | --- | --- |
| es | `[]` | `Lucía Herrera` |
| it | `[]` | `Chiara Ricci` |
| fr | `[]` | `Élodie Chevalier` |
| de | `[]` | `Lena Hofmann` |
| pt | `[]` | `Inês Carvalho` |
| en | `Casey Rowe` | `Casey Rowe` |

Precision, `is_publishable_speaker_name` — a clean diagonal, each language refusing its own role
words and only its own. `Anfitrión Miguel` was publishable before and is refused under `es` now,
while `Host Mike` was already refused under `en`; `Lucía Herrera` survives in all six.

ONE DELIBERATE WIDENING, recorded because it is NOT a no-op. Three sites held three different
hand-written copies of the self-introduction cue — `roster.py`'s match-form detector
(`i'm|i am|my name is`), `roster.py`'s sign-off reader (`I['’]?m` alone) and `resolution.py`'s
window anchor — while `roster.py`'s own comment claims they "match the SAME cue vocabulary
(imported, so they cannot drift)". They had drifted and were not imported. All four sites now read
`SELF_INTRO_WORDS`, which adds `my name's` everywhere and `i['’]?m` (which also admits the
apostrophe-stripped "im" a folded transcript produces). This widens English recall on three paths
and is unmeasured.

WHAT THIS DOES NOT DO — and this is the larger half.

- ~~`roster.py`, `resolution.py` and `gi/speakers.py` do not take a `language` parameter~~
  **Threaded, same day.** The SOURCE language (`transcription_language(cfg)`, the one reader)
  reaches `resolve_speaker_roster`, `build_speaker_diagnostics` and the resolution prompt from
  `diarization/pipeline.py`; `None` reads the analysis row (unchanged behaviour), and a language
  that resolved but has no row reads NOTHING rather than English
  (`naming_vocabulary.vocabulary_row`). `gi/speakers.py` takes one but is deliberately still
  called without it — it reads the post-D-44 English body; see its `_is_publisher_label`
  docstring for what that does not cover.
- **The publish gate is asked in the feed's language everywhere in the pipeline.**
  `is_publishable_speaker_name` checks English UNION the language's row (a reject filter, so a
  union can only refuse more). Every call site passes `language=`, enforced by a source scan in
  `test_the_feed_language_reaches_the_roster.py`; the per-episode host pool gets the feed language
  via a new `HostDetectionResult.language`. Two callers are exempt BY NAME in that test: main's
  one-shot cleanups m0015 and m0017, dry-run on the all-English prod corpus against frozen sets.
- ~~The name-shape character classes are still ASCII~~ **Widened, and NOT measured.**
  `_HOST_SELF_INTRO`, `_HOST_BRANDED_INTRO`, `_HOST_WITH_ME_INTRO`, `_EPISODE_HOST_CUE`,
  `roster._THIS_IS_INTRO` and `resolution._SPOKEN_FULL_NAME` now spell the capital with
  `_STATED_UC`. `roster._NAME_WINDOW_MF` stays ASCII on purpose: the match form strips accents, so
  there is nothing for a wider class to match. Each new class admits every character the old one
  did, but that does NOT make the output a superset: a wider class can move where a match starts
  or ends and so change what is captured (`_SPOKEN_FULL_NAME`'s lookbehind changed too). English
  naming can therefore move in either direction, and that change is unmeasured — the naming gate (`NAMING_GATE_RUNBOOK.md`) runs on the box
  against `.test_outputs/naming_gold/`, which is not on the workstation.
- **The five non-English rows are AUTHORED, not measured.** The English rows are gold-gated against
  #2269's development set; nothing gates the others, and there is no non-English corpus to gate them
  against. #2255–#2259 remain for vocabulary that measurement discovers — these maps are the
  starting point for that, not a substitute for it.

<!-- Append new entries above this line. When v2 is scheduled, each §3-§8 theme gets its own PRD/RFC;
     until then this document is the plan of record for everything deferred out of v1. -->
