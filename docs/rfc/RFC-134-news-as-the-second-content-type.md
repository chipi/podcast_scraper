# RFC-134: News as the second content type

- **Status**: Draft, revised 2026-11-11 after the ingestion and advisor reviews
- **Releases**:
  - **The core release** builds the seams in §1 and the schema half of ontology v3 in §6, and
    proves them with a test-only second type.
  - **News 1.0**, a release of its own after it, builds everything else here.
  - Grounded insights for news, corrected articles as versions and summaries across news are
    News 2.0, after the corpus moves to the database.
- **Authors**: Marko Dragoljevic
- **Stakeholders**: the operator; the ingestion, database and player sessions
- **Related ADRs**:
  - [ADR-168](../adr/ADR-168-installed-extensions.md) (installed extensions)
  - [ADR-170](../adr/ADR-170-private-repo-split-platform-common-apps.md) (the private repo split)
  - [ADR-176](../adr/ADR-176-extension-seams-for-the-split.md) (the seams built for release 1)
  - ADR-177 to ADR-181, proposed below
- **Related RFCs**:
  - [RFC-131](RFC-131-person-identity-one-home.md) (person identity, one home)
- **Related Documents**:
  - `docs/architecture/gi/gi.schema.json`, `docs/architecture/kg/kg.schema.json`
  - `docs/architecture/gi/ontology.md`, `docs/architecture/kg/ontology.md`

## Abstract

- **What:** news articles become the second content type, beside podcast episodes.
  - They arrive over two transports, RSS/Atom feeds and a JSON GET API.
  - They are fetched and reduced to their text, then summarised and given a knowledge graph (KG),
    as podcasts are.
  - They land in the same corpus and graph, and in search, in tables of their own.
- **Why:** the platform is meant to hold more than podcasts. A second type, over a second
  transport, with no audio, is the proof that the core is general, and that the seams and the
  vocabulary refactor did what they claim. It is also the first real-world material that extends
  the podcasts' storylines.
- **How:** content-type and transport seams on the extension interface (ADR-168), a text-in entry
  point beside the podcast pipeline, the medium-neutral half of ontology v3 (R3a), and the existing
  enrichment over article text.

## Problem statement

Every path in the pipeline today assumes a podcast:

- an episode from an RSS enclosure;
- audio, then a transcript;
- insights and quotes anchored to a transcript by character and time.

The GI schema (3.1) makes `episode_id`, `timestamp_start_ms`, `timestamp_end_ms` and
`transcript_ref` required on every quote. A news article has none of them.

Release 1 put seams around the stages that leave public (ADR-176): stage, read and CLI hooks. It did
not add the other R2 seams:

- content types and transports on `Extension`;
- a text-in entry point beside `run_pipeline`.

So there is no way yet to add a type without editing the podcast path.

**Use cases:**

1. **A listener's storyline gets real-world context.** A storyline such as inflation expectations or
   AI regulation shows recent news items about it, each with a summary and a link out.
2. **One graph across media.** A person who speaks on a podcast and is quoted in an article is one
   node.
3. **Search across types.** One query finds podcast moments and news items together.

## Goals

1. **A second content type end to end:** fetch, text, summary and KG, corpus, graph, search,
   viewer. No podcast-only module runs for a news item.
2. **A second transport:** a JSON GET API beside RSS/Atom.
3. **Podcasts unaffected:**
   - podcast artifacts are identical with news sources configured;
   - every podcast surface (the catalog, episode routes, search, the Player, the MCP tools) sees news
     only when it asks for it.
4. **Public alone still works:** a public-only install ingests news with the plain defaults:
   summary and basic KG.
5. **The data model practised before the database:** news writes ontology v3 in the file corpus.
   The corpus database (E06) is then designed against two real types, once.

## Non-goals

- **Rendering an article:** surfaces show the title, the summary and basic item metadata, always
  with a link out (§10).
- **Grounded insights (GI) for news:** News 2.0, after an offline evaluation of quality and cost.
  News 1.0 cards therefore show no quoted evidence.
- **Corrected articles as versions:** News 2.0. News 1.0 records what versions will need (§2).
- **Summarising across news items:** News 2.0.
- **Event clusters, GDELT, divergence, publisher grading:** later (structured intelligence).
- **Other content types** (social posts, advisories, reports): after the corpus database.
- **Migrating the podcast corpus to v3:** R3b, its own release (ADR-178).
- **News on the listener app by default:** at most a flagged proof (§11).
- **An audio-in entry point for a non-podcast type** (earnings calls, for example): a later seam
  (§13).

## Constraints and assumptions

**Constraints:**

- **News is a public capability (the operator, 2026-11-11).** These live in public and run on a
  public-only install:
  - the type, both transports and extraction;
  - the summary, KG, search rows and the public viewer card.

  The private extension adds to news only what it adds to podcasts, where it has value.
- **No change to podcast output, routes or response models.** Shared reads (the catalog builders,
  the search query paths) take a type filter that defaults to podcast items. A surface sees news
  only when it asks. This lands in the core release, before any news exists, so the private repos
  need no change to stay as they are.
- **Public names no private package (ADR-170).** Grounded insights, identity joining and the private
  viewer are referred to here as "the private extension".
- **The full text is never served** (§10 states the measurable rule). It is extracted and stored
  only so that it can be enriched.
- **The corpus is still files** in this release, under the existing change method.
- **No freeze is needed for news itself:** it adds files, rows and search tables of its own, and
  changes nothing that exists.

**Assumptions:**

- **Two or three articles a day per source** on average. That is about 1,000 items a month for 19
  sources, against today's 3,151 episodes. Not true without a cap for Hacker News or BBC World, so
  each source has a per-run cap (§12). To measure on the first month.
- **Enrichment cost per article is well below an episode's:** no transcription and shorter text.
  Not measured yet.

## Design

### 1. Content-type and transport seams (R2, completing ADR-176; built in the core release)

The core release builds this section and proves it with a test-only second type: a plain text type
over one RSS/Atom transport, in fixtures. News 1.0 then only registers its type and transports.

- **New `Extension` fields:**
  - `content_types`: a name, an item id scheme, a stage profile and a viewer card kind;
  - `transports`: a name and a function that lists new items from a source definition.
- **Podcast is registered as the first, built-in type,** with RSS enclosures as its transport. Its
  path does not change.
- **A text-in entry point beside `run_pipeline`:** it takes items that are already text, with their
  metadata, and runs the type's stage profile.
- **Stage profiles name the stages a type runs.** News skips audio, transcription, diarization and
  ad cleaning, and runs:
  - summary;
  - KG;
  - identity joining, when the private extension is installed (§8).
  - Not GI, in News 1.0.
- **One item context in the processing job:** type, profile, item id and body path. The profile
  decides which stage hooks run; hooks gain no type argument (ADR-177 decision 7).

### 2. The `news_article` type

| Concept | Podcast today | News |
| --- | --- | --- |
| Source | show (feed) | publication: one feed or one API query |
| Item | episode | article |
| Body | transcript | extracted article text |
| Publisher | the show's organisation | the publication's organisation (T-12) |
| People | hosts, guests, speakers | bylines (authors), people mentioned |

- **Ids** (decision T-8):
  - existing ids never change;
  - new types use `source:` and `item:` ids;
  - an item id comes from the item's own stable identity, which each type names:
    - for news, the normalised canonical URL (stated and tested rules: scheme, host case, trailing
      slash, tracking parameters, AMP variants), so one article over two transports is one item;
    - for a filing later, the accession number;
  - the transport's id (an RSS `guid`, an HN `objectID`, a Guardian `id`) is kept as a field;
  - never from a feed position or a fetch time;
  - a source whose pages hold several articles under one URL, such as live blogs, is flagged.
- **Item metadata:**
  - title, canonical URL, publisher, byline as given, published time;
  - `editorial_type` (news, opinion, analysis, press release, speech);
  - language;
  - rights fields: licence, paywalled, serveable (what a surface may show), mineable (`false`
    means the body is not stored and the item is not enriched);
  - the body's hash and the fetch time.
- **Corrections:** a corrected or withdrawn article is not re-ingested in News 1.0. Resume skips a
  seen id. The hash and fetch time are recorded, so News 2.0 can start versions (DECISIONS rule 4)
  from evidence.

### 3. Transports

- **RSS/Atom:**
  - feeds read with `feedparser`;
  - the item list comes from the feed and the text from the article page.
- **JSON GET API:** two, to prove the transport is not one source's shape.
  - **Hacker News (Algolia search API):** no key. Each item links to a third-party page, which is
    fetched and extracted like a feed item.
  - **The Guardian Content API:** needs a developer key. It returns the body text and byline
    itself, so it also proves a transport that needs no page fetch.
- **Fetching manners:**
  - a polite user agent;
  - conditional GET (ETag / Last-Modified);
  - a per-host delay;
  - the robots.txt rules honoured;
  - no paywall circumvention.
- **The interface is not shaped by news alone.** It must also carry what a later public transport
  needs, such as SEC EDGAR for filings:
  - required request headers, such as a contact in the User-Agent;
  - a per-host request budget set by the transport, shared by every source on that host (EDGAR is
    one host for every company);
  - a cursor that is the transport's own: a feed position, an API page, or a filing date and
    accession number.

### 4. Text extraction

- **The library:** `trafilatura` gives the body, byline and date.
- **The fallback:** when trafilatura's text is shorter than a stated share of `readability-lxml`'s
  on the same page, the readability body is used. The share is set from the per-source extraction
  goldens. Trafilatura cut one of 19 sampled articles to 1,611 of about 5,500 characters.
- **Bylines:**
  - the feed's author field is used first, the page's second, and publisher names are removed;
  - parsing (several authors in one string, surname first, a date stuck to a name) is measured
    against a golden set before any byline joins a person.
- **Too little text** (a teaser or a video page): the item is kept with its title and feed summary
  and marked `text_status: teaser`. It is not enriched as if it were whole.
- **Evidence:** ADR-179, with the sample measurements.

### 5. The corpus, in file mode

- **Layout:** news items live in the corpus beside podcast episodes, in the same per-source run
  layout. The catalog, the corpus migrations and the index builders then see them through code
  paths that already exist.
- **Fields:** each news item's metadata carries `content_type: news_article`. Podcast writers do
  not write the field until R3b; an absent field reads as `podcast_episode`. This keeps podcast
  artifacts byte-identical.
  - Fields every type shares (title, times, publisher, rights) sit at the top level.
  - A type's own fields sit under a key named for the type (`news_article: {…}`).
  - The corpus database's importer later maps that key to the type module's table, or to one
    column read whole. There is never a column per type on the shared item table.
- **Readers:** any reader that assumes an episode reads `content_type` first, or is limited to
  podcasts by the type registry. The list of such readers is part of this RFC's work.
- **The extracted text:** stored with the item as its body artifact, the way a transcript is,
  with paragraph segments that carry no times. Evidence then anchors to a segment range for every
  type (DECISIONS rule 3), and the corpus database's importer has one path for both. Never served.
- **The corpus token:** each run through the text-in entry point rewrites the corpus run summary,
  so the caches' corpus token moves as it does after an ingest (`perf_cache.corpus_mtime`).

### 6. Ontology v3, the schema half (R3a)

The core release builds this section (R3a).

- **Evidence carries a character range in a named document of its item, always.** The document
  reference names the variant for translations. A time range only when the item is timed:
  podcasts yes, news no. An untimed quote or chunk carries no time at all, never 0.
- **The item id is a field every evidence reader reads where it reads `episode_id` today.** That
  starts with the identity guards' evidence builder, which skips any artifact without
  `episode_id`.
- **Every evidence and insight text carries `text_origin`:** `spoken`, `translated` or `authored`.
- **Also added:**
  - the KG 3.0 edges: `ATTRIBUTED_TO` (`SPOKEN_BY` stays as its audio specialisation, T-5),
    `HAS_ITEM`, `PARTICIPATES_IN`, `CONTRIBUTES_TO`, and `PUBLISHED_BY` from a source to its
    publisher organisation;
  - roles on people, as an open vocabulary: author, host, guest and participant now, others such as
    executive or analyst later without a schema change;
  - `Place` and time fields, declared and not yet filled. The time fields are kept apart: published,
    filed, period covered (start and end), observed, ingested.
- **Evidence locators are open beyond character and time.** An optional page and section, so a
  later document type such as `report` needs no schema bump.
- **News writes v3 from the start. Podcasts keep writing GI 3.x until R3b.**
- **Readers accept GI 3.x and ontology v3.** A GI 3.x quote is read as v3 with
  `text_origin: spoken` and its time range.
- **R3b,** rewriting the podcast corpus to v3 under the change method, comes in its own release,
  strictly before the corpus database's importer (ADR-178).
- **The KG schema is public's** and is versioned here. The GI schema moves with the private
  extension and follows the same rules there.
- **Evidence:** ADR-178.

### 7. Enrichment over text

- **Public alone:**
  - the summary;
  - the basic KG pass (topics, entities);
  - index rows.
- **With the private extension installed:** identity joining (§8).
- **Grounded insights for news are News 2.0.** First an offline evaluation: run GI 4.0 over the
  news fixtures and a sample of real articles, and measure quality, grounding rate and cost per
  article. Then GI 4.0's quotes anchor by character range in the body's segments. Until then,
  News 1.0 does not depend on the private extension's GI 4.0 work.
- **Prompts are per type** (a prompt profile), so a news article is not prompted as a transcript.
- **Proof that no podcast-only module runs:** a test runs a news item through the pipeline and fails
  if any audio, transcription, diarization or ad-cleaning module is imported or called.

### 8. People, publishers and the one graph

- **People mentioned in articles** join the same person graph as podcast people, through the
  existing joining and its guards. The guards' evidence builder reads the item id (§6), so news
  evidence counts, for example for the mention co-occurrence guard.
- **Bylines are people with the role author.**
  - A byline joins a person only within its publisher, in the shape of the host rule. Its verified
    names are the publisher's bylines from item metadata. Hosts' verified names come from the
    show's publisher text, which an article does not have.
  - **Two names in one byline are two people** (a co-byline guard, the analogue of co-voice).
  - Across publishers, or into podcasts, a byline joins only through the general rule and its
    guards.
- **An organisation as author** ("Meduza", "Reuters") is the publisher, not a person. That rule is
  for news only: for a filing or a company press release, the company is the author and the
  newsroom is the source, which later types decide.
- **Publishers are organisations** (T-12), held in a **publisher registry** in public configuration:
  id, name, web domain, external identifiers.
  - A source points at its publisher through `PUBLISHED_BY`.
  - A publisher that also publishes a podcast in the corpus (the BBC, Meduza) is one organisation
    across both.
  - Organisation nodes found by NER are not joined to publishers automatically in News 1.0.
- **An organisation carries its external identifiers side by side, never conflated:** a web domain
  now, and later a company's CIK, ticker or LEI.
- **Evidence:** ADR-180.

### 9. Search

- **News rows go into news tables of their own in the same search index** (the operator,
  2026-11-11), as non-English chunks already have their own table:
  - no change to the existing tables' schema, so no index version bump and no prod reindex;
  - podcast search cannot return news unless a query asks for it;
  - news can be dropped or rebuilt by itself.
- **News rows hold the summary and metadata for display; the article chunks are embedded for
  retrieval but store no text.** A hit shows the summary, never body text.
- **Cross-type search queries both and merges,** as the non-English table's results are merged.
  Results carry `content_type`; the default is podcast items only.
- **No new store:** the database search (E09) later takes all types together.
- **API and MCP tools are generic over content types.** A tool that lists, searches or explains
  items takes a type filter and does not assume an episode. A later product can then query any type
  the way Close Listening queries episodes.

### 10. Display: title, summary and a link

- **Every surface shows:**
  - the title, summary, publisher, byline, published time and source;
  - a link out to the article.
- **Quoted evidence snippets** (one or two sentences, as P-3 allows) come with GI in News 2.0.
- **Never shown: the article body. The measurable rule:**
  - no route or MCP response holds more than a stated number of contiguous characters of a news
    item's body (set with the first snippets in News 2.0; in News 1.0, none);
  - news search rows store no body text;
  - the corpus text-file route refuses a news body.
- **The public viewer** gets a basic news card and the type in search. **The private viewer** gets
  its own card.

### 11. The listener app (optional, behind a flag)

- **A small read-time proof, off by default:**
  - "in the news" on a storyline or theme page, by a cross-type search against its text;
  - a short strip on the home page for the themes a listener follows;
  - a news item kept on a board.
- **No new enrichment:** news does not enter the storyline or theme builders in this release.
- **The first slice to cut if the release is late,** in which case it moves to the release before
  the corpus database.

### 12. The prod sources (first set, 2026-11-11)

Chosen to match and complement the podcast corpus's subjects. Every feed answered HTTP 200 with
recent items on 2026-11-11.

| Area | Sources |
| --- | --- |
| AI and technology | Ars Technica, The Verge, MIT Technology Review, Import AI, Simon Willison's weblog |
| Central banks and economics | BIS central bankers' speeches, Federal Reserve press releases, BBC Business |
| World and geopolitics | BBC World, Al Jazeera, Meduza (English), The Diplomat, South China Morning Post (China) |
| Technology in the Global South | Rest of World, TechCabal |
| Health, biotech and science | STAT News, Quanta Magazine, Nature News, The Conversation |
| APIs | Hacker News (Algolia), The Guardian Content API |

**Known per source:**

- **SCMP:** returns a teaser only.
- **Al Jazeera:** its feed mixes in video pages, which are filtered.
- **The Brazilian Report:** wanted, since it publishes a podcast in the corpus, but its feed
  answered 404. It needs another route.

Fixtures and invented items, as for podcasts, cover the same shapes in CI. Fixtures use invented
names and domains only.

**Gates before a source is turned on in prod:**

- its terms of use are read (and the Guardian key is in place);
- it has a per-run item cap;
- the first month on fixtures and a dry run is measured: articles per day, enrichment cost per
  article, index growth, player-api memory.

### 13. Later types this design must fit

News is the first type built on these seams, not the only one they serve. Each later type below is
out of scope here. Each is a check that this release builds nothing a later type would have to
undo.

| Later type | Transport | What it needs from this release | Where |
| --- | --- | --- | --- |
| Company filings (annual, quarterly and current reports) | SEC EDGAR's public JSON APIs | a cursor of filing date and accession number; a required contact header and a request budget; ids from the accession number; evidence by page and section; period-covered and filed times; company identifiers (CIK, ticker, LEI) | §2, §3, §6, §8; ADR-177, ADR-178 |
| Reports and presentations (HTML, PDF) | HTTP documents | evidence by page and section; a type's own fields under its key | §5, §6; ADR-178, ADR-181 |
| Earnings-call and other recorded audio | investor-relations webcasts | roles beyond host and guest (executive, analyst); **an audio-in entry point for a non-podcast type, which this design does not build:** the podcast path takes RSS items only. A later seam. | §6, §8; ADR-180 |
| Social posts | AT Protocol, ActivityPub | short items, authors as people; a transport with its own cursor | §3, §8; ADR-177, ADR-180 |
| Reported figures (structured facts) | EDGAR company facts | organisations with external identifiers; time kept apart | §8; ADR-178. The fact model itself comes with structured intelligence. |

Every surface and tool that lists or searches items takes a type filter (§9). A later product can
then read these types through the platform's API and MCP, the way the podcast apps read episodes.

### 14. Impact on the pipeline and the tests (review of 2026-11-11)

A read of the code at public 0357b8a96.

- **The half after the transcript is mostly body-neutral.** Summary, GI, KG and the bridge take text
  and optional segments.
- **The half before it assumes an RSS item with an enclosure, and audio.**
- **Nothing in `src/` has a content-type concept yet.**
- **Most podcast-specific things would not fail loudly for a news item.** They would produce nothing,
  or something wrong. So the guards below matter as much as the code.

**Places that must change, and how each fails today:**

| # | Place | Today, for a news item | Change |
| --- | --- | --- | --- |
| 1 | `episode_processor.process_episode_download` (no transcript, no enclosure) | returns nothing, with no error and no metadata | news never reaches it: the text-in entry point (§1) builds the processing job from the extracted body |
| 2 | `transcript_source` Literal (`workflow/types.py:90`, `metadata_generation.py:640`) | a validation error on any new source | widen it to a body source that includes extraction |
| 3 | GI quotes with no segments (`gi/pipeline.py`, `_char_range_to_ms`) | written with fabricated 0 ms timestamps and a transcript reference, and they pass the 3.1 schema | v3 (§6): no time range when untimed |
| 4 | The search indexer's body chunks (`search/indexer.py`) and the corpus text-file route (`GET /corpus/text-file`) | would return article text in search hits and as a raw file | news rows in news tables of their own that store no body text; a hit shows the summary; the text-file route refuses a news body (§9, §10, ADR-179) |
| 5 | Ad excision inside summary, KG and GI (three call sites) | would cut "pre-roll" and "post-roll" from article text | skipped by the stage profile |
| 6 | `rss/parser.parse_rss_items` | an Atom feed gives zero items, silently | the news RSS/Atom transport reads feeds with `feedparser`; the podcast parser stays as it is |
| 7 | The feed language gate | refuses a whole source that declares no language, which every JSON API does | language comes from the transport or the item |
| 8 | The catalog row (`CatalogEpisodeRow`) and the episode summary (`content_source`) | news shows as podcast episodes, "ready" with a podcast title | rows carry `content_type`; podcast surfaces filter on it |
| 9 | Host and guest detection, the speakers validator, the show health check | bylines labelled as hosts; every item flagged with no host or no transcript | skipped or replaced by the stage profile; bylines go through ADR-180 |
| 10 | `RssFeedEntry` and `Config` (`extra="forbid"`), and the two multi-feed loops (`service.py`, `cli.py`) | no place to declare a type or transport, and two loops to keep in step | the source declares `content_type` and `transport`; one dispatch point used by both loops |
| 11 | Resume and de-duplication (`append_resume`, with ids from `get_episode_id_from_episode` over the RSS item) | every run re-processes or writes twice | the item id from the source's own id (§2) goes through the same skip; no separate seen-set |
| 12 | Translation (`translation_stage`, converging in `generate_episode_metadata` on transcript variants) | undefined for an article; variant file names are transcript-named | the type's profile says whether an article is translated; variant names follow the body file |
| 13 | Corpus integrity in the ops routes (`server/routes/ops.py`, missing artifacts) | may report every news item as missing a transcript or segments (not checked line by line) | expected artifacts per type |

**Also affected:**

- **Corpus migrations:** 27 of the 44 walk every metadata file and touch transcripts, segments,
  speakers or timestamps. Each either skips non-podcast items by type or is shown to be a no-op on a
  news fixture.
- **Prompts:** 208 of 219 prompt files mention a podcast, a transcript or an episode. News gets its
  own prompt profile (§7).
- **The private viewer:** 33 places hard-code the `Episode` node type. That is the private viewer's
  own work.

**Where the seams go:**

- **The registries:** `Extension.content_types` and `Extension.transports` beside `pipeline_hooks`,
  with podcast built in, as `builtin_stage_hooks` does.
- **The source:** declares its type and transport. One dispatch point, used by both batch loops,
  chooses `run_pipeline` or the text-in entry point.
- **The text-in entry point reuses:** the environment setup, the run manifest, the processing
  resources, the processing jobs (with the extracted body as their body file) and the finalisation.
  It replaces only the feed fetch and the download and transcription threads.
- **One item context, not a type argument on every hook** (ingestion's review):
  - the processing job carries one item context: type, profile, item id and body path;
  - the type's profile decides which hooks run at all;
  - a hook that needs the type reads it from the context.
  - Podcast calls stay byte-identical. "News never calls X" becomes a property of the profile,
    which the guard tests directly.
- **Built since this review, on ingestion's local branch** (`feat/prompt-and-cleaning-seams`,
  behaviour-neutral, recorded in ADR-176, not pushed):
  - `Extension.prompt_dirs`: the prompt store searches extensions' directories first, so a type's
    prompt profile (§7) is a directory or a name prefix;
  - five cleaning stage hooks: transcript cleaner, clean for summary, cleaning context, remove
    commercials, cross-promotion excision. Place 5 (ad excision for news) is then a profile
    decision: a news job does not call them.
- **Cursors** are the transport's own, in one small file per source beside the source's runs,
  written atomically. They do not move the caches' corpus token: `corpus_mtime` keys only on named
  stamp files at the corpus root (`perf_cache.py`).
- **The text-in entry point must move that token** the way an ingest does (the corpus run
  summary), or the caches keep serving the corpus as it was before the news items. Reusing the
  pipeline's finalisation gives this.
- **Readers that are hard to make dual-version** are the ones that use quote times, which news
  quotes do not have:
  - the Player's moments and clips (seek by start time);
  - the transcript-chunk lift (it matches by character overlap but carries times);
  - the indexer's quote rows with times;
  - strict-validation reads, which need the v3 schema beside 3.1.

  Readers that use character ranges only are easy.

**Sizes and order** (ingestion's judgement, not measured):

| Step | Size |
| --- | --- |
| Content-type and transport registries, and the dispatch point | M |
| R3a: schemas, dual-version readers, fixtures for both | M |
| The text-in entry point (the processing job assumes a transcript path and an episode) | M–L |
| The news type on the seams: transports, extraction, profile, prompts, viewer card | L |
| Public's plain defaults | M |

They come in that order. News is a public capability that does not depend on the move, so it can
land before the move. That is the cut to make if the core release is split again.

**Tests and guards to add.**

None of the proof items in "Proof" has a test today. Every layer is podcast-shaped: fixtures, both
mock HTTP hosts, the canned LLM replies, the GI and KG schemas, the import guard, the viewer corpus.
In order of importance:

1. **The architecture guards** (unit and guardrails, on every PR):
   - **No podcast module for news:** an import probe cloned from
     `test_public_imports_load_no_speaker_stage.py`. It names:
     - the audio preprocessing subpackage (not all of preprocessing, whose core text cleaning public
       keeps);
     - the transcription providers;
     - the speaker stage and diarization;
     - commercial cleaning;
     - audio sync;
     - the audio download.

     Plus a runtime test that runs a news item with those modules made to raise.
   - **The guard's own removal proof:** with a podcast stage wired in on purpose, the guard fires
     (`test_guard_capability.py`'s pattern).
   - **Public names no private package:** a test of the rule ADR-170 states but nothing enforces
     today.
   - **The registries:** a consistency test for the new registries, and a test that every route,
     MCP tool and search call accepts a type filter.
2. **Byte-identical podcast output:** the e2e fixture feed run twice, plain and with a test
   extension installed, comparing artifact hashes. Today's nearest check
   (`test_english_artifact_allowlist.py`) compares shapes, not bytes, and never runs with and without
   an extension.
3. **Schemas, both versions:**
   - v3 in the GI and KG schemas, with the item id in place of a required `episode_id`;
   - minimal valid and invalid v3 artifacts for podcast and news;
   - a fixture corpus holding GI 3.1 and v3 side by side;
   - every route that reads insights or quotes run over both.
   - The N-1 compatibility test today only strips manifest fields. It gains a GI 3.x corpus read by v3
     code.
   - A schema change trips the migration guard, so that PR carries a migration or its
     "migration not required" reason.
4. **HTTP fakes for both transports:**
   - news RSS and Atom feeds;
   - saved article pages, including a teaser and a video item;
   - Hacker News and Guardian JSON with paging and an API key;
   - error injection (404, 429 with Retry-After, malformed JSON, end of cursor).
   - The same files go into the stack-test's mock host.
   - A transport conformance suite, in the style of the search backend contract suite: cursor,
     idempotent ids, rate budget and retry, run against every transport.
   - The LLM mock learns the news prompt profile. Otherwise news prompts get transcript replies.
5. **Fixtures:** `tests/fixtures/news/v1`:
   - invented publishers, feeds, API JSON and pages;
   - extraction goldens, including the readability fallback case;
   - byline shapes;
   - one publisher that also has a podcast;
   - captured LLM output for news;
   - a generator script.
   - News items join the app validation corpus, or a sibling corpus, so the API, MCP, search and
     viewer tests read both types.
   - **Invented names and domains only:** the identifier deny-list and gitleaks scan fixtures.
6. **Never served:** an API-level negative test that no route or MCP response contains a news body.
7. **The PR tiers:** the news e2e is marked `critical_path`, so the PR subsets of integration and e2e
   run it. The full stack-test runs on main only, so it cannot be the only cover.
8. **Search across types:** `content_type` in the search backend contract, so the fake and LanceDB
   must agree; a query returning both types, and a filter returning one.
9. **Bylines:** rows in a guardrail matrix like the speaker-roles matrix: within-publisher joins, an
   organisation as author, no join into podcast people without the general rule.
10. **Acceptance:** a news feed fragment. The acceptance check of GI quote offsets assumes transcript
    chunks and needs an article-chunk equivalent.
11. **The viewer:**
    - a corpus with news;
    - a Playwright spec for the basic card (title, summary, publisher, byline, time, link, no body)
      and the type chip in search;
    - its surface-map entry and unit tests.
    - The private viewer's tests are its own repo's.
12. **Housekeeping:**
    - a test-policy rule that news tests use no podcast fixture paths;
    - wire the existing search lint into CI, where it gates nothing today;
    - the test and fixture READMEs, whose counts are stale and which describe the data as podcasts
      only.

**The existing gates hold news to the podcast bar:** unified coverage (70%), Codecov patch (77.47%),
the test policy and network isolation.

## Proof

**The core release** (the seams with the test-only type, and R3a):

1. **Moved and unmoved code give identical podcast artifacts** on a snapshot clone with the
   private extension installed, except run manifests.
2. **Podcast artifacts are unchanged when the test-only type is configured beside them.**
3. **A public-only install runs end to end.** Its artifacts differ by design: no GI, no speakers.
4. **Shared reads return podcast rows only** over a mixed fixture corpus unless a filter is passed:
   the catalog, the episode routes, search.
5. **Every corpus-wide walker runs over the mixed corpus** (podcast GI 3.1 beside test-type v3):
   - the identity ledger build, which succeeds with no podcast join changed;
   - entity clusters, the enrichers, enrich-edges;
   - the catalog builders, index staleness, the corpus migrations.
6. **v3 and GI 3.x artifacts both read** on every route that reads insights or quotes; an untimed
   quote or chunk carries no time, never 0.
7. **A run through the text-in entry point moves the caches' corpus token.**
8. **No podcast-only module runs for a test-type item:** an import probe and a runtime guard, with
   the guard's own removal proof.

**News 1.0:**

1. **Podcast artifacts are unchanged** with news sources configured.
2. **Fixture feeds and fixture API responses ingest end to end in CI** on both transports, marked
   critical-path so the PR test subsets run them.
3. **No podcast-only module runs for a news item** (as core 8, over the news profile).
4. **A public-only install** ingests and serves news.
5. **The transport conformance suite**, run against every transport:
   - cursor and idempotent ids;
   - the same article over two transports is one item;
   - rate budget and retry;
   - robots.txt and conditional GET.
6. **Bylines:**
   - parsing measured against the golden set;
   - within-publisher joins;
   - two names in one byline never merge;
   - an organisation as author is the publisher;
   - no byline joins a podcast person without the general rule;
   - a news-only pair of people gets the mention co-occurrence guard's verdict.
7. **Search:** podcast only by default; both types when asked; the news tables store no body text.
8. **Never served:** the measurable rule (§10) asserted on every route and MCP tool.
9. **Rights:** `serveable` and `mineable` honoured; teaser items never chunked or enriched as whole.

The full test and guard list is §14.

## Rollout and rollback

- **Sources are configuration.** Turning them off stops new items. Removing the type's items is a
  corpus migration with its own undo.
- **No existing file changes,** so rollback is the previous images and the sources off.
- **Each source goes on only after its gates (§12):** terms read, a cap, the first month measured.
- **Neither release needs a freeze for this design.** R3b, which does, is its own release.

## Risks

- **Extraction quality varies by site and over time.** Monitored by text length per source and the
  teaser rate. The sample behind ADR-179 was one article per source, so per-source goldens set the
  fallback threshold.
- **Bylines are noisy.** A false join is worse than a missed one (RFC-131's rule), so bylines are
  parsed against a golden set, scoped to their publisher, and guarded against co-byline merges.
- **Volume:** news adds rows every day, and player-api's memory is already tight (the cause of its
  10-26/27 OOM kills is unknown). Per-source caps, the first month's measurement, and the bigger
  box with the next deploy.
- **Terms of use differ per source.** A gate before each source goes on (§12).
- **Sizes are judgements** (ingestion's, 2026-11-11), not measurements. The text-in entry point
  may run to L, since the processing job assumes an RSS episode and a transcript path.

## Proposed ADRs

| ADR | Decision |
| --- | --- |
| ADR-177 | Content types and transports are extension seams; podcast is the first built-in type; one dispatch point and one item context; a text-in entry point beside `run_pipeline` |
| ADR-178 | Ontology v3 is adopted in two halves: the schema in the core release (new types write it, readers accept both), the podcast corpus migration in R3b, strictly before the corpus database's importer |
| ADR-179 | Article text is extracted with trafilatura and a readability fallback; it is stored for enrichment and never served |
| ADR-180 | Bylines join people within their publisher, in the shape of the host rule, with a co-byline guard; publishers are organisations in a registry, one across media |
| ADR-181 | A content type's own fields live with the type: under a key named for it in files, in its module's table or one JSON column in the database; never a column per type on the shared item |

## Not covered yet

- **The private extension's side:** identity joining for bylines in News 1.0, GI 4.0 for news in
  News 2.0, designed in that repo. What it needs from public (ingestion's review):
  - the prompt search path (built on ingestion's branch);
  - the item context in the processing job;
  - KG 3.0 in public, with roles and `ATTRIBUTED_TO`, since GI 4.0 refers to them;
  - bylines as item metadata (role author), with the publisher's organisation id, because today's
    speaker readers read transcript segments and have nothing for an article;
  - the publisher registry in public, so a show's organisation and a publication's are one id.
- **Owners** are agreed with ingestion (2026-11-11) and recorded in the platform plan, not here.
- **The ops integrity check (place 13)** is not checked line by line.
- **Volume and cost:** articles per day, enrichment cost per article, index growth. Measured on the
  first month of fixtures and a dry run.
- **Each source's terms of use** and the Guardian key.
- **The measurable never-served cap** is set with News 2.0's snippets.
