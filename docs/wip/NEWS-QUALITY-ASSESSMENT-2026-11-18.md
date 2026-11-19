# News 1.0: quality of the first real ingestion (2026-11-18)

The operator asked (2026-11-18) whether the summary and KG providers podcasts use can be reused
unchanged for news, on both the LLM path and the ML path, and what a first real ingestion says
about quality. No fixes were made for this assessment; findings that need work are listed at the
end. Numbers are from the runs named in each table; nothing here is a prediction.

## Data

| Run | What | Items |
| --- | --- | --- |
| `all8` | every configured source, writer + `dev_dgx_full` (vLLM Qwen3-30B on the DGX: summary, KG, translation) | 257 from 18 sources (BBC has 2 feeds; Hacker News included; the Guardian off: no key) |
| `all-air` | a fresh run, `airgapped` profile (SummLlama 3.2-3B summaries; KG through the summary provider), stopped for memory after 6 sources | 78 |
| shared | items both runs hold (same `item_id`) | 70 |
| `sample-air` | `airgapped`, the newest item of 8 varied sources | 8 |

`all8` ran with code from before the prompt-example guard (public 44b87e08f), so it measures the
KG leak as it was.

## The answer in short

- **The LLM path (`dev_dgx_full`) is reusable for news as it is.** Summaries and KG come out of the
  existing providers with no news-specific method: about 10 topics and 13 entities per full article,
  summaries written for an article, not for a conversation. Its problems are entity-level (below),
  and one (the leaked example entity) is already guarded.
- **The ML path (`airgapped`) is not usable for news as it is.** Its summaries are accurate but four
  times longer, written as one run-together bullet list, and framed as speech in 13% of articles;
  it extracts no KG at all (no topics, no entities): #2498, #2499. That gap is the ML path's, not
  news's: the same KG route gives nothing on a podcast either (not measured here).
- **Both prompts are written for podcasts** ("episode", "transcript", "host", "speaker": 12 to 13
  mentions each in the KG prompt `shared/kg_graph_extraction/v5.j2` and the bundled summary
  prompts). The large model reads an article correctly anyway; the small model takes the framing
  literally. A news prompt profile (ADR-177's `prompt_profile` on the type) is the clean place for
  news wording when it is needed, and the ML path needs it first.

## Summaries

On the 70 shared articles:

| | DGX (LLM) | airgapped (ML) |
| --- | --- | --- |
| Present | 69 / 70 | 70 / 70 |
| Median length | 359 characters | 1,478 characters |
| Framed as speech ("the speaker…", "this episode") | 1 | 9 (13%) |
| Stored as one run-together bullet list (" - ") | 0 | 70 |

Over all of `all8` (257): 250 summaries; the 7 missing are 4 Nature teasers, the French TechCabal
item (held back for translation: no translator on that path), 1 Hacker News and 1 Verge item (not
diagnosed). About 20 summaries were read side by side (both profiles): the two never contradict
each other, the airgapped ones keep more detail and the DGX ones read as a summary. Facts were
not checked against the articles.

Example, the same article (Ars Technica, AT&T v. Comcast "fiber-powered" advertising):

- DGX: "Comcast's use of 'fiber-powered' to describe its Xfinity Internet service is misleading
  because the final connection to homes uses coaxial cable…"
- airgapped: "AT&T filed a lawsuit against Comcast, claiming the company's "fiber-powered" marketing claim
  for its Xfinity Internet service is false… - Comcast's fiber deployment is limited to small
  por…" (the bullets joined into one string).

## Knowledge graph

`all8`, 255 KG files (the 2 without: the French item, held for translation, and one not
diagnosed): 2,268 topics, 966 people, 1,400 organisations, 755 objects.

| Per article | Full text | Teaser (Nature standfirst) | Teaser (SCMP) |
| --- | --- | --- | --- |
| Topics | 9 to 10 | 4.6 | 5.5 |

Topics are canonical noun phrases and on subject ("open-source ai movement", "nvidia chip access",
"hybrid fiber-coaxial network", "monetary policy" on central-bank speeches). Two kinds of noise:
lower-casing drops apostrophes ("huangs resignation"), and an occasional garbled term ("noul
questions").

Entity problems found:

| Problem | Count in `all8` | Example |
| --- | --- | --- |
| A prompt example copied into an article that never names it | 4 of 257 (1.6%; podcasts 0.45%) | "Cascadia Alliance" in a BIS speech on Japan's current account |
| A person also present as an organisation | 19 | "Simon Willison" as Person (byline) and Organization (the publisher node the writer adds, and the LLM's own) |
| Not a person typed as Person | seen, not counted | "Reddit user", a model name ("Claude Fable 5.2") |
| A site or tool typed as Organization | seen, not counted | "Hacker News", "Pyodide" |

The airgapped run's KG holds only the nodes the writer adds itself (the publisher and byline
authors): no topics, no entities from the text.

## The writer (both profiles)

From the real runs, before enrichment:

- 257 items, 257 ids, none written twice (after the duplicate fix); every item has its raw page.
- `all8`: full text on 224 of 257 (87%), 32 teasers (SCMP is teaser-only; Nature is paywalled,
  so its items are the page's standfirst; STAT, MIT TR and Quanta a few each) and 1 video page.
- Bylines on 210 of 257 (Meduza and the Fed name no authors; Hacker News links go anywhere).
- Language: 257 English and 1 French (detected from the text; enabled, so it goes to translation).

## Not covered

- The comparison on identical articles is 70 items over 6 sources, not all 257: the full airgapped
  re-derive (`--pipeline-stage rederive_only`, about 17 hours on this machine) was stopped to free
  memory and waits for a quieter moment.
- Summaries were read, not scored: about 20 by hand; no rubric, no reference summaries, facts not
  checked against the articles.
- KG precision was not measured against a gold set: the entity problems above are the ones seen
  and grepped for, not a rate over every entity.
- Translation was not exercised, and cannot be for news yet: it is off by choice in News 1.0
  (gap 8), and turning the switch on reaches the translation stage but news text has no path
  through it (the stage translates a podcast's speaker screenplay).
- The all8 corpus on disk predates gaps 2, 8, 9 and 10: its figures above are as written then.
- GI is out of News 1.0 and not assessed.
- The ML KG gap was not measured on podcasts.

## Gaps this found, and what closed them

Each fixed with a failing test first (public commits on `feat/news-release`, 2026-11-18):

1. The byline parser split a name from its affiliation ("Alexander Korolev, Associate Professor
   in Politics and International Relations, UNSW" became four people). Fixed in bc4b2288b.
2. Teasers and video items were enriched as if whole (E05: they must not be). Operator: KG, no
   summary. Fixed in 1a6e5a058 (`skipped_reason: not_whole_text`).
3. `mineable: false` was not enforced. Fixed in 9fe67077a.
4. A duplicate's second transport id was not merged into the item it duplicates. Fixed in
   19b067505 (the written item's metadata gains it). In all8 the only duplicates were 3 BBC
   articles in two RSS feeds, where nothing is added.
5. The airgapped summary was stored as one run-together bullet list. Corrected cause (first
   written here as "the writer takes the raw text"): `sanitize_summary` turned the reply's line
   breaks into spaces before parsing, so SummLlama's one-bullet-per-line reply became one bullet
   (78 of 78 items). Fixed in 2ca276cb0; podcast summaries from SummLlama had the same fault.
6. The prod pipeline image lacked the `news` extra. Fixed in 2619ff1a6 (not built locally).
7. Corrected: 2 Hacker News pages (not 3) were not written, and both are right: GitHub answers
   404 for `github.com/microsoft/onnxruntime` even to a browser, and `hy.tencent.ai` answers 567
   (bot protection) with no article text. The 2 items without a KG are the 2 non-English items
   (8, below). Of the 5 without a summary, 4 are Nature teasers (no summary by design since 2)
   and 1 is The Verge, below (9).

Found while closing those:

- **8.** Translation never ran for a news item: the stage needs a podcast's segments and speaker
  screenplay, so both non-English items were skipped, and their unusable marker was lost, so
  they were served untranslated. Operator: translation off for News 1.0, with a per-source
  switch (`translate: true`). Fixed in 27c60d2d1: off, such an item is written, its stages are
  recorded as `translation_off`, and it is marked unusable (top-level fields, contract
  news-item-1.0); public's reader honours the marker in its own commit.
- **9.** The Verge item lost its summary to invalid JSON: the first bullet opened with a quoted term
  (`["AGI-pilled" describes …`), and the re-roll at temperature 0 repeated it. Not the first
  time: prod VictoriaLogs shows the same shape on a podcast episode on 2026-11-08 (`[“Are We
  Cooking?” rebrand …`), saved without a summary, unnoticed. Fixed in 2cab5b549 (the parser
  escapes the term's quotes); the DGX rederive of the Verge source then wrote that item's
  summary. The repaired-summary warning's wrong text is fixed in 0e8258df0.
- **10.** A 75-word English Mastodon post was stored as German (its server's `lang=de`): the text
  could overrule a declaration only from 200 words. News uses 50 since 40bb2bb6c (measured on
  all8: any threshold from 20 to 75 changes exactly that item, rightly).

On the providers, for later (the operator, 2026-11-18: no relabel now): issues #2498 (ML KG
empty) and #2499 (ML summary framing), a news prompt profile, entity typing (persons that are
not, sites as organisations).
