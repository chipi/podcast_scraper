# ADR-179: Article text is extracted with trafilatura and a readability fallback, stored for enrichment, never served

- **Status**: Proposed
- **Date**: 2026-11-11
- **Authors**: Marko Dragoljevic
- **Related**: [RFC-134](../rfc/RFC-134-news-as-the-second-content-type.md) (news as the second
  content type)

## Context & Problem Statement

A news item arrives as a feed entry or an API record pointing at a web page. Enrichment needs the
article's text, without the page's navigation, comments and related links.

The operator decided (2026-11-11) that surfaces show only a title, a summary, basic item metadata
and short evidence snippets, always with a link out. The article itself is never rendered.

Five Python extractors were run on the newest article of each of 19 candidate sources on
2026-11-11. All five were run on the same downloaded page:

- **trafilatura 2.3.1** (Apache-2.0):
  - a byline on 15 of 19 and a date on 18 of 19;
  - 5–100 ms;
  - it cut one article (Ars Technica) to 1,611 of about 5,500 characters, and `favor_recall` did not
    change that.
- **readability-lxml 0.9** (Apache-2.0): a complete body on that page and the others, but no byline
  or date.
- **newspaper4k** (MIT):
  - cut a Verge article to 533 characters;
  - bylines mixed with page text ("www.facebook.com", comment-policy text);
  - 70–2,300 ms and the heaviest dependencies.
- **goose3** (Apache): cut two articles short and missed some bylines.
- **resiliparse** (Apache-2.0):
  - fastest;
  - keeps page chrome ("Skip to content", comment counts);
  - returned 279 characters for one newsletter.

## Decision

1. **trafilatura extracts the body, byline and date.**
2. **readability-lxml is the fallback for the body.** When trafilatura's text is much shorter than
   readability's on the same page, readability's text is used.
3. **feedparser reads RSS and Atom feeds.**
4. **Bylines come from the feed's author field first and the page second.** Publisher names are
   removed from bylines.
5. **An item with too little text** (a teaser behind a paywall, a video page) is kept with its title
   and feed summary and marked `text_status: teaser`. It is not enriched as a whole article.
6. **The extracted text is stored with the item as its body artifact, the way a transcript is,** and
   used only for enrichment and evidence ranges. No route, viewer or app serves it. The measurable
   rule:
   - **response cap:** no response holds more than a stated number of contiguous characters of a
     news body;
   - **search:** news chunk rows keep their text, for keyword and vector retrieval alike; a news
     hit returns the item's summary and metadata, never the chunk's text (the operator,
     2026-11-11);
   - **the text-file route** refuses a news body.
   - **Snippets:** quoted evidence snippets of one or two sentences come with grounded insights in
     News 2.0, which sets the cap. In News 1.0 no body text is shown at all.
7. **`mineable: false` means the body is not stored and the item is not enriched;**
   `serveable` says what a surface may show.
8. **The fetched page is kept as the item's raw cache** (the operator, 2026-11-11): the HTML, or the
   API's JSON record, beside the extracted text, never served.
   - A better extractor, or a fix from stored evidence, then re-runs with no re-fetch, which could
     return a corrected article or a paywall.
   - It moves to the archive store, as audio does, once news ingestion is stable. Not during the
     first deploys and source onboarding, which will reprocess often.

## Consequences

- **Positive:**
  - two Apache-licensed libraries on `lxml`;
  - metadata comes from one of them;
  - a measured fallback for the case that failed.
- **Negative:** new dependencies:
  - `feedparser`, `trafilatura` (with `lxml`, `justext`, `htmldate`, which pulls `dateparser`, and
    `courlan`, which pulls `babel` and `tld`);
  - `readability-lxml` (with `chardet`, `cssselect` and `lxml-html-clean`).
  - Approved by the operator on 2026-11-11.
- **Neutral:**
  - the sample was one article per source;
  - extraction quality is tracked per source after launch (text length and teaser rate).

## Alternatives considered

- **readability-lxml alone.** Rejected: no byline or date, both of which the graph needs.
- **newspaper4k.** Rejected: slower, noisier bylines, and the heaviest dependency set.
- **resiliparse.** Rejected for now: fastest, but its output needs its own clean-up and it failed on
  a newsletter page.
- **Serving the extracted text in the viewer.** Rejected by the operator: surfaces link out.
