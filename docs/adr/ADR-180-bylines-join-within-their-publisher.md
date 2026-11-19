# ADR-180: Bylines join people within their publisher; publishers are organisations across media

- **Status**: Accepted (the operator, 2026-11-17); rule 6 amended 2026-11-18 (below)
- **Date**: 2026-11-11
- **Authors**: Marko Dragoljevic
- **Related**: [RFC-134](../rfc/RFC-134-news-as-the-second-content-type.md) (news as the second
  content type)

## Context & Problem Statement

The operator wants one graph across media (2026-11-11): a person discussed on a podcast and in the
news is one person. News brings people two ways:

- **people mentioned** in the text;
- **authors** named in the byline.

Hosts already have a rule of this shape. A person seated as host joins a name only within that
show, against the show's own verified names (`identity.guards.host_join_pair_rule`). Elsewhere a
host joins only through the general rule and its guards.

Bylines are noisy. In a sample of 19 sources on 2026-11-11 they included:

- several authors in one string;
- a surname first ("Arnold; Mark");
- a date stuck to a name ("Perri Thaler November");
- the publisher as the author ("Meduza", "TechCabal; Adonijah Ndege").

A wrong merge is worse than a cautious split.

## Decision

1. **People mentioned in articles join through the existing joining and its guards,** like any
   person in the corpus. The guards' evidence builder reads the item id (ADR-178). Today it skips any
   artifact without `episode_id`, so news would add no guard evidence.
2. **A byline is a person with the role author, and joins within its publisher only,** in the shape
   of the host rule. Its verified names are the publisher's bylines from item metadata (the feed's
   author field first, the page's second). The host rule's verified names come from the show's
   publisher text, which an article does not have, so this is a new evidence source in the same
   shape, not the same rule unchanged.
3. **Two names in one byline are two people.** This is a co-byline guard, the analogue of the
   co-voice guard: they never merge.
4. **Across publishers, or into podcasts, a byline joins only through the general rule and its
   guards.** There is no byline-specific shortcut.
5. **An organisation in a news byline is the publisher, not a person.** It is removed from the
   byline before joining. This holds for news only: for a filing or a company press release, the
   company is the author and the newsroom is the source, which those types decide.
6. **Publishers are organisations (T-12), held in a publisher registry in public configuration**
   (`config/publishers.yaml`, loaded by `podcast_scraper.publishers`): id, name, web domains,
   external identifiers, and the KG `Source` ids it publishes.
   - A source points at its publisher through `PUBLISHED_BY` (KG 3.0), drawn from the registry's
     list of `Source` ids (`podcast:…`, `source:…`), one list for every type.
   - A news item's publisher is found by its canonical URL's host: a listed domain matches itself
     and its subdomains, the longest match wins, and no match means no publisher, never a guess.
   - A publisher that also publishes a podcast in the corpus (the BBC, Meduza) is one organisation
     across both.
   - External identifiers (a web domain now; later a company's CIK, ticker or LEI) are kept side by
     side and never merged into one key.
   - **Amended 2026-11-18 (the operator: "Yes on same organization"):** a publisher's id is an
     organisation id, `org:{slug}`, the namespace NER's organisation nodes use. An NER organisation
     with the identical slug is the same organisation as the publisher. Nothing else joins them:
     never a fuzzy, partial or translated name. (Before: "Organisation nodes found by NER are not
     joined to publishers automatically, for now.")
7. **Roles are an open vocabulary.** Author joins host and guest now. Later roles, such as an
   earnings call's executive or analyst, follow the same shape: joining within the scope that
   verifies the name, and the general rule elsewhere.
8. **Byline parsing is measured before it joins anyone:** a golden set of the shapes above, with
   its precision recorded.

## Consequences

- **Positive:**
  - one graph across media;
  - the host rule's shape, and the general rule's guards, apply to bylines.
- **Negative:**
  - new code for byline clean-up (splitting, order, organisation removal), the co-byline guard and
    the publisher registry, each with fixtures;
  - a journalist who appears on a podcast joins only when the general rule allows it, so some true
    joins are missed at first.
- **Neutral:** the joining itself moves with the private extension. Public provides the bylines as
  item metadata with their role and the publisher's organisation id.

## Alternatives considered

- **Join bylines like any name.** Rejected. Common names across many publishers would merge
  different people.
- **Keep bylines out of the graph.** Rejected by the operator. Authors are part of one graph.
- **Join bylines to podcast people on name alone when the publisher also has a podcast.**
  Rejected for now. To revisit with measured joins after launch.
