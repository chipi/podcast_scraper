# ADR-164: Discover is a taste — server-paged rails, one card per episode

- **Status**: Accepted
- **Date**: 2026-10-11
- **Authors**: Marko Dragoljevic
- **See Also**: [ADR-163](ADR-163-the-jobs-of-discover.md) (the jobs of Discover),
  [PRD-048](../prd/PRD-048-discover-and-explore.md), [RFC-126](../rfc/RFC-126-discover-rails-reasons-paging-and-reels.md),
  [UXS-012](../uxs/UXS-012-consumer-home.md)

## Context & Problem Statement

With five rails of up to six cards each, a listener with history scrolled about twenty-one phone
screens of Discover (15,472 px, measured on the fixture on 2026-10-11). The same episode appeared on
up to four rails at once. Each rail loaded independently in the browser, so removing repeats would
have depended on which rail answered first, and "show more" could only reveal cards already loaded.

## Decision

1. **A taste.** Every rail shows two cards. "Show more (N)" asks the server for the next page; N is
   what the server says is left. "Show less" returns to the first page.
2. **Paged on the server.** `GET /discover/rails` returns every rail's first page in page order.
   `GET /discover/rails/{rail}?offset&limit&exclude` returns a later page. Every page is a slice of
   one ranked list per rail, built on the server (`app_discover_rails.rail_cards`), with "Less like
   this" and the outside share already applied.
3. **One card per episode on the page.** The highest rail keeps an episode. A later page leaves out
   the episodes the rails above showed first (`exclude`), so offsets stay consistent.
4. **The server decides cold start.** Under Mine with an empty world, the quote rail is everyone's
   cards, flagged `fallback`, and the page labels them.
5. **Same control everywhere.** "Show more" is the shared `ShowAllToggle` (incremental mode), the
   control Library already uses for sections of a multi-section hub.

## Consequences

**Positive**

- One request renders Discover; the page's dedup no longer depends on load order.
- Telemetry counts exactly the cards shown.
- The rules (dedup, "less", cold start) live in one place and apply to every page.

**Negative**

- A rail's later pages depend on what the rails above showed, so the client must send `exclude`.
- If two rails are both expanded, an episode can appear in both expanded tails (the dedup covers the
  first pages). Accepted: rare, and visible only after two explicit "Show more" taps.

**Neutral**

- The per-rail endpoints were removed (pre-launch; no other caller).

## Alternatives Considered

- **Client-side reveal** (load six, show two). Rejected by the operator: more must come from the
  server, as with every other paged API.
- **Client-side dedup** across independent requests. Rejected: depends on load order; miscounts.
- **A hard cap of three cards** with no "more". Rejected: drops cards a listener may want and does
  not remove the repeats.
