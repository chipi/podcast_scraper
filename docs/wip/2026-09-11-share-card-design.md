# Shareable card — design note (2026-09-11)

Design record for the shareable "collectible" card (#2036). Captures the aesthetic decision, the
one that was rejected and *why*, the exact tokens, the layout, and what's built vs left. This is the
design SSOT for the card; the interaction spec lives in UXS-014 (§Sharing).

## Use case

A **short, beautiful overview** of an entity (show / episode / topic / storyline / person / org)
you can post anywhere — the outward growth loop. Inspiration was "collectible / player cards," but
the audience is people who **listen to learn**, so the card has to read **smart**, not gamified.

## The rejected direction (and why)

First concept was a Higgsfield-generated **glossy holographic trading-card frame** (teal foil,
sheen, ornamented border). Operator rejected it: *"too glossy and childish, not aligned with the
rest of the project."* The lesson: foil/holographic is **game-card** language; it fought the app's
restraint. A generated decorative frame is the wrong tool — the app's own design system already
*is* the "smart" look.

## The chosen direction — design-system-native editorial minimalism

**Modern, minimal, few colors.** The card is the app's own language rendered beautifully; **no
generated frame**. Quote-led editorial layout (operator pick over stat-led / type-only).

### Tokens (default dark theme — `theme/directions.css`)

| Role | Value |
|---|---|
| Canvas (bg) | `#07090a` |
| Foreground | `#d6e2d8` |
| Muted | `#7f958a` |
| Border (hairline) | `#1e2a28` |
| Accent | `#8ad2e5` (the topic cyan; the card's single accent) |
| Display font | Georgia serif |
| UI font | Inter / system-ui |
| Mono font | ui-monospace (kickers, stats, wordmark) |
| Radius | 0 — **square** (a printed rule, not a rounded card) |

### "Few colours" discipline

Mono palette (canvas / foreground / muted) + **exactly one accent per card**. The accent is spent on
precisely three things: the short **hairline** under the title, the one **live stat** (e.g.
`↑ 2.3× rising`), and the **wordmark dot**. Nothing else is coloured.

### Layout (portrait 1080×1440, padding 96)

```
KICKER            ← mono, muted, tracked, uppercase ("TOPIC" / "EPISODE · CROSS-SHOW")
Title             ← Georgia serif, ~92px, wrapped
──                ← teal hairline (the accent)
"Signature quote" ← italic serif, ~46px, wrapped
— byline          ← Inter, muted (speaker / "42 min · 3 insights")
        (air)
STATS · ↑2.3× RISING   ← mono, muted; the one hot stat in accent   (bottom-anchored)
● closelistening.app   ← mono, muted, + accent dot
```

Generous negative space is deliberate — the empty middle is the "editorial" restraint.

## Mechanism

- **Render:** client `<canvas>` (`composables/entityShareCard.ts`), generalized from the highlight
  card (`useShareCard`). **No new dependency.** The literal hexes live in the `.ts` (canvas needs
  literals; the no-hex-in-components guard only scans `.vue`, and the component omits accent so the
  engine's `DEFAULT_ACCENT` — a token mirror — applies).
- **Bridge-only:** the card carries transcript-derived text + KG metadata only, never audio.
- **Share menu** (`components/ShareMenu.vue`): one affordance → **Share card** (PNG via Web Share →
  download), **Share link** (URL via Web Share → clipboard), **Share text** (caption fallback).

## Built (`feat/player-improvements`)

- **Client engine + Share menu** (`composables/entityShareCard.ts`, `components/ShareMenu.vue`);
  wired on the **entity card** (topic / person / org), the **episode** (PlayerView), the **show**
  (PodcastView) and the **storyline** (StorylineView) — a card model per surface.
- **Per-kind accent** (`accentForKind`): topic cyan, person gold, every other kind the brand cyan
  (holds "few colours"). Token→hex in the `.ts` so no literal hex lands in a `.vue`.
- **Signature quote on the topic entity card:** the leading voice's strongest take
  (`perspectives[0].insights[0]`), fetched best-effort + current-guarded; person/org stay clean.
- **Server OG-image** — a shared LINK now unfurls AS the card, not just an explicitly-shared image:
  - `server/og/card.py` renders the same card to a PNG with **Pillow** (added to core deps) and
    **bundled DejaVu fonts** (`server/og/fonts/`, shipped in the wheel — no OS-font dependency).
  - `server/og/build.py` assembles the card model per kind from the SAME KG builders the
    `/api/app/*` routes use (bridge-only), so the unfurl says what the in-app card says.
  - `routes/app_og.py` serves `GET /og/{kind}/{id}.png` — **unauthenticated** (unfurl bots carry no
    session), outside `/api/app`; the `.png` suffix rides the edge's static rule to the backend.
  - `server/spa.py` (`SpaStaticFiles`) replaces the bare static catch-all: injects `og:*` /
    `twitter:*` into the entity document head (pointing `og:image` at the card), adds the SPA
    history-mode fallback the bare mount lacked (deep links no longer 404 at the backend), and
    preserves real-asset 404s.
  - **No Caddy change:** the edge already reverse-proxies documents + `*.png` to the backend, so OG
    activates at launch when the coming-soon gate is removed (pre-launch everything is coming-soon).

## Per-card content (tuned)

Each kind fills the "lede" slot + footer from the KG so no card is a bare title:

- **Topic** — leading voice's top take as the quote, attributed (`— Name`); `N episodes · M voices`;
  `↑ N× rising` when genuinely rising (from the same `trending` computation the app uses).
- **Person** — web bio one-liner as the blurb (else top co-occurring topics); byline `Host of {show}`.
- **Organization** — org_web (#2035) description as the blurb; `founded {year}` / industry in stats;
  logo as the identity square.
- **Episode** — strongest salience-ranked GI insight as the quote; `N min · N insights`; artwork.
- **Show** — feed description as the blurb; `N episodes · {cadence} · ~N min`; artwork.
- **Storyline** — member topics as the blurb (what it's *about*), `Topics discussed together` as the
  byline (what a storyline *is*), `N topics · N episodes` + trend — self-explaining.

**Artwork** rides as a restrained masthead square top-right (show/episode art, person photo, org
logo), framed to match the hairline — an identity anchor, not a glossy hero. Topic/storyline stay
text-only (no artwork). Undecodable/absent art drops the square silently.

## Not done / next

- **Higgsfield:** reserved for a *whisper* of matte texture/motif at most, at design time only —
  NOT a frame, NOT per-share. Ships frameless and is better for it.
- **Org / storyline standalone pages:** org has no page (overlay-only) so no org LINK to unfurl;
  the OG PNG route supports org for completeness. Storyline has a page and full support.

## Refs

- Rejected concept + chosen previews: `test-results/shots/2036-*` (local, untracked).
- Interaction spec: `docs/uxs/UXS-014-interaction-patterns.md` §Sharing.
