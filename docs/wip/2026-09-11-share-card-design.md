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

## Built (v1, `feat/player-improvements`)

- Engine + Share menu; wired on the **entity card** (topic / person / org) and the **episode**
  (PlayerView), the episode using its strongest salience-sorted insight as the quote.

## Not done / next

- **Surfaces:** show + storyline (engine is entity-agnostic — just a model per surface).
- **Per-kind accent:** resolve the kind token (`--lp-topic` / `--lp-person` / …) to a hex at render
  time, instead of the single cyan, if we want card-type colour identity (weigh vs "few colours").
- **Signature quote on the entity card:** the topic card omits it in v1 (perspective data isn't in
  the shell); pull the top perspective insight.
- **Server OG-image:** entity-page `og:image` = the card, so a shared **link** unfurls AS the card
  (the growth loop). Storyline/org have no standalone page yet — decide page vs deep-link.
- **Higgsfield:** reserved for a *whisper* of matte texture/motif at most, at design time only —
  NOT a frame, NOT per-share. v1 ships frameless and is better for it.

## Refs

- Rejected concept + chosen previews: `test-results/shots/2036-*` (local, untracked).
- Interaction spec: `docs/uxs/UXS-014-interaction-patterns.md` §Sharing.
