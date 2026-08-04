# closelistening — Curation, Delivery & Moat Architecture (design)

**Status:** DRAFT / discussion — not approved, nothing built from this yet.
**Tracking:** epic #1413; children (a) #1414, (b) #1415, (c) #1416, (d, delegated infra) #1412,
(e) #1417, (f) #1418, (g) #1419.
**Superseded by (canonical):** `docs/prd/PRD-046`, `docs/rfc/RFC-110`, `docs/rfc/RFC-111`,
`docs/adr/ADR-144`, `docs/adr/ADR-145`. This WIP note is now the scratch/thinking layer; the
numbered docs are the source of truth.
**Author:** design session 2026-08-04 (Marko + Claude).
**Frames:** how to go beyond "a player" into curation/management + delivery of interesting
bits, how that establishes a moat vs the podcast-Readwise category (Snipd, Podwise), and
where Obsidian/PKM interop slots in (explicitly the *next* arc, but its export schema is
decided now so this arc doesn't paint us into a corner).

---

## 0. Thesis & moat

**The category is a *feeder*. We are the *graph*.**

Verified competitor read (2026-08-04): Snipd (leader) and Podwise both do
snip/episode → transcript → per-clip AI summary → **sync into the user's *external* PKM**
(Obsidian, Notion, Readwise, Logseq). Snipd's own words: *"Snipd doesn't replace your PKM
system — it feeds into it."* None of them are the connected corpus; they make atoms and
hand the connecting work to the user's Obsidian vault.

closelistening already shipped the layer they don't have (`PRD-041`, v2.7): a **per-user
knowledge corpus** — a read-time projection over the shared GIL/KG ontology — with grounded
cross-episode recall, canonical-identity guest threads, and contradiction surfaces, **on our
own surface, no external PKM required**.

**Moat statement:** competitors sell a highlight reel that you must connect yourself; we sell
a second brain that connects itself and *compounds* as you listen. The clip is the atom
everyone has; the grounded, cross-episode, canonical-identity graph over your clips is the
defensible layer — it requires the whole pipeline (KG extraction RFC-055, canonical identity
RFC-072, hybrid retrieval RFC-090) that we already run and they would have to build from zero.

**Honest limits of the moat (do not oversell):**

- We do **not** win on distribution or integration breadth today. Snipd is on iOS/Android, is
  where people already listen, and has a mature Obsidian/Notion/Readwise export mesh. We are a
  coming-soon-gated web player, single-medium.
- The moat is **depth (the intelligence layer)**, not breadth. Strategy follows: make the
  in-app corpus so good it's the reason to switch, and make every *outbound* surface (digest,
  share card, Obsidian export, MCP) **carry the graph**, not just the clip — otherwise an
  Obsidian export is just a worse Snipd.

**Design rule that falls out of the moat:** every new surface must distribute the *connected
corpus*, never a flat list. A digest resurfaces highlights *linked to their entity/guest/topic*.
A share card names the guest + topic. An Obsidian export emits wikilinked entity notes, not a
flat highlight dump. That is the category-of-one move.

---

## 1. Layer 0 — the shipped substrate (the moat foundation; do not rebuild)

Everything below *distributes* this. It is already live (`PRD-041` / `RFC-101`, v2.7):

| Capability | Where | Note |
| --- | --- | --- |
| Per-user corpus projection (heard ∪ captured, read-time, no rebuilt graph) | `app_user_corpus.py` | `user_episode_set()` = ≥30% played ∪ any capture |
| Capture: moments + spans + insights, colors, notes, Markdown export | `stores/capture.ts`, `HighlightsView.vue`, `PRD-040` | native share path exists (`saveAndShareText`) |
| Grounded recall (no request-time LLM), `scope=mine` | `routes/app_search.py`, `SearchView.vue` | extractive, verbatim, jump-to-moment |
| Cross-episode person/topic threads, `scope=mine` | `routes/app_relational.py`, `EntityCardBody.vue` | canonical identity RFC-072 |
| Spaced resurfacing (2d/1w/1mo/3mo ladder, reflection prompt, pacing) | `app_resurfacing.py`, `ResurfacingInbox.vue` | **read-time only — pull, no delivery** |
| Interest profile (opt-in personalized ordering) | `app_resurfacing.py`, `interests.ts` | flag-gated `rank_discover` |
| Enrichment substrate (temporal velocity, contradictions, grounding rate) | `routes/app_enrichment.py`, `RFC-088` | read-only consumer over envelopes |
| MCP server ("ask your own agent") — **stdio only** | `RFC-095` | HTTP/SSE transport = open question OQ-1 |

**The gap this doc addresses:** the substrate is entirely **pull** (open the app → Revisit tab).
There is **no outbound delivery, no curation/organization layer above raw highlights, and no
graph-aware interop.**

---

## 2. Cross-cutting constraints (bake in from the start)

1. **D6 — no request-time LLM in the core/CI path** (`PRD-041` parked it). Digest, recall,
   share cards, export = **extractive/deterministic**. Generative synthesis happens only via
   the user's *own* agent through MCP — never our server. This is what keeps CI airgapped
   (memory `no_llm_in_ci`). Any "AI intro line" in a digest violates it → out of core path.
2. **Per-user store is file-based JSON + FileLock** (`app_user_store.py`), not Postgres.
   Fan-out jobs (digest for all users) are O(users) directory scans. Fine at current scale;
   flag for materialization later (`RFC-101` OQ-1). Don't design as if a DB query exists.
3. **No consent/email-identity model yet.** `User` = `user_id, email, name, provider,
   subject, disabled, role`. No `email_verified`, no opt-in, no unsubscribe. Delivery needs an
   additive consent schema **first** (§3.1).
4. **Audio hosting is bridge-only** (memory `transcript_vs_audio_hosting`): we never rehost
   audio. Transcript/text share cards are safe. An **audio** share-clip (even 15–30s) is a
   *new legal question* → must be ruled on against `LEGAL.md` / `THREAT_MODEL.md` before build.
5. **New outbound deps need approval** (email provider, any SDK) — AGENTS rule 12. Listed in §6.

---

## 3. Tier 1 — the delivery loop ("the digest that comes to you")

The single highest-ROI move: it converts the *already-built* resurfacing engine into the
Readwise loop Marko actually loves (push, low-effort, habitual). No new intelligence — plumbing
on top of shipped selection logic.

### 3.1 Consent + email identity (prerequisite)

- Extend the per-user profile (additive, back-compat): `email_verified: bool` (Google OAuth
  emails carry a verified claim — trust it), and `comms: { digest: { enabled, cadence:
  weekly|daily, day_of_week, hour, paused }, push: { enabled }, unsubscribe_token }`.
- New `GET/PUT /api/app/comms` + a section in `ProfileView.vue`.
- Unsubscribe = tokenized link, no auth required, one-click (legal requirement).

### 3.2 Personal digest assembler (extractive, D6-safe)

- New `app_digest_personal.py`: assembles a **"Your Week"** payload per user by *reusing*
  `user_episode_set()` + the resurfacing due-selection + interest profile + new-episodes-in-
  -followed-shows + a few **auto-extracted GI picks** (see §3.4). Read-time compute, same model
  as resurfacing. **Not** RFC-068 (that's the corpus-wide operator digest — different scope).
- Payload is structured data (highlights, episodes, threads), rendered to HTML at send time.
  Every item carries its entity/guest/topic links (moat rule).

### 3.3 Delivery channels — **infra owned by issue #1412** (the app↔infra seam)

Delivery *transport* is deferred to an infra agent via **GH #1412**. Split at the seam:

- **App side (this doc / our work):** assemble the extractive payload (§3.2), enforce consent
  (§3.1), and **enqueue a `DeliveryEnvelope` to an outbox**. That's the whole app responsibility.
- **Infra side (#1412):** self-hosted **Listmonk** queue on the homelab (behind the shared Caddy
  edge, like GlitchTip) + **SES** last-mile relay for email reputation + fully self-hosted **Web
  Push** (VAPID) transport + deliverability (SPF/DKIM/DMARC) + retries + bounce/unsub write-back.
- **The seam contract** (frozen; full schema in #1412): channel-agnostic `DeliveryEnvelope`
  `{ id, user_id, channel, template, recipient, consent_snapshot, payload, not_before }`; app
  exposes `GET /internal/outbox/pending` + `POST /internal/outbox/{id}/status`; suppression writes
  back to the §3.1 consent store. App and infra join **only** here.
- **Decision (2026-08-04):** self-host the queue/manager, use a service (SES) for last-mile
  reputation — do not direct-send from a homelab/VPS IP.
- **Scheduler** (app side) — extend the existing in-process APScheduler (`scheduler.py`, today
  feed-sweep-only) with a per-user digest cron that iterates profiles, assembles, and **enqueues
  envelopes**. O(users) scan — acceptable now, materialize later.

### 3.4 Auto-highlights seed (kills the cold-start + is a moat feature)

- Snipd requires you to tap-to-snip *in the moment* (active listening + memory). We already
  **auto-extract** key moments via GI. Seed the resurfacing pool + digest with **"editor's-pick"**
  moments from episodes the user heard but didn't manually capture.
- Effect: the corpus and the digest have value on day 1 with zero manual capture — a structural
  advantage the tap-to-snip category cannot match. Mark auto-picks distinctly from user captures.

---

## 4. Tier 2 — curation & management ("beyond player")

The "management and curation of interesting bits" ask. Additive per-user surfaces, no ML.

### 4.1 Collections / boards

- User-defined named sets of highlights/moments spanning episodes. New per-user store +
  `GET/POST/DELETE /api/app/collections`, a highlight ↔ N collections join. This is the active
  *organization* layer that turns a flat highlight list into curated knowledge.

### 4.2 Shareable clip cards (outward growth loop)

- **Text/transcript card first (safe):** quote + guest + topic + episode + timestamp rendered
  to a share image; reuse the existing native share path. Names the guest/topic (moat rule).
- **Audio snip card (gated):** a few-seconds audio clip — **blocked on the audio-hosting ruling
  (§2.4).** Do not build until ruled on.
- This is the one outward/viral surface the whole category under-serves for audio.

### 4.3 Highlights → graph nodes

- Link each highlight to its KG entity/topic so a saved quote becomes a queryable node in recall,
  not just text. Partially there (insights captured); finish the wiring. Reinforces the moat and
  is the substrate the Obsidian export (§5) serializes.

---

## 5. Tier 3 / NEXT ARC — interop (Obsidian/Notion + MCP remote)

**Decided now, built next arc.** The schema decisions here constrain Tier 1/2, so lock them now.

### 5.1 Graph-aware export (the Snipd-differentiator)

- **Decision: export the connected corpus, not flat highlights.** Emit Markdown where each
  highlight note wikilinks to `[[Entity]]`, `[[Guest]]`, `[[Topic]]`, `[[Episode]]` notes — the
  user's Obsidian vault becomes a *mirror of their personal KG*, not a highlight dump. Snipd's
  Obsidian plugin exports flat highlights + summary; a wikilinked entity graph is category-of-one.
- **v1 = one-way pull**, incremental: `GET /api/app/export?format=obsidian&since=<cursor>`
  returns a bundle of Markdown files + a manifest. Two-way sync is explicitly a later, bigger arc.
- **This is why Tier 2 §4.3 must land first** — the export serializes the highlight↔entity links.

### 5.2 MCP remote transport (BYO-agent north-star)

- `RFC-095` shipped stdio; remote/HTTP-SSE is OQ-1. "Ask *your* Claude about everything you've
  heard" for remote users needs the HTTP/SSE transport. It's the D6-safe generative path (the
  model is the user's, never ours). Same moat face as export: *your corpus is portable and
  queryable by your own tools.*

---

## 6. Decisions (resolved 2026-08-04)

1. **This-arc scope → Delivery + curation core.** Full §7 slice below.
2. **Delivery → both channels this arc (Web Push + email).** Email provider still to pick
   (one open sub-decision — recommend **Resend**: simple REST, generous free tier, good
   deliverability, minimal SDK footprint; needs final nod as a new runtime dep, AGENTS rule 12).
3. **Audio share-clips → text cards only.** §4.2 audio-snip card is **out of this arc**; ship
   transcript-quote cards only (safe under bridge-only audio). Revisit audio later.
4. **Digest cadence** — still open; default proposed **weekly (Sun AM)** with a daily option in
   settings. Confirm at build time.

---

## 7. Sequencing (committed)

- **This arc (delivery + curation core):** §3.1 consent → §3.3 **Web Push + email** (both this
  arc) → §3.2 personal digest assembler → §3.4 auto-highlight seed → §4.1 collections → §4.2
  **text** share cards → §4.3 highlight↔entity wiring.
- **Next arc (interop):** §5.1 graph-aware Obsidian export → §5.2 MCP remote. Export schema
  (§5.1) is frozen *now* so §4.3 serializes to it.
- **Gated / separate:** §4.2 **audio** share cards (deferred this arc), full two-way sync, real
  SRS tuning (`PRD-041` non-goal), request-time LLM digest intro (`D6`).

---

## 8. NOT covered / open questions (equal weight — T6)

- **Not validated with usage data.** "Highest-ROI" is a product-logic claim; there is no
  analytics read behind it. Could be wrong about what users want first.
- **Digest assembler reuse is asserted, not prototyped.** I claim it reuses resurfacing
  selection + `user_episode_set()`; I have not written a spike proving the payload assembles
  cleanly read-time at acceptable latency for a weekly all-user fan-out on the file store.
- **Web Push feasibility not verified end-to-end.** PWA exists (`PwaUpdateToast`), but I have
  not confirmed VAPID keys / push subscription storage / iOS-PWA push support in this codebase.
- **Consent/legal surface under-explored.** I have not read `LEGAL.md` / `THREAT_MODEL.md` for
  the actual consent + retention requirements email delivery imposes; §3.1 is a sketch.
- **Obsidian export schema is a proposal, not designed against a real vault.** Wikilink shape,
  entity-note dedup across exports, and conflict-on-re-export are unspecified.
- **MCP remote transport effort unknown.** OQ-1 is open in `RFC-095`; I have not scoped HTTP/SSE.
- **Competitor read is a point-in-time snapshot** (2026-08-04 web search); Snipd/Podwise ship
  fast and may already have a graph feature I didn't surface.
- **No cost model.** Email volume, push infra, share-image rendering compute — unpriced.
