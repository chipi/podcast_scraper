# Multi-tenant platform sketch — catalog, shared processing, UI (WIP)

**Status:** Architecture / product sketch — **not** a PRD or RFC. Captures how to think about
evolving from the **single-user CLI + per-run `Config`** model toward a **small server + UI**,
**continuous ingestion**, **Postgres**, and **many users** with **shared pipeline work**.

**Audience:** Future PRD/RFC authors and implementers.

---

## Goal (target picture)

- A **small server** holds **configuration**; the primary object is a **list of podcasts (feeds)**
  the system should process — not one static YAML per machine.

- The system **continuously** pulls and processes subscribed content.
- A **small UI** lets users **manage** which shows they care about.
- **Many tenants:** processing is **central and deduplicated**; users only **see** slices of
  canonical data (summaries, GI, KG projections) they are **entitled** to.

**Design intent:** Shape the **data model and boundaries** for multi-tenancy **from the start**,
even when **v1** runs a **single** tenant — avoid a painful “add `tenant_id` everywhere”
migration later.

---

## Product constraints (CLI + optional platform)

These are **non-negotiable directions** for how the platform relates to the existing tool:

1. **CLI stays first-class** — Easy runs from **config** (YAML/JSON): scripts, CI, ad-hoc use,
   outputs on disk **without** running a server. Most users can stay on this path forever.

2. **Service / platform mode is optional** — Long-lived process for **continuous** pull,
   multi-feed orchestration, Postgres projection, optional **UI/API**. Headless single-user

   platform and multi-tenant UI are **graduations** of the same service layer, not a separate
   product fork.

3. **One pipeline core, multiple shells** — Execution stays **`run_pipeline` / `service.run`**
   (one `Config` in, artifacts out). Platform code **schedules**, **dedupes**, **materializes**

   `Config` per job, handles **cursors** and **projection** — it **must not** duplicate or
   diverge from core pipeline logic.

4. **Avoid diverging config dialects** — Prefer **one `Config` model** and shared validation;
   the platform adds **catalog, subscriptions, tenants** in the DB and **builds** a `Config`

   (or equivalent dict) for each worker invocation instead of maintaining a parallel
   “platform-only” config schema that drifts from the CLI.

**Summary:** Library + **CLI** (simple path) + optional **service** (daemon / API / UI) on the
same engine — not “CLI tool vs platform” as two implementations.

---

## Separate three concepts

| Concept | Meaning |
| --- | --- |
| **Catalog** | Global **directory of feeds** (and normalized show metadata) the **platform** knows. This is “what exists / what we can process,” not “Alice’s list” alone. |
| **Subscription** | **Tenant** ↔ **catalog entity** (e.g. feed). “I want this show in my library.” |
| **Entitlement / visibility** | What the tenant may **read**: which episodes, GI rows, KG nodes/edges, summaries. Often: subscription + optional admin grants or feature flags. |

**Ingestion** should be driven by **catalog + policy** (what is globally enabled, what has
subscribers), not by “run this user’s private config file” as the only source of truth.

---

## Process once, serve many (reuse)

**Idea:** Expensive steps (download, transcribe, summarize, GI, KG) run **once** per logical
episode + **pipeline fingerprint**, not once per user.

- **Episode identity:** Canonical feed id + episode `guid` (aligns with universal episode
  identity — ADR-007).

- **Pipeline fingerprint:** Versions / hashes of models, providers, prompts, schemas that
  affect outputs (summaries, `gi.json`, KG artifact).

**Rule:** At most **one canonical artifact set** per `(episode_key, pipeline_fingerprint)`.
If tenant B subscribes to the same feed+episode, they **attach** to existing rows — no second
Whisper run for the same fingerprint.

**Workers** should consume a **deduped queue** of “work needed,” not “user A’s job” vs
“user B’s job” as independent full pipeline runs.

---

## Storage layout (conceptual)

- **Canonical blob/object paths** for transcripts, metadata, GI, KG — **not**
  `users/<tenant>/...` as the primary store for pipeline outputs.

- **Postgres** (or metadata DB) holds: catalog, subscriptions, cursors, job state, **pointers**
  to blobs, and **projected** tables (PRD-018 / RFC-051 style) keyed by **global episode and
  artifact ids**.

**Tenant-specific** data in v1+: preferences, notes, stars — **separate** tables with
`tenant_id`, not duplicate transcripts.

---

## Multi-tenancy model (Postgres)

**Default assumption:** **Single database**, **shared tables**, **`tenant_id`** on every
tenant-scoped row, plus **Row Level Security (RLS)** for defense in depth.

- **v1 single user:** Still insert one row in `tenants` (e.g. `default`) and always set
  `tenant_id` in API and queries.

- **Catalog** (`feeds`, `episodes`, canonical artifact registry): often **global** (no tenant).
- **Subscriptions** (`tenant_feed_subscriptions`): `tenant_id` + `feed_id`.
- **Reads:** “My library” = join subscriptions → episodes → canonical projections; enforce
  RLS or equivalent in the API layer.

Avoid **schema-per-tenant** early unless compliance forces it (higher ops burden).

---

## Service shape (logical components)

Can start as **one deployable** (e.g. monolith + workers) and split later:

1. **API + UI** — Admin: manage catalog. User: manage subscriptions, browse entitled data.
2. **Scheduler / worker** — Select feeds that need polling (≥1 subscriber or globally enabled),
  enqueue work, respect rate limits, update **cursors**.

3. **Projection / indexer** — Canonical files or object storage → Postgres (RFC-051); optional
   search index later.

**Pipeline core:** Keep **`workflow.run_pipeline`-class logic** as “**execute one unit of work**”
(config in, artifacts out); the **platform** wraps it with **queue, dedup, tenant-agnostic
storage paths**, and **post-run projection**.

---

## GI / KG / summaries under reuse

- **Store once** per `(episode, pipeline_fingerprint)` in **canonical** storage + projection
  tables.

- **Tenant view:** Filter by **subscription** (and entitlements). No duplicate `gi.json` per
  user for the same fingerprint.

- **Per-tenant overlays** (bookmarks, labels) live in **tenant-scoped** tables.

---

## Phased delivery (compatible with “design complex early”)

| Phase | Deliver |
| --- | --- |
| **A** | `tenant_id` + `tenants` table; subscription table; catalog table; single tenant row. |
| **B** | Long-lived worker + cursor table; feeds driven by catalog ∩ subscriptions. |
| **C** | Projection to Postgres (RFC-051); API reads from DB + blob pointers. |
| **D** | UI for catalog + subscriptions; auth (even if one login). |
| **E** | Second tenant + RLS verification; quotas / billing later. |

Defer: billing, orgs/teams, per-tenant pipeline overrides, legal/compliance review — but
**name** shared-corpus and ToS risks in a future threat model.

---

## Risks / watch

- **Pipeline fingerprinting** must be correct or tenants share **wrong** or **stale**
  artifacts.

- **Operational surface:** Postgres + workers + object storage + auth + UI — scope consciously.
- **Content / legal:** Multi-user + shared corpus may imply takedown, copyright, and log/PII
  policies — non-technical but real.

---

## Relation to current repo

- Today: **one `Config`**, one **`rss`**, **`run_pipeline`** → **filesystem** truth,
  **service.run** is **one-shot** per invocation. **CLI** remains the default mental model.

- This sketch: **catalog + subscriptions** complement (not replace) config-driven CLI — they
  replace “YAML is the **only** source of feeds” **when** the operator chooses platform mode;

  **workers** replace “cron per feed”; **Postgres** holds **state + projections**;
  **reuse** is explicit via **episode key + fingerprint**.

---

## Related documents

- [Platform deployment: API, workers, queues, Docker (WIP)](platform-deployment-api-worker-queue-sketch.md)
  — Compose services, Redis/queues, Whisper/`heavy` queue, worker topology, RFC checklist.
- [Architecture](../ARCHITECTURE.md) — one pipeline, one `Config` (today); platform wraps it.
- [ADR-007: Universal episode identity](../adr/ADR-007-universal-episode-identity.md)
- [PRD-018: Database projection](../prd/PRD-018-database-projection-gil-kg.md) / RFC-051
- [Corpus digest & weekly rollup (WIP)](corpus-digest-weekly-rollup-product-idea.md) —
  downstream product on top of queryable corpus

---

## Promotion

When hardened: split into a **PRD** (product / tenancy / UI) and **RFC** (schema, worker
protocol, projection, API). Until then, **WIP** only.
