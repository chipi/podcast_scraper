# Cluster sibling episode load (plan iteration)

**Status:** Design locked for implementation phase (2026-04-15 iteration).

## Clarification: what the earlier “options” meant

Three **UX triggers** were discussed for *when* to pull in sibling episodes:

| Option | Meaning |
|--------|--------|
| **Button only** | User clicks something like “Load cluster episodes” — nothing loads until they click. |
| **Confirm dialog** | After an action (e.g. Open in graph), a dialog: “Load N related episodes?” Confirm / Skip. |
| **Automatic with cap** | The app loads siblings without a click, but **stops at a max** (configurable) to avoid huge graphs by accident. |

## Locked product decision (this iteration)

- **Trigger:** **Automatic** in the **Graph** context: when a topic cluster applies and sibling episodes can be resolved from `topic_clusters.json` + catalog, **merge-load** their GI/KG into the current graph selection up to a **safety cap** on how many **additional** episodes to pull in per merge (not unlimited).
- **Safety cap — default `10`:** At most **10** sibling episodes are auto-merged per trigger (tune if needed). Order: deterministic (e.g. catalog sort, or first N unresolved ids) — specify in implementation.
- **Configurable “higher up”:** One central place (e.g. viewer env `VITE_CLUSTER_SIBLING_EPISODE_CAP` with default `10`, or a small `viewerConfig` module parsing `import.meta.env` + fallback). Document in Development / Polyglot guide and optional `.env.example` so power users can raise the cap without code edits.
- **Scope:** **Graph / artifact merge path only** for this phase — **no change** to **Digest** or **Library** behavior or layout (those can be revisited later if we want parity).
- **Transparency:** When the cap trims the list, UI should indicate **“Loaded N of M sibling episodes (cap …)”** or similar so users know more exist.

## Is this a “good” answer?

**Yes.** A default of **10** with a **configurable** cap balances full-cluster intent with predictable worst-case load. Raising the default or env for large monitors / fast machines is easy; lowering it protects laptops and huge corpora.

**Technically:** Same as before: `episode_ids` on cluster members, resolve to paths via catalog, **merge** (not replace) `selectedRelPaths`, apply cap when selecting which sibling paths to add.

## Implementation reminders (unchanged core)

1. **Store:** `appendRelativeArtifacts` / merge load — required because `loadRelativeArtifacts` replaces selection today.
2. **Server:** Resolve `episode_id` → `gi_relative_path` / `kg_relative_path` (catalog scan per request or cached server-side if needed).
3. **Client:** From loaded graph + `topicClustersDoc`, compute sibling episode ids → resolve → merge load **automatically** when conditions are met (define exact hook: e.g. after `loadSelected` completes when graph tab is active and cluster has unresolved siblings).

## Out of scope (this phase)

- Digest “open in graph” flows.
- Library episode detail / “Open in graph” (unless we later wire the same merge helper behind an explicit control there).

## See also (layout, separate workstream)

- [`wip-graph-layout-topic-cluster.md`](wip-graph-layout-topic-cluster.md) — tighter TopicCluster footprint on the main canvas; neighborhood minimap COSE/2D instead of breadthfirst strip.
