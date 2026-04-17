# CIL Bridge Autoresearch Plan

Validate the Canonical Identity Layer bridge that merges GI and KG identities
per episode. The bridge is the layer between extraction (GI + KG) and
clustering/search (topic clusters + CIL queries).

**Depends on:** GI (#579) and KG (#584) quality validated first.

**Related:** RFC-072 (CIL), #580 (Gemini long labels + empty clusters),
`docs/wip/TOPIC_CLUSTERING_AUTORESEARCH_PLAN.md`

---

## How the bridge works (from code analysis)

`src/podcast_scraper/builders/bridge_builder.py`:

1. Iterates all nodes in GI artifact, then all nodes in KG artifact
2. Strips layer prefixes (`g:`, `k:`, `kg:`) but keeps CIL prefixes
   (`person:`, `org:`, `topic:`)
3. **Exact ID match** to merge — if `person:john-smith` appears in both GI
   and KG, they merge into one identity record
4. Aliases merged via set union; display names pick longer string
5. Sources tracked: `{"gi": true, "kg": true}` per identity

**No fuzzy matching.** No embedding similarity. No name variant detection.
The bridge is deterministic and fast but brittle to ID inconsistencies.

---

## Known failure modes

| Failure | Example | Impact |
|---------|---------|--------|
| **Slug inconsistency** | GI: `person:john-smith`, KG: `person:john_smith` | Same person = 2 identities |
| **Case sensitivity** | `person:JohnSmith` vs `person:johnsmith` | Same person = 2 identities |
| **Name variants** | "John Smith" (GI) vs "J. Smith" (KG) | Different slugs → not merged |
| **Unicode** | `person:josé-garcía` vs `person:jose-garcia` | Accent stripping differences |
| **Topic label style** (#580) | GI: sentence-slug, KG: noun-phrase-slug | Same topic = 2 nodes |

All of these produce **false negatives** (missed merges), not false positives.
The bridge never incorrectly merges unrelated identities — it only fails to
merge related ones when IDs differ.

---

## What to validate

### 1. Identity merge rate on real corpus

On a 5-10 episode corpus with both GI and KG artifacts:
- How many identities appear in **both** layers (`gi=true AND kg=true`)?
- How many are single-layer only?
- For single-layer identities: are any of them the SAME real-world entity
  that got different IDs? (manual spot-check, ~20 identities)

**Metric:** merge rate = identities_in_both / total_unique_identities.
Higher = better. If most identities are single-layer, the bridge is not
doing much useful merging.

### 2. Slug consistency audit

Compare how GI and KG generate slugs for the same entity:
- Take 10 people who appear in both summary (→ GI) and transcript (→ KG)
- Check: do they get the same `person:slug` in both layers?
- If not: what's the difference? (hyphen vs underscore, case, unicode, etc.)

**This is the root cause investigation.** If slugs are consistent, the bridge
works. If they're not, we fix the slug generation, not the bridge.

### 3. Topic ID alignment post-KG optimization

After switching to `kg_extraction_source="provider"` (per KG autoresearch):
- Do KG topic IDs (noun-phrase slugs) align better with GI topic IDs?
- Does the merge rate improve?
- This directly tests whether #580's root cause (sentence-length KG labels)
  was causing topic ID mismatches in the bridge.

### 4. CIL query correctness

On a corpus with bridge artifacts:
- Run `position_arc(person, topic)` for known person+topic pairs
- Run `person_profile(person)` for known hosts/guests
- Run `topic_timeline(topic)` for known topics
- Verify results include ALL expected episodes (no missed merges)

### 5. Alias chain resolution

Test `resolve_id_alias()` with:
- Single-hop aliases (A→B)
- Multi-hop chains (A→B→C)
- Cycles (A→B→A) — should terminate at 16 hops
- Mixed manual + auto aliases from cil_lift_overrides.json

---

## Suggested experiment order

```
1. Generate GI + KG artifacts on 5 held-out episodes
   (use provider mode for both, n=12 insights + n=10 topics)
2. Build bridge.json per episode
3. Measure merge rate + single-layer counts
4. Manual spot-check 20 identities for missed merges
5. If slug inconsistencies found → fix slug generation
6. Rerun bridge → remeasure
```

---

## Connection to topic clustering

The bridge is INPUT to topic clustering:
- Bridge identities include `topic:` nodes from both GI and KG
- Topic clustering operates on KG topic vectors (FAISS)
- CIL aliases from clustering merge back into `cil_lift_overrides.json`

**Order:** validate bridge first (are the right topics being identified?),
then validate clustering (are similar topics being grouped correctly?).

---

## Estimated effort

- Generate GI + KG artifacts: already have silver refs; need pipeline runs (~30 min)
- Build bridges: automatic in pipeline
- Merge rate measurement: ~15 min scripting
- Manual spot-check: ~30 min human time
- **Total: ~1-2 hours, no API cost** (bridge is local computation)
