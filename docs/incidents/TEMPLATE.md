# INCIDENT-YYYY-MM-DD — <short title>

| Field | Value |
| --- | --- |
| Date | YYYY-MM-DD |
| Duration | HH:MMz–HH:MMz UTC (~Nh total) |
| Severity | SEV-1 / SEV-2 / SEV-3 / near-miss |
| Affected services | e.g. prod podcast viewer, prod API, both apps on prod-podcast-1 |
| Author(s) | name (operator), agent (if applicable) |
| Status | draft / final / closed |
| Last updated | YYYY-MM-DD |

## Summary

Two sentences max. What happened, what was the user-visible impact, how was it resolved.

## Impact

- **Customer-facing**: yes / no — describe what end users saw if applicable.
- **Data lost or corrupted**: yes / no — describe scope. Quantify recovery point if relevant (e.g. "restored from snapshot dated YYYY-MM-DD; ~N days of corpus rewritten").
- **Time to detect (TTD)**: from underlying cause arising to operator/agent noticing something was wrong.
- **Time to resolve (TTR)**: from detection to prod fully restored.
- **Time on incident response**: total active operator/agent time (not just wall-clock).

---

## Phase 1: Facts (timeline)

Timestamped events. Factual only. UTC.

| Time (UTC) | Event | Source |
| --- | --- | --- |
| HH:MM | What happened | log / commit / run id / etc. |
| HH:MM | … | … |

**No interpretations, no "should haves" in this section.** If you don't know the timestamp for sure, leave it out or write `~HH:MM (approx)`.

---

## Phase 2: Analysis

### Root cause

Be specific. Cite the exact commit, line, config value, or event that caused the incident. If multiple causes contributed equally, list them.

### Contributing factors

Conditions that didn't cause the incident on their own but amplified its impact or slowed recovery. Each should be addressable.

### Why detection took as long as it did

What signals were available before the incident manifested? Why weren't they noticed? Was there no monitoring at all, or was the signal there but lost in noise?

### Why recovery took as long as it did

Walk through the actual recovery as it happened. Where did time go? What information was missing? What documented procedures were absent or wrong?

### Counterfactuals (what didn't break that could have)

Where did luck save us? What backup, what alternate path, what existing safeguard prevented this from being worse?

---

## Phase 3: Improvement plan

Actionable items only. Each item has a tracking issue + owner + target date.

### Prevention (would have stopped this happening)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| … | #N | name | vX.Y / YYYY-MM-DD |

### Detection (would have surfaced the problem sooner)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| … | #N | name | vX.Y / YYYY-MM-DD |

### Mitigation (would have reduced impact / recovery time)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| … | #N | name | vX.Y / YYYY-MM-DD |

### Process (would have changed how we respond)

| Item | Tracking | Owner | Target |
| --- | --- | --- | --- |
| … | #N | name | vX.Y / YYYY-MM-DD |

---

## What went well

Be honest — there's always something. Backup that worked. Rule that fired. Workflow that did its job. Naming these reinforces what to preserve.

## What went wrong

Blameless framing. "X tool produced Y outcome under Z condition." Not "person did wrong thing."

## Lessons learned

Meta-observations that don't fit cleanly in the improvement plan — things about how the system is shaped, how recovery feels, what kinds of incidents are likely next. These inform future architectural decisions and shouldn't be lost.

---

## References

- Linked commits / PRs / runs
- Related GH issues
- PROD_RUNBOOK sections touched
- Prior PIRs with related themes
