# Incident reviews

Post-incident reviews (PIRs) for production-affecting events. Modeled on
Google SRE's blameless postmortem practice — the goal is **transparency
and closed-loop learning**, not assigning fault.

## When to write one

Write a PIR for any event where **at least one** of these is true:

- Prod was unreachable, degraded, or producing wrong data for > 15 min.
- Data was lost, corrupted, or required restoration from backup.
- Recovery required operator intervention beyond running a documented workflow.
- An on-disk or in-state divergence required manual reconciliation.
- A "near miss" — a destructive action was triggered but stopped before impact — and the lesson is non-obvious.

Skip PIR for:

- Routine CI failures and their fixes (those belong in commit messages / PR descriptions).
- Single-flake test failures.
- Local-dev environment issues.

## Structure (three phases)

Every PIR follows three sequential phases. **Write them in order**;
don't analyze before gathering facts, and don't plan before analyzing.

1. **Facts** — timestamped timeline of what happened. No judgment, no
   "should have", no root-cause speculation. Just events. If you can't
   timestamp something, leave it out or annotate clearly.

2. **Analysis** — root cause + contributing factors + why detection and
   recovery took as long as they did. This is where speculation,
   interpretation, and "if X had been in place" belong. Be honest about
   what *didn't* break that easily could have (counterfactuals).

3. **Improvement plan** — action items with owners, tracking issues
   (GH issues / PRs), and target dates. Categorize as:
   - **Prevention** — would have stopped this happening at all.
   - **Detection** — would have surfaced the problem sooner.
   - **Mitigation** — would have reduced impact / recovery time.
   - **Process** — would have changed how we respond.

## Blameless principle

Names of operators / agents involved are fine to include — they're
useful context. Judgments about competence are not. The framing should
always be "what about the system let this happen" or "what about the
process didn't catch it", never "X made a mistake."

If a step was triggered by an agent (this codebase uses Claude Code +
similar tooling), describe it the same way: "the agent triggered X
without explicit approval" is factual; "the agent shouldn't have done
that" is process commentary that belongs in the improvement plan
(e.g. "AGENTS.md needs a rule to prevent this").

## Files

- [TEMPLATE.md](TEMPLATE.md) — copy this to start a new PIR.
- Per-incident files: `INCIDENT-YYYY-MM-DD-<short-slug>.md` (sorts
  chronologically; the slug should be 2-4 words describing the
  surface, e.g. `prod-rebuild-cascade` or `corpus-restore-blocked`).

## Process

1. Create file: `cp docs/incidents/TEMPLATE.md docs/incidents/INCIDENT-YYYY-MM-DD-<slug>.md`
2. Fill out Phase 1 (Facts) while details are fresh. **Same day or next day** ideally.
3. Phase 2 (Analysis) and Phase 3 (Improvement plan) can land in a
   follow-up commit if the immediate priority is recovery.
4. Open a PR for the PIR. Reviewer's job: spot-check facts, push back
   on speculation in the Facts section, suggest missing improvement
   items.
5. Track improvement items as GH issues with the `incident` label so
   they show up in the v2.7+ planning view.
6. Once all improvement items are landed or explicitly de-scoped,
   update the PIR's `Status:` field to `closed` with the close-out
   date.

## Cross-references

- [PROD_RUNBOOK § Disaster recovery](../guides/PROD_RUNBOOK.md#disaster-recovery) — what to do during an incident.
- [RELEASE_PLAYBOOK § Phase 8](../guides/RELEASE_PLAYBOOK.md) — release verification (catches a class of issues before they become incidents).
- [AGENTS.md "RULES YOU KEEP BREAKING"](https://github.com/chipi/podcast_scraper/blob/main/AGENTS.md) — agent-side rules accumulated from prior incidents.
