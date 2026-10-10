# Handover — player 1.0.4 branch (2026-10-10)

Branch `feat/player-1.0.4`, no PR yet. Living document: updated as each open item lands.

## Moments now matches the Brief

- Moments draws on exactly the insights the Brief shows (`app_moments.score_candidates` →
  `insights_from_gi(..., unattributed_fallback=True)`). Naming no longer decides whether an
  episode has moments; an unnamed moment shows no speaker.
- The Brief's fallback (#2198) no longer fails because of one quoteless insight: an ungrounded
  insight has no speaker, so it does not count as "named" (`app_gi_view.insights_from_gi`).
- Every "▶ Moments" door reads `has_moments` (episode payloads) / `episode_has_moments` (search
  hits), the reel's own rule, computed by `app_moments.episode_has_moments` and cached per GI file
  (~1 ms cold per episode).
- Measured on a read-only copy of the prod corpus (2,958 served episodes): Brief vs Moments
  disagreed on 289 episodes before, 2 after. Neither of the 2 has a time on disk.

## For someone else to fix: the insights upstream of Moments

**#2304** has the full root-cause analysis:
- 230 episodes where speaker naming resolved nobody (212 have an unnamed `host` voice; 62 have the
  guest's name in metadata but never placed on a voice);
- 3 where the value gate made every insight tier 0;
- 1 transcript with no timestamps;
- 1 episode whose shown insights are all quoteless.

## Open on this branch

- Limitation 2: speaker- and topic-scoped search still filters after ranking.
- Limitation 3: prod's search index has no time on transcript passages (read-time workaround only).
- Limitation 4: word forms / misremembered words found 89–92% of the time.
- The phone still needs the newest dev build.
- Prod duplicate-milestone cleanup needs operator OK.
