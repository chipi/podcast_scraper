# Handover — Enrichment 3rd-tab + session state

**Date:** 2026-07-01
**Branch:** `feat/consumer-remember` (NOT pushed; ~12 commits ahead of the
last push point — see `git log`)
**Repo/worktree:** `podcast_scraper-ai-ml-improvements`
**Dev server:** viewer HMR is UP on `:5173` (vite). Player + backend via
`make serve` (NEVER `npm run dev` — needs backend for `/api`).

---

## 1. Immediate state — what's uncommitted RIGHT NOW

`git status --short`:

```
 M web/gi-kg-viewer/src/components/episode/EpisodeDetailPanel.test.ts
 M web/gi-kg-viewer/src/components/episode/EpisodeDetailPanel.vue
 M web/gi-kg-viewer/src/components/graph/NodeDetail.vue
 M web/gi-kg-viewer/src/components/shell/SubjectRail.vue
?? docs/wip/enrichment-visual-inspection-plan.md   (earlier, uncommitted)
?? web/gi-kg-viewer/src/components/graph/NodeEnrichmentSection.vue  (NEW)
?? docs/wip/handover-enrichment-tabs-session.md     (this file)
```

**This work is DONE + verified green, awaiting the operator's visual check
before commit.** Do NOT commit until the operator confirms they eyeballed all
3 cards. When they say go: one commit, e.g.
`feat(#1128): enrichment as a third subject-card tab (topic/person/episode)`.

### What the task was
Add an **"Enrichment" third tab BETWEEN "Details" and "Neighbourhood"** on the
graph subject cards, for all 3 subject types. This is an inline third tab —
NOT a link to a "full profile", NOT a 2-tab restructure. (The operator burned
me once for building a 2-tab Details/Enrichment restructure of
`TopicEntityView.vue` — that was reverted. **DO NOT touch `TopicEntityView.vue`
or `PersonLandingView.vue`** for this.)

### What landed
- **Topic + Person** = graph **node cards** → `NodeDetail.vue`
  (`GraphRailDetailTab = 'details' | 'enrichment' | 'neighbourhood'`, tab button
  `node-detail-rail-tab-enrichment`, panel mounts new
  `NodeEnrichmentSection.vue`).
  - `NodeEnrichmentSection.vue` (NEW): topic → `temporal_velocity` +
    `topic_cooccurrence_corpus` (velocity chip + co-occurs chips →
    `subject.focusTopic`); person → `grounding_rate` + `guest_coappearance` +
    `nli_contradiction` (grounding %, co-appears chips → `subject.focusPerson`,
    disagrees-with rows). Uses `fetchCachedCorpusEnvelope`. Best-effort: missing
    envelopes silently hidden.
- **Episode** = episode rail → `SubjectRail.vue` (tablist) +
  `EpisodeDetailPanel.vue` (panels).
  - `SubjectRail.vue`: `EpisodeSubjectDetailTab` now includes `'enrichment'`;
    tablist `<nav>` **always renders** (was gated on
    `episodeSubjectNeighbourhoodEnabled`); Enrichment button
    (`episode-detail-rail-tab-enrichment`) sits between Details and
    Neighbourhood; **Neighbourhood button is now `v-if`-gated** on
    `episodeSubjectNeighbourhoodEnabled`.
  - `EpisodeDetailPanel.vue`: `railDetailTab` type +`'enrichment'`; new computed
    `railTabsEnabled = props.railNeighbourhoodEnabled || slot present`
    (`useSlots()`); `EpisodeEnrichmentSection` **moved out of the Details panel**
    into a new `#episode-detail-rail-panel-enrichment` panel
    (`v-show="!railTabsEnabled || railDetailTab === 'enrichment'"`). Details/enrichment
    a11y (role/aria/tabindex) now key off `railTabsEnabled`, not
    `railNeighbourhoodEnabled`.

### Tab-count logic (operator's explicit requirement)
- **Graph** tab → **3 tabs**: Details · Enrichment · Neighbourhood.
- **Digest / Library** tabs → **2 tabs**: Details · Enrichment (Neighbourhood
  hidden, since there's no graph-connections id off the graph tab).

### Verification (already run, all green — do NOT re-run unless you changed it)
- `cd web/gi-kg-viewer && node_modules/.bin/vue-tsc -b` → exit 0.
- `node_modules/.bin/vitest run` on SubjectRail / EpisodeDetailPanel /
  NodeDetail / RFC088ChunkNineSurfaces → **84/84 pass**. (Added a prop-driven
  test: "renders the enrichment panel as a tabpanel on the enrichment rail
  tab".) The `ECONNREFUSED :3000` lines in vitest output are unrelated jsdom
  fetch noise, not failures.

### Operator's live-check script (they'll do this)
1. Topic — graph tab → click a topic node → Enrichment tab.
2. Person — graph tab → click a person node → Enrichment tab.
3. Episode — open an episode subject → Enrichment tab (also on digest/library).

---

## 2. Gotchas that bit this session (keep them in mind)

- **Editing viewer `.vue` files:** native `Edit` gets stuck ("File has not been
  read yet") because the running dev server touches the files mid-edit. Use
  `mcp__lean-ctx__ctx_edit(path, old, new)` for viewer files instead.
- **vitest cwd:** always `cd web/gi-kg-viewer` first; use
  `node_modules/.bin/vitest` (NOT `npx`). `--prefix` does NOT change cwd.
- **`$pipestatus` not `$PIPESTATUS`** in zsh (this shell). `vue-tsc -b` prints
  nothing on success — no output == clean.
- **Local `make serve` OpenMP crash** (fixed this session, commit `382c8d5e`):
  torch (query embedder) + numpy/scipy/sklearn double-init OpenMP → "Abort trap:
  6" on digest/search. Fix = `KMP_DUPLICATE_LIB_OK=TRUE` in `serve-api`. NOT a
  faiss problem — faiss is fully retired (LanceDB, ADR-099); `vectors.faiss`
  files were stale orphans and were deleted this session (~6.7 GB freed across
  worktrees).

---

## 3. Open thread the operator raised (unanswered — pick up here)

**`topic_cooccurrence` enhancement.** Operator asked "how to 'fix'/'enhance'
topic co-occurrence to generate useful data". My recommendation (not yet acted
on, no GH issue filed — operator alone opens issues, see memory):
- **Corpus scope:** replace raw co-occurrence counts with **PMI / lift**
  weighting so ubiquitous topics don't dominate; surface the top-weighted pairs.
- **Episode scope:** add **insight-level proximity** (topics co-occurring within
  the same insight/segment, not just same episode) for tighter signal.
- Related open issues from this session: **#1139** (derived interests
  auto-influence ranking), **#1140** (insight_density → skip-guide, with
  eval-side density visualization). Do NOT open new issues without explicit
  operator approval.

---

## 4. Bigger picture — this branch bundles a lot (all committed, NOT pushed)

`feat/consumer-remember` carries, in addition to the uncommitted tab work:
- **#1128 viewer auth epic**: reuse player OAuth/session for viewer; roles
  admin/creator/listener (`app_roles.py`); login landing + header user menu +
  admin Users view; operator API admin-gated (`OperatorWriteGuard`);
  **Dashboard is admin-only** (creator has NO dashboard); auth-status gate
  (`GET /api/app/auth/status` → `{enabled,user}`) so backend-less viewer e2e
  specs stay open; seeded roster (1 admin/2 creators/2 listeners); dev-user
  sign-in picker; multi-user parallel isolation tests.
- **Enrichment**: `make enrich` target; schema-default prefill in
  `EnrichmentConfigEditor`; ML enrichers `records_written` fix.
- **Serve fix**: KMP_DUPLICATE_LIB_OK.
- **Deps**: folded-in Dependabot bumps (closed those PRs with "comes via this
  branch").
- Paragraph-transcript rewrite (player) landed earlier on this branch.

### Before pushing (when operator says push)
- `git fetch origin main && git rebase origin/main` (memory: rebase every push).
- Push with `--force-with-lease`.
- **PR body must list `Closes #N`** for every delivered issue (#1128 etc.) +
  `Part of #<epic>` — squash erases commit mentions, PR body is the only carrier.
- Default PR to **ready**, not draft.
- Run `make docs` if any `*.md` changed (mkdocs strict not in pre-commit).
- For viewer PRs touching testids/chip surfaces, `make ci-ui-full` (stack-test
  is only in ci / ci-ui-full, not ci-ui-fast). Run `npm run build` locally
  (vue-tsc strict) — ci-ui-fast doesn't invoke it.
- `make ci-fast` / `ci-ui-fast` only as the FINAL step, once.

---

## 5. Do-NOT list (learned the hard way)
- Do NOT touch `TopicEntityView.vue` / `PersonLandingView.vue` for the tab work.
- Do NOT commit the tab work until the operator confirms the visual check.
- Do NOT push anything without explicit "push"/"go".
- Do NOT open GH issues without operator approval.
- Do NOT re-run ci-fast to verify a subtarget fix — run the subtarget.
