# Multi-user support + graph FSM — analysis (2026-06-07)

WIP review requested alongside the diarization work. **Conclusion: the graph FSM
is *not* the multi-user blocker; real multi-user is a server-side platform
initiative, much larger than a normal batch — track it as its own epic/PRD/RFC,
separate from `feat/diarization-followups`.**

## Reframe — the graph FSM is already per-user

The graph handoff orchestrator FSM (ADR-094 / RFC-085) is **entirely client-side**:
a Pinia store (`web/gi-kg-viewer/src/stores/graphHandoff.ts` +
`services/graphHandoffFsm.ts`), states `idle → loading → merge → redraw → apply →
ready`, orchestrating navigation handoff across ~13 surfaces. **No server-side FSM
state.**

"Shared generation counter / shared store" race hazards only fire if **two users
share one browser tab** — not the multi-user model. In real multi-user each user
has their own browser → own Pinia instance → own FSM. So the FSM is per-user by
construction. The only FSM-adjacent gap is **single-user UX**: its state (corpus
path, subject rail, lens) is global per-browser, so one user switching
corpora/workspaces leaks state — a workspace-switching cleanup, not multi-user.

## The real lift — server-side, platform-scale

`src/podcast_scraper/server/` is built for **one operator, one corpus, one
deployment** (codespace-per-operator).

| Gap | Evidence | Lift |
| --- | -------- | ---- |
| Zero auth/identity (no login/sessions/`Depends` security) | `server/routes/*`, `app.py` | Medium |
| Single shared corpus (`app.state.output_dir` set once) — no tenancy | `app.py` | **Large** |
| 5 unguarded write endpoints → last-write-wins, no authz/audit | `routes/{feeds,operator_config,jobs}.py` | Medium |
| Global process caches + single job queue (no owner/quota) | `search/corpus_graph.py`, `pipeline_job_registry.py` | Medium |
| Client state global (unscoped `localStorage`) | `stores/shell.ts` | Small |

## Recommendation

1. **Don't block on it** — codespace-per-operator already isolates one user per
   deployment. That's the pragmatic multi-user until shared deployments are needed.
2. **If/when shared multi-user is real**, sequence: identity (API-key/OAuth) →
   authz + audit on the 5 write paths → per-user corpus tenancy →
   concurrency/resource isolation → client sessions. Quick wins first (API-key
   auth, audit, per-user localStorage, job-config snapshots) before the tenancy lift.
3. **Graph FSM needs ~nothing** for multi-user; optional per-corpus state-scoping
   cleanup for single-user workspace switching.

→ Promote to a dedicated **PRD + RFC ("multi-tenant serving") + GH epic** when
prioritized. Out of scope for the diarization batch.
