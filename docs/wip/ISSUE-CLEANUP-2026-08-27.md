# GitHub issue cleanup — v2.7 refocus (2026-08-27)

Goal: refocus on **bug cleanup for v2.7** (not features). Assessment → close dups → retag features to v2.8 → verify/close fixed → triage remaining bugs to v2.7.

## Before → after
- Open issues: **221 → 198** (23 closed as duplicates).
- Milestones: v2.7 **20 → 30 open** (now all bugs + 2 release-docs) · v2.8 **16 → 34 open** (features) · unmilestoned bugs **49 → 0**.

## Step 1 — duplicate clusters closed (23 dups → 5 canonicals)
GlitchTip auto-files a fresh issue per occurrence of the same recurring bug. Consolidated:

| Canonical (kept, v2.7) | Closed as dup |
| --- | --- |
| #1849 cold-storage truncated/byte-inconsistent upload | #1840–1848 (9) |
| #1556 ADR-148 re-roll summary-schema parse fail | #1557,1575,1576,1577,1695,1730,1820,1861,1866 (9) |
| #1570 OpenAIProvider called before initialize() | #1578,1579,1580,1581 (4) |
| #1395 transcript cleaner over-strips content | #1558 (1) |
| #1483 unhandled PermissionError on appdata file-lock | #1485,1859 (2) |

Note: the auto-filer keeps regenerating (#1861, #1866 arrived after the first close pass and were folded in). A durable fix is to point the filer's dedup at these canonicals, or fix the root bugs.

## Step 2 — features retagged v2.7 → v2.8 (18)
All 20 v2.7-milestoned issues were non-bugs. Moved to v2.8: #46, #102, #179, #216, #372, #426, #538, #806, #860, #911, #972, #976, #993, #1002, #1028, #1062, #1142, #1143.
**Held in v2.7 for operator call** (release-gating docs, not features): **#255** (pre-release checklist), **#627** (viewer release QA).

## Step 3 — verify + close genuinely-fixed: NONE closed as fixed
Verified against current code; the auto-filed bugs are **real and still open**, not already-fixed:
- **#1483 (PermissionError)** — `app_collections_store.create_collection:174` does `with _lock(...)` with no PermissionError guard → uncaught → HTTP 500. Real. (My earlier `chown 1000:1000` only fixed the smoke user's dir, not the code gap.)
- **#1570 (OpenAI-init)** — `summarize()` correctly raises the guard `RuntimeError` (L1322); the bug is a caller calling it before `initialize()`, unhandled upstream. Real.
No false "fixed" closes were made.

## Step 4 — remaining unmilestoned bugs triaged → v2.7 (23)
Real, distinct (dups already removed): #1865,1864,1862,1855,1854,1852,1603,1600,1599,1598,1597,1592,1591,1585,1564,1546,1529,1480,1454,1409,1346,1345,1323.
No native-crash dup: #1345 (iOS SIGABRT, single/unsymbolicated) vs #1346 (Android Looper) are distinct platforms.

## Operator follow-ups (not actioned — need your call)
- **#1345** — single, unsymbolicated iOS SIGABRT (one occurrence). Low signal; candidate for close-as-wontfix.
- **#1854 / #1855** — single feed-run failures; possibly transient. Verify before committing to v2.7 fix.
- **#255 / #627** — confirm these two stay v2.7 (release-gating) or move to v2.8.
- **v2.7 now holds 28 bugs** — that may be more than one release should carry; consider a de-scope pass to v2.8 for the deep ones (e.g. #1323 GI MiniLM MPS SIGSEGV, #1346 Android native).
