# Session handover — 2026-08-20

Three changes landed on `main`, two issues filed for later, one filed for a future capability.
Nothing has been deployed and nothing has touched the production corpus.

---

## 1. What is on main

| Commit | Issue | What changed |
|---|---|---|
| `806d55cc` | **#1686** | an episode that loses its summary now says so on the artifact, retries a transient cause once, and is requeueable |
| `255b0b82` | **#1799** | 31 cross-suite test failures fixed — module stubs that outlived their tests |
| `b70c35d7`, `ac8dbb65` | **#1685** | the resolvability measurement, then a correction: it was reading one graph layer |
| *(pending)* | **#1685** | bare person names no longer minted as global followable ids |

Filed, not started: **#1798** (numpy/mypy), **#1799** (remaining test-isolation work), **#1801**
(the entity-resolution enricher).

---

## 2. #1686 — a lost summary is no longer silent

**The measured problem.** Production had 8 episodes with no summary. The audit could not say
whether the pipeline had produced nothing and recorded success, or degraded by design — because
`_summary_text` returned `""` for both `summary: null` and a summary object with no text. The
audit now splits them. Answer: **8 absent, 0 blank** — all the designed #1496 degradation.

**What was actually wrong** was the absence of a mark. #1647 built a stage ledger precisely so
"did this stage run?" stops being a guess, and the five paths that end with no summary were never
wired into it — the only `summarization` entry it ever carried was
`deadline_exceeded_but_completed`, a summary that was LATE but real. It recorded the harmless case
and stayed silent on the harmful one.

Four changes: the ledger now records `failed` with a cause slug; the tokenizer race (the one
failure the code itself calls transient) is retried once, at the caller so every guard re-runs; a
genuinely lost summary is reported at `error`; and `make corpus-summary-audit` /
`corpus-summary-worklist` make the degraded set addressable, with a hard loop guard.

**Marko's framing:** "it's never acceptable to have an episode without the summary of a single
one." There is no tolerance threshold — one missing summary fails the gate.

---

## 3. #1799 — 31 failures nothing was running

`pytest tests/unit tests/integration` in ONE process failed 31 tests. Both halves passed alone.
Three leaks, one shape: a Mock put into `sys.modules` and never taken back. Five unit modules did
it to core SDKs (and each *documented the leak in a comment*); one left a Mock `spacy` with
`.load` deleted, which made a **feed-error** test fail for a **spaCy** reason two suites away; one
snapshotted "the original" one line *after* installing its own mock, so teardown faithfully
re-installed it forever.

**Why CI never saw it:** unit and integration run as separate jobs, xdist-sharded, so the two
never share a process. Green by partitioning, not isolation — the same shape as #1798, where the
mypy gate passes only because CI happens to resolve numpy 2.4.6.

Two guards now in `tests/conftest.py`. The session-scoped one took three attempts: per-fixture
flagged 12 tests using `monkeypatch` correctly, per-test flagged 16 using the documented
`setUpModule` pattern. **Both would have failed honest code.** Only survival past the end of the
run is unambiguous.

---

## 4. #1685 — a bare first name is not a person

`person:jensen` is what happens when three shows each say "Jensen" with no surname: three
identical slugs collide on one **followable** node. `POST /interests/{token}` accepts it, and
derived interests mint it from listening behaviour with no click.

**Measured (678 episodes):** 208 occurrences of 172 single-word person ids — **12 resolvable
within their own episode, 0 ambiguous, 196 orphan**. Hand-checked shapes: hollow
(`person:jensen`, no insights at all), pooled (`person:sam`, carrying a position about *Samuel
Moyn*), locally resolvable (`person:alex`, whose full name is a co-speaker in that episode).

So this is **prevention with a small repair attached**, not a repair. Its value is not minting a
global id for 196 occurrences, not healing 12.

Implementation, after an advisor review that changed the design in two places:

- `identity/bare_name_scope.py` — the rule, shared **verbatim** by the pipeline and the migration
- the seam is a post-extraction pass in `metadata_generation`, **not** `entity_node_id()` (pure,
  per-name, and misses `identity/slugify.person_id` which the GI speaker path uses)
- one line added to `is_unresolved_speaker_placeholder` — consulted in twelve modules including
  `entities_from_kg`, so scoped ids leave cards, follows and derived interests at once. **Without
  this the change would not have implemented the decision**
- `upgrade run` (m0007) backfills; verified on a real corpus copy: 7 bare ids → 0, dry-run wrote
  nothing, re-apply a clean no-op, 0 dangling edges, 0 duplicate node ids
- `cfg.bare_name_heal` (default True) governs the only unrecoverable half

---

## 5. Next steps

1. **Dispatch the capability audit** — confirm the bare-name population collapsed. Needs `prod`
   environment approval.
2. **Run m0007 against production** — corpus backup first; afterwards rebuild enrichment and the
   search index (delete `episode_fingerprints.json` alongside `lance_index`, or the rebuild
   silently indexes nothing).
3. Parked: **#1798**, **#1799**, **#1801**, and resurfacing's missing terminal state (a product
   decision — what does "retired" mean — not a bug).

---

## 6. The pattern worth carrying forward

Nearly every real finding today was **a measurement reporting something adjacent to what it
claimed**, not the thing measured being broken:

- the audit conflating absent and blank summaries — the whole #1686 verdict rested on it
- the resolvability measure reading only the KG layer, undercounting by 4x
- a sabotage that silently no-op'd and read as "the tests are weak"
- a guard that would have printed and exited 0
- two guards that would have failed correct code
- a work-list where a scoped id matched **itself** as a resolution candidate

The fixes were mostly making the instrument ask the real source, and the tests that mattered
asserted data was **found** rather than that a count was zero. When a check reports something
surprising, verifying the check is cheaper than acting on it.
