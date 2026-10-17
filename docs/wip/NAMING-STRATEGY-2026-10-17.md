# Naming strategy — from 24% hidden to the limit of what the text allows

Status: **DRAFT v0.2 for discussion** (2026-10-17). Nothing in it is started.

Every item below is done under [the corpus change method](../guides/CORPUS_CHANGE_METHOD.md). The
method says *how* we judge a change. This document says *what* we change, in which order, and why.
It follows the Fable 5 advisor's way forward and its second review (both 2026-10-17). Targets use
the method's scoreboard (S1–S11).

## The goal

A listener sees every insight, under the person who said it, wherever the episode's own text and
audio make that knowable. Where they don't, we say so honestly. **No gating number gets worse**
(method principle 5, including its trade rule). There is no deploy until items 1–3 below are
proven better (operator, 2026-10-17).

## Where we are (method §3)

| | `snapshot-prod-20261016` | prod live 2026-10-17 ~16:09 |
| --- | --- | --- |
| S1 hidden by our naming failure | 16,212 / 116,649 (13.9%) | not measured (no 10-17 snapshot, #2378) |
| S2 hidden total | 28,251 (24.2%) | not measured |
| S3 principal voices typed "nobody names them" | 8,236 (committed command; earlier 6,542 and 7,710 were join bugs) | not measured |
| S4 named quotes, en (cross-check) | 73.7% | 74.8% |
| S5 episodes with no named voice, en (cross-check) | prod record 267; not reproducible from the backup (379) | 199 |
| S7 insight-weighted gold | to be re-derived (v0.1 join double-counted) | — |
| S8 undecidable | to be re-derived | — |
| S1c, S9, S10, S11 | not measured | not measured |

Trend: S1 was 12.4% on 2026-09-30, so it got worse over 16 days of naming work.

The 10-16 snapshot predates #2332, which finished at 16:30 UTC on 2026-10-17 (91 of 91 episodes;
batch 1 `full`, batches 2–4 `rediarize_only`, per prod). Any baseline for this strategy needs a
newer valid snapshot.

## Already done on 2026-10-17 (no code)

- **Prod told:** hold 0031; no deploy. Whether 0030, already pending on the deployed sha, runs is an
  open operator decision. I had extended "no deploy" to "no 0030" myself.
- **My wrong statements corrected with prod:** "13× faster", "expect refusals", "0031 ready",
  "rediarize is worse", "I re-measured it".
- **Issues updated:** #2200, #2338, #2332, #2097, with corrections. #2380 opened for the 0031
  defect.
- **max_model_len warning:** handed to prod to own (operator's instruction).

## The items, in order

### 1. Instruments and gates first

Nothing can be judged until the yardstick and the gates exist (method step 0).

1a. **A valid snapshot.** Check that the backup carries every file the scoreboard reads (D12: 122
episodes read "no segments" on 10-16). Take a fresh snapshot after #2332.

1b. **One scoreboard command** producing S1, S1n, S1c, S2, S3 (raw-voice join), S9–S12, S10a/S10b,
A1–A4 and S7–S8. It is built from the existing modules (`quality/attribution.py`,
`quality/naming_coverage.py`, `naming_gate.py`), from the prototype `lost/scoreboard2.py`, and from
the reviewers' scripts (`reviewer/sb.py`, `reviewer2/sb_mine.py`, `diff_mine.py`,
`overshoot_mine.py`), and committed with the method.

1c. **S7 and S8 inside `naming_gate.py`**, on the raw-voice join (insight → first grounded quote →
segment → raw `speaker` id). A run that moves 0 voices prints "no effect".

1d. **The migration outcome hook** in `upgrade/migration.py`. The receipt records S1/S2 before and
after from the clone run. Verify fails if S1 or S1c rises without an approved, identity-listed
reason.

1e. **The residue census.** Insights on a voice with a real name that are still
`surfaceable: false` number **1,786 on the 10-16 snapshot**: S1n, confirmed by both reviewers.
Example: Ben Horowitz on Invest Like the Best, whose voice carries his name on 481 segments, is
typed `unknown`, and has all 51 insights hidden. The cause is not diagnosed (m0025/m0029, or the
pipeline itself). Count voices typed
`unidentified` that have a stated or known name unbound (Moscow 232's host). Both are inputs to
items 2 and 3.

1f. **A new unseen validation set**, drawn now: half sampled in proportion to hidden insights per
feed, half uniform. Labelled now, scored for the first time on item 3's first candidate. val_v1
becomes development-tier. The 51 stale gold cases are re-cut as a new labels version.

1g. **The plan listing and diff** (method D15), generalised from `lost/m0030_list.py` and
`diff30.py`. Every migration's planner emits per-episode counts for every counter, plus input
sha256s, so clone-vs-prod is a diff, not an argument.

- **Target:** the same command on the same snapshot gives the same numbers.
- **Acceptance:** a fresh reviewer re-derives S1, S1n, S2 and S7 with its own code.

### 2. Fix 0031 and the rewrite it inherits (#2380)

- **Name gained:** the voice type is cleared on segments, quotes and insights; visibility and
  routing are restored with the pipeline's own `_apply_route_and_tag`; diagnostics say `person`.
- **Name lost:** the voice gets the type the replay gives it (an advert read is `commercial`), not
  a blanket `unknown`.
- **The verify** checks name, role and voice type on every surface, and the consistency of every
  insight on the voice.
- **The table test** covers every transition (method D7).
- **Re-read the 166 entries under the deciding-evidence rule** (method D13). Every entry stores
  the line that decides the name. Read every entry whose stored evidence is an ad read first. The
  Hard Fork "OpenAI's Two-Week Pause" pair is wrong: SPEAKER_02 is Kevin Roose, not Jill Lepore,
  and SPEAKER_00 is Jill Lepore, not Kevin Roose. Correct it or drop it. The reviewer's heuristic
  flags 5 entries, of which Jason Goldberg is not read.
- **Promo voices that lose a name** (about 26 insights: the Odd Lots "Our Town" trailer, Tracy
  Mumford's promo) are typed `commercial`, not `unknown`. The other ~113 wrong-name removals land
  in `unknown` under the trade rule, listed by identity.
- **If 1e finds residue** from m0025/m0029, the corrected rewrite repairs those episodes too.
- **Target, predicted as exact integers before the run:**
  - S1 falls by the unknown-typed insights on gained voices, minus any the replay types unknown,
    plus any residue from 1e;
  - S1c = 0;
  - 0 insights on a named voice stay hidden;
  - S2 moves only by trade-rule moves listed by identity;
  - **S4 does not change** (0031 already reattributed the quotes);
  - nothing outside the listed voices and the residue changes visibility.
- **Scope, stated honestly:** 166 voices in 126 episodes, plus any residue. A repair of its own
  list, not the fix for #2200.

### 3. The naming code, ranked by what it costs

Order is decided by measured loss (method principle 6). Step 3.1 produces the ranking; the rest are
the known candidates it will rank.

3.1. **Rank the losses, with a committed script.**

- Hidden: join held voices to their stored decision outcomes (`inputs.llm_resolution.verdicts`:
  invented / third_person / duplicate / unmapped_voice / role_only / abstained) and to #2200's
  buckets. Rank by insights.
- Misattributed: rank S7's wrong-person insights by pattern and insights.

The candidates it will rank:

- **Wrong person on a shown insight.** Already tracked: the gate's `wrong_name` class, K1 swaps
  (#2333), F4a (the LLM put the guest's name on the host's voice), #2224, #2367 and #2352 Open 1.
  Only the insight weighting is new.
- **K8, a stated guest who owns the talk (#2338).**
  - Build the cross-post guard proposed there: do not fire when a feed host is unplaced and the
    dominant voice is hailed by that host's first name.
  - Keep the two pinned tests unchanged. Tighten K8's floor to "surname spoken in an introducing
    clause by a host voice, or first name hailed with the dominant voice answering next", using
    the existing `_refuted_unless_hailed` machinery.
  - Measure with S7's insight weights. The 50% share floor stays.
- **#2200 bucket A:** the model found the name, then abstained or a guard discarded it.
- **Junk known-host pools** ("Norman Conquest", "Machine Learning Street", "China Plus"). First
  measure whether a repool already clears them. If it does, this is a record repair, not a rule.
- **Solo host split or unnamed** (In Moscow's Shadows, The Flip). Cross-episode voice identity,
  **forward capture only** (option A in `VOICE-IDENTITY-DESIGN-2026-10-12.md`, already built on
  Track B). Options B/C re-embed stored episodes on the DGX and are **not** a stored-corpus repair
  under the prod rule.
- **Two-host seats** (#2078), the eponymous host (#2136), the Conversations with Tyler host
  (#2224). These are folded into the ranking, not done separately.

3.2. **The LLM-would-solve list**, a named output of this item (naming-arc rule 4). It lists the
cases the text cannot decide and a rule must not brute-force: K1 swaps, #2366 silent hosts, solo
splits in stored episodes until option A has data, and S8's undecidable voices.

Each rule goes through the method:

- scenario test first;
- dev gate, then the unseen validation set once;
- a whole-corpus replay with the reviewer's read sample;
- the scoreboard before and after;
- **target:** S1 and S7-wrong fall by the predicted amounts, and S6 = 0.

### 4. Repair the stored corpus

After the code is proven:

- a replay-derived migration with the corrected rewrite from item 2;
- text on disk only (no DGX, no LLM, no rebuild);
- judged on a snapshot clone;
- **target:** the corpus moves the way the replay predicted.

### 5. Keep the gates running

- **Nightly guardrail (#2334 S8):** an alert when S1, S1c or S5 rise or S4 falls. S1–S3 and S9–S11
  are added to the corpus snapshot and the `/metrics` gauges, so they get a history.
- **Prod's acceptance:** the refusal rule is written into the prod session's own instructions
  (method §6).

### The deploy

One deploy, when items 1, 2 and the first proven rules of 3 pass the method's deploy gate (step 10)
on one fresh snapshot clone. The operator decides.

## Running in parallel, not part of this strategy

- **#2332:** complete (prod, 16:30 UTC): 91 of 91 episodes. The product trade it made, on my local
  5-episode test (`lost/five.py`; the advisor reproduced it):
  - before, those episodes had no voices, so every insight was shown unattributed;
  - after, some are attributed and the rest hidden: visible 109 → 104, hidden 0 → 90, total
    109 → 194.

  Live S5 fell 267 → 199 today. That #2332 caused it is not verified. Still to ask of prod:
  - keep each episode's pre-run files so any episode can be reverted from text;
  - report before/after on its own output with the scoreboard command once item 1 exists;
  - report the stale-gold-case count it adds.
- **0030:** evaluated as the method's first practice run (`EVAL-0030-0031-2026-10-17.md`). The
  operator accepted the measured talk loss. Prod's dry run gave 714 / 412 / refused 0, and its
  plan equals the snapshot's minus 13 explained episodes. A fresh reviewer gave GO WITH
  CONDITIONS. **Operator GO, 2026-10-17; prod is applying it with conditions C1–C5.**
- **Left outside 0030, for this strategy or the pipeline:**
  - ad copy still in served text after today's cut (A1: 406 marker hits in 296 episodes);
  - ad copy published as a guest's insight when one of its quotes is real talk (Krishna Rao /
    Felix);
  - the show's own plugs restored by the cut (A4).
- **Nightly 401s on speaker detection:** with prod and the operator.
- **max_model_len warning:** prod owns it.

## Operator decisions (2026-10-17)

1. **Scoreboard:** yes, method §3 and its gating rows.
2. **Reviewer: yes, and it is the key rule.** A fresh advisor is always spun out per change, bound
   by the §6 checklist. Never the builder, never a resumed reviewer.
3. **Gold sets:** yes. Refresh the stale cases, draw the new validation set, audit the undecidable
   sample (item 1f).
4. **K8:** OK. The cross-post guard, plus the tightened floor with the pinned tests unchanged.
5. **Voice identity:** OK, forward capture only. **We may come back to this later.**
6. **0030 and 0031:** evaluated now under this method as its first practice run; see
   `EVAL-0030-0031-2026-10-17.md`.
7. **Where the prod session's refusal rule is written.** The prod session answered (2026-10-17). It
   loads:
   - `~/.claude/CLAUDE.md` and `~/.config/AGENTS.md`, both symlinks into the agentic-ai-homelab
     repo;
   - the CLAUDE.md of a stale podcast_scraper-infra checkout (detached at bce4ce754), which does
     not import the repo's AGENTS.md;
   - its auto-memory.

   No skill takes incoming handovers. A rule in podcast_scraper's AGENTS.md would **not** reach
   it. Places that do, both in the homelab repo:
   - the global `workstation/config/AGENTS.md`;
   - the `corpus-migration` and `prod-batch-chain` skills (`workstation/claude/skills/<name>/SKILL.md`),
     where the rule fires exactly when a corpus change is about to run.

   Builder's proposal: the rule goes into those two skills, plus one line in the global AGENTS.md.
   That is a homelab-repo change, so it needs the operator's approval.
