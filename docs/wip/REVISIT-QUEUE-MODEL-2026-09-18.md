# Revisit queue at one year — model, and what it says we do not know

Date: 2026-09-18
Companion script: `docs/wip/revisit_queue_sim.py` (uses the real `select_due`, not a restatement
of its rules, so the numbers move if the ladder moves)

## The question

The operator's framing: *"I listen to an episode per day over 365 days, and conservatively I
highlight two things per episode."* → **730 captures in year one.** Does the Revisit tab need
filtering, a cap, or anything else to stay usable at that size?

## Setup

- 2 captures/day for 365 days.
- Ladder unchanged: 2d → 7d → 30d → 90d, indexed by surface count, capped at the last rung.
- `GET /resurfacing` applies **no limit** and does **not** mark anything surfaced by rendering it
  — only an explicit action does. So "due" below is exactly what the route returns and what the
  tab draws.

## Result

| scenario | peak | median | day 365 | never reviewed | backlog |
| --- | --- | --- | --- | --- | --- |
| Diligent — opens daily, clears the queue | 12 | 8 | 12 | 4 | 2% |
| Realistic — opens daily, gets through 10 | 94 | 8 | 96 | 20 | 13% |
| Weekly — opens Sundays, gets through 10 | 706 | 326 | 698 | 406 | 96% |
| Light — opens Sundays, gets through 5 | 722 | 358 | 719 | 532 | 98% |
| Lapsed — opens it, reviews nothing | 726 | 362 | 728 | 730 | 100% |

## Reading it

**Volume is not the problem; cadence is.** A daily user ends year one with 12–96 due — a list, not
a crisis, and filtering would be a solution to a problem they do not have. The interesting cliff is
between *daily* and *weekly*: the same person, the same 730 captures, ends with **698 due (96%)**
simply for opening the tab on Sundays instead of every day.

**The ordering is worse than the depth.** The surface sorts most-overdue-first, and an unreviewed
capture never moves `last_seen` off its capture date, so it grows more overdue for ever. Measured
on the weekly run at day 365:

- top 10 cards were captured **200–295 days ago**
- the first card from the **last 30 days** sits at **position 593**

A weekly user would scroll past 592 cards to reach anything recent. Their newest thinking never
resurfaces, while the oldest permanently holds the top. The surface inverts its own purpose: it is
supposed to bring back what you are still working on, and instead it is an archive read
oldest-first. Retiring (shipped in `09870b852`) gives the ladder an exit, but a user must press it
698 times to dig out — it does not address this.

**Therefore a cap alone is the wrong fix.** Truncating to the first N makes it *strictly worse*:
the N shown would be the 200-day-old ones, so recent captures move from unreachable-by-scrolling
to unreachable entirely. Any cap has to come with a change to what "most important" means —
recency-weighted, interleaved old/new, or a per-session budget that deliberately mixes rungs.

## What I am NOT claiming

- **I have not modelled real behaviour.** The five scenarios are invented shapes with round
  numbers, not measured ones. "Two highlights a day, opens it weekly, gets through ten" is a
  guess dressed as a parameter. The cliff between daily and weekly is real *within the model*;
  whether real users sit on the near or far side of it is unknown.
- **Capture rate is assumed flat.** Real capture is probably bursty (a great episode yields eight,
  a dull week yields none). Burstiness changes peak depth and this model cannot see it.
- **"Reviewed" is assumed to mean the button.** Opening the player from a card also marks
  surfaced (#35). The split between those two paths is unmeasured.
- **No abandonment.** Nobody in this model stops using the app, and everyone who opens the tab
  reviews the same number every time.
- **Nothing here is validated against a human.** Zero users have touched this surface.

## Decision

Do **not** design filtering, caps or re-ranking off this model. It is sufficient to say the
current design is safe for a daily user and degrades badly for a weekly one — and that the failure
mode is *ordering*, not *volume*, which is the non-obvious part and the thing worth instrumenting.

Instrument first, then model from measurements. Event list: see the analytics issue.
