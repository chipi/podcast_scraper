# Epic #2097 activities 6-7: the #2075 measurements re-derived (2026-10-27)

Open-items item 15 (moved from prod to the naming session, 2026-10-27). Read-only, on
snapshot-prod-20261027 (taken 11:04Z, after 0034-0036 and the 0030 re-run; sha256-verified with
`download_and_verify_snapshot.sh`), 3006 served episodes (`select_served_artifacts`). These are
inputs to the operator's closure decisions on #2075, #2093 and #2095; closing an issue needs the
operator's go. Scripts in the session scratchpad (`rm27/zero2075.py`, `t2095.py`); the golden set
is committed.

## #2075: episodes that end with no named speaker

#2075's definition: an empty speaker roster and zero attributed quotes.

| Measure | Value |
|---|---:|
| served episodes | 3006 |
| empty roster | 151 |
| empty roster and zero attributed quotes | 151 |

By feed: The a16z Show 41, The Rest Is History 37, NVIDIA AI Podcast 21, The Rest Is Politics:
Leading 12, Machine Learning Street Talk 9, The Open Africa Podcast 7, The Flip 5, BizNews Radio 3,
Planet Money 2, Interpreting India 2, others 12.

The issue's 72 (48 of them a16z) was taken on an older, smaller corpus, so the two counts are not
comparable as a trend. What they show: the shape #2075 describes is still served, on more feeds than
a16z. Not fixed by any migration of this release (none adds guest names from the title).

## #2095: does the guest detector return the guest a title form names?

For each served episode, a guest named by one of the issue's three written forms in the title (or
description), matched against what the detector stored (`tried.metadata_named` and
`tried.detected_guests` in the speakers diagnostics), by last name and a fuzzy ratio over 0.75:

| Form | Found | Detector had it | Rate |
|---|---:|---:|---:|
| title-lead `<Name> on <Topic>` | 197 | 165 | 84% |
| `<GUEST> joins <hosts>` | 94 | 51 | 54% |
| `<X> speaks with <GUEST>` | 28 | 13 | 46% |
| **total** | **319** | **229** | **72%** |

Not comparable to the issue's 53% (241 on the 2026-09-14 snapshot): the issue's extraction script
is not in the repo, and this name pattern is looser (the misses include non-names such as an
organisation or "<Company>. He"), so the denominator holds some false extractions and the rate is a
floor, as the issue's was.

## #2093: the four guest forms

#2093's own measurement was held as circular (scored against the roster being repaired). The table
above is the form-by-form extraction count it needs (title-lead, joins, speaks-with; bio-opener not
measured). Nothing was built.

## The #2075 golden set

`scripts/measure/2075_speaker_golden_set.py --snapshot <snapshot-prod-20261027>`: **10 of 11**
cases pass, 0 could not run. The one failure is `hardfork-guest-in-record` (a guest neither bound to
a voice nor unplaced in the record). It is a `record` case, which the script documents as depending
on ingest-time detection frozen in the diagnostics; the code of this pass does not change the roster
(the forced-host-seat veto was reverted), so this pass did not cause it. Its cause is not
established.

## Not covered

- The bio-opener form ("<Name> is a professor of ...") of #2093.
- No episode was read to confirm a miss is a real miss.
- The 151 episodes were not classified by why they have no name (no guest form, a form the detector
  missed, a solo episode).
