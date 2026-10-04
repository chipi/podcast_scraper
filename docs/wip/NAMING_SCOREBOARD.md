# Naming scoreboard — every slice attempt, and an honest conclusion at the end

Operator, 2026-10-02: finish the planned slices, keep track, and conclude honestly whether the
rule-by-rule approach is hitting a wall or how far it can be pushed. Method:
`docs/guides/NAMING_GATE_RUNBOOK.md` (gold sets, gate; loops per problem: 3 for seats, then 2 max
from 2026-10-02).

Numbers are voices scored by `scripts/measure/naming_gate.py`: development = 101 episodes (393
scored voices, labels_v1), validation = 500 episodes (2,010 scored voices, labels_v1).

## The ceiling (what a careful reader gets from the same text)

The labellers worked only from the case files (each voice's opening/closing and the first turns —
less than the pipeline has). On validation they put a supported name on **1,368 of 2,382 voices**;
an independent audit agreed with 89–99% of the labels it checked. The pipeline today publishes a
correct name on **602**. Most of the gap is recoverable from text the pipeline already holds; the
question this scoreboard answers is how much of it RULES can recover.

## Plan — one step at a time, each result recorded here (operator, 2026-10-03)

1. Measure the LLM step's marginal value on the gold sets (same code with vs without its answers).
2. Problem 2, last loop (host evidence outranks a lone guest phrase; self-introduced hosts).
3. Problem 3: host pools — give the right names to voices the person check emptied.
4. LLM guards (veto/check the LLM's answers; replayable).
5. Feed-level analysis with the Fable advisor: per-show patterns; local beats global.
6. Mini autoresearch loop on the gate: propose → score on dev → keep only if better with zero host
   regressions → validate once on the 500; rules and per-show profiles first, the LLM step after
   (DGX budget agreed first).
7. Honest conclusion: how far this can be pushed.

## Cumulative table (validation, 500)

| Step | correct name | wrong name | missing name | promo/ad/clip published | host/guest swapped | hosts correct |
| --- | --- | --- | --- | --- | --- | --- |
| today (baseline) | 602 | 155 | 384 | 49 | 31 | 342 |
| + seats (v4, accepted) | 604 | 138 | 399 | 47 | 31 | 342 |
| + person check (accepted) | 604 | 119 | 418 | 47 | 31 | 342 |
| + presenter evidence (accepted) | 615 | 118 | 417 | 47 | 22 | 352 |
| (pools rebuilt from metadata with the committed detector — stale stored pools, not a change) | 623 | 118 | 408 | 47 | 23 | 360 |
| + host pools (accepted) | 644 | 112 | 395 | 46 | 21 | 378 |
| + clipped first-name fix (accepted; LLM guards rejected) | 644 | 112 | 395 | 46 | 21 | 378 |

From the host-pool row on, every column is measured with `naming_gate.py --repool` (each variant
rebuilds the episode's pool from stored metadata with its own hosts module); the rows above it
replay the stored pools. The unlabelled row between them is the repool itself.

## Attempts

| # | Problem | Loop | Change | Dev better / worse | Val better / worse | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | host seats | 1 | seat logic v4 | 6 / 1 | 20 / 1 (host) | loop again |
| 1 | host seats | 2 | + late-guest absorption, dominant-voice guest name, Tracy/Tracey snap | 0 / 0 | 2 / 1 (host) | loop again |
| 1 | host seats | 3 | + snap guard (no claim on a host another voice already is) | 0 / 0 | 2 / 0 | accepted with one known host regression (operator) |
| 2 | host/guest judged across the whole voice | 1 | a guest phrase counts only in the first 60% of a voice | 3 / 0 | 0 / 3 | rejected (guests whose only "thanks for having me" is a farewell) |
| 2 | host/guest judged across the whole voice | 2 | presenter evidence (advisor p7): a voice naming the show in a presenting formula, introducing the stated guest, or one half of an "I'm A / And I'm B" pair is a host even outside the pool, and outranks a lone guest phrase; episode-stated guest host joins the pool; show-name pool entries never named | 22 / 0 | 13 / 2 | accepted (last loop; 2 worse: Hard Fork Kevin's voice named Casey, an unnamed Business of Africa host voice named) |
| 3 | junk names passing the person check | 1 | show/brand and region tails, count words, captured role words, job titles in long names, product mononyms, stray `?` | 4 / 0 | 20 / 0 | accepted (census: 15 distinct names newly refused, all junk) |
| 4 | host pools | 1 | statement ∪ author tags with a per-name person check, wider stated-name grammar, episode-described hosts (byline silenced when the description names one), respelt/merged host not a stranger, solo/ownership/show-mononym guards (measured with `--repool` on both sides; repooling alone with the committed detector = +8 correct on val, stale stored pools) | 11 / 1 | 28 / 6 (3 guests given the host's name or role, 2 co-hosts given the pooled host's name, 1 pooled host unnamed) | loop again |
| 4 | host pools | 2 | + unnamed introducer seat (1c), a forced pool name declines when another voice presents, a self-introduced stated guest who sounds like one is never a stand-in host, a respelt guest name is not spare, a guest the description names after the cue is not a described host, a carried pool name is never forced onto a second voice, a pool host's own self-introduced name stays on a non-host voice (production wiring: the roster pool also gets the episode's stated people) | 2 / 0 vs loop 1 (13 / 1 vs before) | 4 / 0 vs loop 1 (31 / 5 vs before) | accepted (last loop; 5 worse remain: an MLST guest named Tim Scarfe, a Flip guest named Justin Norman, two Latin America in Focus co-host voices named Carin Zissis, one Carin Zissis voice unnamed) |
| 5 | LLM guards | 1 | drop an LLM name the voice's own words refute anywhere in its text (names another pool host as itself, "X and I", introduces X, addresses a stated guest X), plus a joint drop of all LLM host names when one is refuted | 15 / 0 (wrong 29 -> 16, correct 133 -> 141) | not run | rejected on the full-corpus read: of 107 changed voices, 37 better, 23 worse (8 real hosts lost: Hard Fork, Empire, Capitalisn't, a16z), 47 uncertain; the refuting text was mostly another speaker's words bled into the voice, and the joint drop spread one false refutation to both hosts |
| 5 | LLM guards | 2 | refutation read only in a non-dominant voice's OPENING (its first uninterrupted turns), no joint drop, forced names never re-place a refuted name; plus the clipped-name fix below | 6 / 0 (wrong 29 -> 25) | 4 / 3 (correct 644 -> 644, wrong 112 -> 110, missing 395 -> 398; worse: an Unhedged guest named Robert Armstrong, a promo named Olaf Storbeck, a Latent Space guest unnamed) | rejected (last loop): a wash on validation, and one of the three is a guest given the host's name; corpus read 22 better / 10 worse (guests), 8 of the 10 from the vocative rule -- the host's greeting is diarized into the start of the guest's voice |
| 5 | clipped first names (a defect from seat v4, 7e8d25b9b) | - | a clipped given name is never an ordinary word ("case"/Casey on 72 episodes, "and"/Andrew-Andy on 29, "nor"/Norman, "just"/Justin, "the"/Theo); "between you and me, X" is not a co-host formula | 0 / 0 | 0 / 0 | accepted (removes the defect; corpus: 1 voice changed, an improvement -- Trivium "I'm your host... Andrew Polk") |
| 6 | stated guest unnamed in a one-guest interview (Freakonomics review, 2026-10-03) | 1 | the two-voice host-introduction rule also accepts ONE dominant unseated voice (>= 0.5 of talk) with any named host; the stated name matched as an ASR spoken variant; cue "pleasure/honour/privilege to have" | 1 / 0 | not run | loop again (6 of 8 bound-but-unnamed guests were blocked downstream by the third-person guard) |
| 6 | same | 2 | + the third-person guard's SURNAME branch narrowed on mixed-case text (not inside another name -- "the Michael Lewis book", "North America" -- not an eponym -- "the Munger test" -- not a lowercase word -- "last year"); "bank"/"code" org tails; rank-only and greeting-first names refused ("Lieutenant General John", "Hey Josh"); a vocative veto tried and dropped (it cost John Platt his name) | 7 / 1 | 10 / 1 after the rank/greeting fix (correct 644 -> 654, wrong 112 -> 113; worse: Capitalisn't host named Matt Iglesias) | PARKED on branch `feat/naming-stated-guest-binding`: the full-corpus read (54 voices) was ~31 better / ~15 worse / ~6 unclear -- 8 of the worse are junk names in the stated-guest pool ("China Belt", "How Football Shirts", "Meter Redwood", "AI White House", "Russian Spring", "CEO Joseph Nelson") that the over-broad surname match used to refuse by accident, several replacing a correct name; plus Hard Fork's Casey Newton voice renamed to a guest. Next: fix junk at the pool's source as its own problem, then re-measure this; or ship only loop 1 after measuring it alone |

## Problem 6 — next steps (agreed with the operator, 2026-10-03)

Goal: extra help on what is junk and what is a name, and cut what is not a person EARLY — before it
becomes a candidate the roster may bind (the over-broad surname match was doing this by accident).

1. **Trace the junk sources (read-only, on the box).** For each junk name the corpus read surfaced
   ("China Belt", "How Football Shirts", "Meter Redwood", "AI White House", "Russian Spring",
   "CEO Joseph Nelson", "Lieutenant General John", "Hey Josh") find which step put it in the stated
   list: the speaker-detection call's stored output vs another path into `metadata_named`.
2. **Option 1 — a `kind` per name in the speaker-detection call** (`detect_speakers`, prompt
   `openai/ner/guest_host_v1.j2`, prod `vllm` Qwen3-30B). Output one entry per name with
   `kind: person|organisation|place|product|phrase`; only `person` enters the stated list, next to
   the existing placeholder/organisation filters (`workflow/stages/processing.py`, the
   `filter_default_speaker_names` / `drop_non_person_names` block). Veto only: it can remove a
   candidate, never add one.
3. **Measure:** re-run the detection call on DGX for the dev 101 and validation 500 episodes (small
   structured call), then the gate replays the roster with the new name lists; plus a full-corpus read.
4. **Then re-measure the parked surname fix on top of it** — the junk it let through should now be
   cut at the source.
5. Also from the Gray Area review (2026-10-03): promo voices named as guests (Kara Swisher, Sky
   Galloway, Anne Applebaum) and "John Gwynn-Hill" published for Jonquilyn Hill.
6. From the StarTalk review (2026-10-03): the regular co-hosts (Chuck Nice, Gary O'Reilly, Paul
   Mecurio) are published as guests because the feed states only Neil deGrasse Tyson (a per-show
   profile case, #2261); on Cosmic Queries / TYTYK episodes the co-host voice stays unnamed (39 of
   42 insights hidden on one voice); listener names read from Patreon questions ("Ernie
   Carducci", "Lani Lum") are captured as self-introductions while the stated guest (Hakeem
   Oluseyi) stays unplaced.

## Show-sidecar census, all 80 shows (2026-10-03 night)

- Host on a voice: 767 of 920 recorded episodes. Not on a voice: 86 `no_host_name_found`, 63
  `host_known_not_on_a_voice`, 4 single-voice transcripts.
- 8 shows with NO host detected: People I (Mostly) Admire (Steve Levitt — should be findable),
  Planet Money and Unexplainable (rotating hosts), BizNews Radio, The Living Philosophy,
  Philosophy For Our Times, The Open Africa Podcast, How to Touch Grass. Per-show profile (#2261).
- Low-visibility episodes from one unnamed voice: Google DeepMind "When millions of AI agents
  meet" (3/49 insights visible), "From deepfakes to DNA" (26/46); see also the Freakonomics,
  Gray Area and StarTalk reviews above.

## Deepen reviews, night of 2026-10-03/04

- Listener / caller first names captured as self-introduced guests: Curious Cases (Bernie, Jack,
  Laurie, Andrew, Marlon, Keith, Elizabeth) — the same shape as StarTalk's Patreon question names.
- ASR spelling not snapped to the stated guest: "Maggie Adair" placed, stated "Maggie
  Aderin-Pocock" unplaced (Curious Cases).
- One episode per show with nearly every insight hidden behind one unnamed voice: ZOE 0/40,
  AI 4 UX 1/64, FUTURES 2/60, Google DeepMind 3/49 — the stated-guest problem (problem 6).
- Brand author tag in a host pool: "Brilliant Experience" (AI 4 UX), never placed on a voice.
- The Knowledge Project (2026-10-04, job 4502a2dc): host on a voice 10/10, artifacts clean. One
  narrated episode ("John D. Rockefeller…") hides 58 of 104 insights on one `unidentified` voice;
  the Steve Jobs episode lists the people it is ABOUT (Steve Jobs, Ross Perot, Larry Ellison, Bill
  Gates) as unplaced guests next to the real guest Geoffrey Cain, also unplaced. One DEADLINE
  EXCEEDED alarm (metadata > 1,200 s on a long episode), completed.
- Philosophy For Our Times (IAI debates, 2026-10-04, job 8dc97a09): **no host** for the feed
  (5 of 8 `no_host_name_found`); three episodes have nothing visible — two with an EMPTY roster
  ("The secret history of cool" 0/37, "Love is more than magic" 0/44) and Catherine Liu 0/45 with
  her stated name unplaced; a four-person panel 1/40. Junk self-intro names: host "Osbir" (twice)
  and guest "James Madison" (a quotation) placed while stated Elizabeth Anderson is unplaced.
  Rotating-chair debate format: a per-show profile case (#2261), like Planet Money.
- The Living Philosophy (2026-10-04, job 70e6aef3): **no host** for the feed (7 of 9
  `no_host_name_found`); the guest is the only roster entry and is made HOST twice ("Dr. Sharon
  Blackie", "Benjamin Studebaker", LLM); nothing visible on three episodes (an EMPTY roster 0/77,
  Michael Montgomery 0/78 and Layman Pascal 0/59 with the stated guest unplaced and the dominant
  voice unnamed). Same class as Philosophy For Our Times: a show that states no host (#2261).

## Low-visibility episodes, classified (2026-10-04 morning)

| Episode | What happened | Problem |
| --- | --- | --- |
| AI 4 UX "A New Role for Researchers - as Orchestrators" (1/64 visible) | stated guest Ned Dwyer unplaced; the dominant voice (63 insights) unnamed | 6, exactly |
| ZOE "Is your gut the secret to anti-aging?" (0/40) | the LLM put stated guest Prof Elaine Dennison on SPEAKER_01 (9%, the show-intro read); the real guest voice SPEAKER_00 (62%, answering) left `unidentified` | 6, variant: the stated name is spent on a minor voice, so the dominant-voice rule never sees it spare |
| FUTURES "Why Machines Can't Replace Us w/ Neil Lawrence" (2/60) | guest Neil Lawrence given the HOST role; real host Luke Robert Mason unplaced | **7 (new): host/guest swap** — the title's "w/ <name>" marks the guest |
| Google DeepMind "When millions of AI agents meet" (3/49) | no guest stated anywhere in the metadata (timecodes-only description); the 78% voice opens "Very happy to be here" | **8 (new): guest named only in the transcript** — check why the introduction reader did not bind it |

## What the decision trace showed (#2276, 2026-10-04)

On main, not on this branch: the per-voice decision trace (`docs/wip/NAMING_DECISION_TRACE.md`
v3 on main supersedes this branch's v1 design — take main's on the next rebase). Numbers below are
from replaying the 2,319 prod episodes with the trace (rule counts, NOT correctness), plus a
5-episode run with the real LLM on the DGX (Vox, Sean Illing feed) and a join to the gold labels.

Fixed on main (`acbfaecb0`): **the non-regression contract restored arithmetic names.** Of 56
restored voices in gold episodes, 49 carried a forced name (spare pool host name forced onto a
seat, or spare guest name forced onto the last voice) and 42 of those were wrong — "Misha Glenny" on
promos, a host's name on the guest, often beside the same name on the right voice. Forced names are
no longer restored (`SpeakerRole.forced`); expected on the gold cases 42 fixed / 7 lost. Not yet
measured on a live ingest.

Open — new problems, each to be gated dev -> val like every slice:

| # | Problem | Evidence | Size |
| --- | --- | --- | --- |
| 9 | **An LLM-inferred host name takes host-seat step 1** ("named as a stated host"), the strongest rule, exactly like a spoken self-introduction | DGX No Priors run; the Tyler / Julia Ioffe case (c087) is this shape | 710 of 1,712 step-1 seats rest on an LLM name (970 self-intro, 21 co-host formula, 11 intro reader) |
| 10 | **A guest the LLM calls "guest" is seated as host by step 2** (performs host role) — step 2 does not consult the LLM's guest verdict | DGX run ep. 5: "Anna Luise Sussman" (LLM: guest) seated host; her respelling "Anna-Louis Sussman" left spare | not counted yet; related to problem 7 (host/guest swap) |
| 11 | **Cross-promo ads named as guests by their own self-introduction** — NOT new: problem 6 step 5 (Gray Area review, 2026-10-03); the DGX run reproduced it on the same show ("Kara Swisher", "Sky Galloway", "Anne Applebaum", "Jake Sullivan") | all 5 DGX-run episodes | local run had no ad signatures; check what prod's signatures catch BEFORE designing anything |
| 12 | **A corroborated guest is lost**: the LLM puts her name on a voice that only talks about her (refused, third person), a promo self-intro then takes that voice, the name ends spare | DGX run ep. 4: "Theo Baker" | not counted yet |
| — | `host_elimination` (forced guest variant) never fired | 0 of 2,319 episodes | dead rule, or its conditions never hold — read before deleting |
| — | the failure pool: voices typed `a_name_existed_and_we_failed` | 707 voices (2,179 more that nobody names) | the set phase 2 should classify first |

Order (operator, 2026-10-04): these after the deploy, on this branch. First #2276 phase 2 — join the
replayed traces to the gold labels for per-rung precision — so problems 9-12 are ranked by
measured damage, not by how vivid the example was. Problem 11 starts with a prod check, not code.

## #2276 phase 2 — per-rung precision on the gold sets (2026-10-04)

`scripts/measure/trace_rung_precision.py` over the corpus replay's traces (`095a2954d`, 2,319
episodes, `--repool --signatures`) joined to dev 101 + val 500 (601 episodes matched), scored with
the gate's own `outcome()`. Totals: correct 777, missing 465, wrong 141, non-participant 57,
role error 23, spurious 20. Caveat: offline replay, so the LLM rung sees only the LLM names that
survived (lossy) and the pipeline's non-regression restore is not in it.

**Published names, by the rung that brought the name in**

| Origin rung | correct | wrong | non-participant | role err | spurious | precision |
| --- | --- | --- | --- | --- | --- | --- |
| self_intro | 329 | 20 | **48** | 13 | 3 | 79.7% |
| llm_merge (accepted) | 246 | **78** | 2 | 8 | 10 | 71.5% |
| intro_reader | 126 | 25 | 5 | 2 | 3 | 78.3% |
| forced pool host name | 47 | 12 | 1 | 0 | 3 | 74.6% |
| forced guest, one name one voice | 21 | 6 | 1 | 0 | 1 | 72.4% |
| cohost_formula | 6 | 0 | 0 | 0 | 0 | 100% |
| forced guest, dominant voice | 2 | 0 | 0 | 0 | 0 | 100% |

**Missing names (465) — where the gold name is in the trace instead**

| Where | n |
| --- | --- |
| never in the trace at all | **212** |
| published on ANOTHER voice | 88 |
| a known host never placed (stated only as `known_hosts`) | 76 |
| stated in the metadata only, never bound (`metadata_named` 45, `detected_guests` 17) | 62 |
| harvested from the host's introduction, then not placed | 19 (+3 refused) |
| other | 8 |

**Host seats by step** (gold role of the voice): step 1 "named as a stated host" 387 right / 20
wrong, and **17 of the 20 wrong are LLM names** (self-introduced: 248 / 2); step 2 "performs host
role" 76 / **33 wrong** (12 unnamed guests, 9 promos, 6 LLM-named guests, 3 forced pool names on a
guest, 3 clip/ad); step 3 opener 25 / **16 wrong** (39%); 1b 56 / 0; 1c 13 / 0. Not seated at all:
77 real hosts published as guests, 135 hosts and 169 guests left with role `unknown`.

**What this changes**

- The vivid example behind #2276 (an LLM name reused on a second voice once the episode bound it)
  is RARE: 1 of the LLM's 78 wrong names. The LLM's errors are misassignment: 41 are a host-pool
  name (27 the wrong host of the pool on a host voice, 14 a host's name on the guest).
- Problem 9 is real but smaller than the 710-seat count suggested: LLM-named step-1 seats are
  132 right / 17 wrong (88.6%) against 248 / 2 for self-introductions.
- Problem 11 (promos named by their own self-introduction) is the self-intro rung's biggest error:
  48 of its 413 published names are ads / promos / clips.
- The largest pool of loss is upstream of the roster: 212 missing names never reach the trace
  (neither stated, nor heard, nor proposed) — the next thing to classify.
- Seat steps 2 and 3 are where host roles go wrong (49 of 63 wrong seats), not step 1.

**Ranked by measured damage (voices)**: names never in the trace 212 · the LLM's wrong names 78
(41 pool-host misassignment) · the right name on the wrong voice 88 · known hosts never placed 76 ·
stated names never bound 62 · promo/ad self-intros 48 · wrong seats at steps 2+3 49 · LLM names at
step 1 17. Next: classify the 212, then pick the first fix from the top of this list.

**The 212 names that never reach the trace, classified** (where the gold name appears in the case
text; `name_evidence` from the labels):

| Where the name is available | n | gold host / guest |
| --- | --- | --- |
| the episode DESCRIPTION, in full | 101 | 28 / 73 |
| the episode TITLE, in full | 33 (+1 surname) | 1 / 33 |
| only spoken, by another voice ("That's Joyce Mongi…", "Sarah, let me come back to you") | 27 (+1) | 6 / 22 |
| only spoken, in the voice's own words ("Christian Sinclair. I'm the production manager…") | 14 | 4 / 10 |
| not in the case excerpt at all (later in the transcript) | 34 | 8 / 26 |
| the feed description | 1 | 1 / 0 |

So 135 (64%) are in the episode's own title or description and still never became a stated name —
not lost at corroboration: speaker detection never extracted them. Not (mainly) a truncation issue:
the name sits at a median of 327 characters into the description (p75 862). Mostly OLD ingests: 78
from August, 54 from September, 3 from October; 74 of the 135 have no stated name at all, others
only the old failure placeholders (`['Host', 'unknown_guest_1']` on "Today's guest is Aigboje
Aig-Imoukhuede"). Next: run TODAY's detector (vllm Qwen3-30B, `guest_host_v1`) on these ~70
episodes' title + description — ~70 small DGX calls, budget to agree first — to split "fixed since"
from "still missed". The corpus cannot be re-detected and re-rostered under the prod rule; this is a
forward fix plus that measurement.

## Also on this branch — the two open plan steps (from 2026-10-02/03)

### Feed-level analysis and per-show profiles — #2261 (scoreboard plan step 5)

The rule tail per episode is thin (host pools' last loop moved the 500 by 4 voices) and the remaining
errors cluster by show (MLST, The Flip, Latin America in Focus; today also Freakonomics' narrated
"that is <name>" attributions and Gray Area's rotating Vox hosts). Analyse per-show patterns with
the Fable advisor from the show sidecars (`show.json`), the gold sets and the corpus replay; output a
per-show profile the roster reads (host count, presenter style, recurring voices). Dev only while
designing, every changed corpus voice read, validate once on the 500, 2 loops max. Acceptance: better
than 644 correct / 112 wrong / 378 hosts correct on validation, every regression read.

### Mini autoresearch loop on the gate — #2262 (scoreboard plan step 6)

Propose -> score on dev -> keep only if better with zero host regressions -> validate once on the 500.

- **Phase A — rules and per-show profiles.** Offline replay only, no LLM calls; builds on #2261.
- **Phase B — the LLM step (the LLM idea).** The resolution call reads each voice the way the
  labellers did: the labelling guide as the prompt, the rules as guards, a per-voice case view.
  Real DGX calls on dev, so the call budget is agreed with the operator first and gpu-mode is
  checked before any call; never in CI. Context: the LLM step is +102 correct and +41 wrong on
  validation (better 138 / worse 87), so it is where most of both the gains and the damage are.

Relation to problem 6's option 1: the `kind`-per-name field in the detection call is the first,
smallest LLM change and the cheapest to measure; Phase B is the larger one on the resolution call.

## Step 1 result — the LLM step's marginal value (2026-10-03)

Same committed code (seats + person check), replayed without vs with the LLM's stored per-voice
answers (stored FINAL answers, not raw verdicts — an approximation of the live step):

| Set | correct name | hosts correct | missing | wrong | host/guest swapped | better / worse voices |
| --- | --- | --- | --- | --- | --- | --- |
| val 500, no LLM | 502 | 316 | 572 | 78 | 20 | — |
| val 500, with LLM | 604 (+102) | 342 (+26) | 418 (-154) | 119 (+41) | 31 (+11) | 138 / 87 |
| dev 101, no LLM | 81 | 49 | 114 | 26 | 13 | — |
| dev 101, with LLM | 104 (+23) | 55 (+6) | 76 (-38) | 36 (+10) | 18 (+5) | 32 / 25 |

Reading: the LLM is the only component adding correct names in volume (102 of 604), and it also
causes about a third of the wrong names (41 of 119) and of the swaps (11 of 31): roughly one damaged
voice for every two improved. Steps 4 (guards on its answers) and 6 (the LLM loop) target exactly
that.

## Running conclusion (updated after every problem)

After 4 accepted slices and 1 rejected loop: correct names 602 -> 644 on validation (+42, of
which +8 is only the stale stored pools being rebuilt), wrong names 155 -> 112, hosts correct
342 -> 378, swaps 31 -> 21. The first two slices turned WRONG names into NO name; presenter
evidence (+11) and host pools (+21) are the first to ADD correct names, and both did it by giving
the roster better METADATA (who presents, who the feed and the episode say host), not cleverer seat
arithmetic. Each accepted slice still leaves a few new errors (host pools: 5 on validation, three
of them a guest given a host's name), and the last loop of host pools moved validation by only 4
voices: the per-episode rule tail is getting thin. Against the labeller ceiling (1,368 named
voices), the gap was 766 voices; four slices closed 42 of them (5%). The rejected loop is the
held-out set working as intended (the development set alone would have accepted it). Not yet
enough slices to call a wall; whether per-show profiles (step 5) and the LLM step (steps 4 and 6)
move more than this is the open question those steps measure.

LLM guards (step 4) were rejected after two loops. On dev they looked like the biggest single win
(wrong 29 -> 16), but the full corpus read and the 500 showed why: the text the pipeline holds for a
voice is too often SOMEONE ELSE's words (a host's greeting or question diarized into the guest's
cluster, a joke, a mid-roll), so "the voice's own words refute the LLM's name" misfires about as
often as it is right. Anchoring to the opening did not fix it, because the bleed sits exactly at
the start of the guest's voice. Lesson for step 6 (#2262): improve the LLM step itself (prompt,
candidates, a per-voice case view) rather than vetoing its answers after the fact from the same
noisy text. Steps 5 and 6 are tracked as #2261 and #2262.

Next after the planned slices (operator, 2026-10-03): a FEED-LEVEL analysis with the Fable advisor —
patterns per show (who hosts it, who recurs across its episodes, how its transcripts/voices look),
using the show sidecar (`feeds/<feed>/show.json`) and the gold sets, to find rules that work at the
show level rather than per episode. Framing (operator): at some point LOCAL (per-show) optimisation beats
GLOBAL rules — a learned per-show profile, kept honest by the same gate (validation spans 53 shows).

Then (operator, 2026-10-03): LLM-step optimisation as its own loop — prompt, candidate list and the
per-voice case view, with real DGX calls on dev, validated once on the 500, 2 loops max. The
"LLM reads like a labeller" experiment below belongs to this loop.

Candidate beyond rules (not yet tried): make the production LLM resolver read each voice the way
the labellers did (the labelling guide as the prompt, the case view as input, the rules as guards),
measured on the same gate with real DGX calls on the development set first.
