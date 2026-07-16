# Evolving the fixtures for corpus v4 — every failure we actually hit

**Status:** WIP note. Not authoritative. Written 2026-07-14, at the end of the v3 speaker/ads arc.

**Tracked by:** #1189 (golden fixtures — one per show, real diarization + feed metadata + hand-labelled
truth). Mid-roll house ads: #1188.

## Why this note exists

Every single bug in this arc was found by a **human reading output**, not by a test.

That is the finding. Not "we had bugs" — bugs are normal. The finding is that the suite was green
through all of them, and stayed green while a corpus shipped with two sports reporters hosting a
technology podcast and a lawsuit defendant speaking the words of a doctor he has never met.

So this note is not "add more tests". It is: **what class of test would have caught each thing**,
and what we are still missing. The fixture ladder at the bottom is the proposal.

---

## The failure taxonomy

### A. Silent wiring — the component works, and is connected to nothing

The dominant class. In every case the code was correct, tested, and **never reached**.

| what broke | how it announced itself |
| --- | --- |
| evidence align read `summary_provider` from args, never from the profile | silence — production grounded with a model no eval ever used |
| `model_copy` does not re-run validators, so eval cells lost their grounder | **513 insights, 0 quotes** |
| GI had no detected-person list, so `build_named_turns` never ran | every quote shipped `speaker_id: None` |
| `corroborate_guests` — thoroughly tested, never called by the pipeline | suite stayed green with the call **deleted** |
| GI invariants — thoroughly tested, never called by the stage | suite stayed green with the call **deleted** |
| registry raised `ValueError`, which `_resolve_profile` catches to mean "not a preset" | a typo would silently disable the whole registry |
| renaming `prod_dgx_only` → `experiment_dgx_only` dropped it out of a prefix match | **empty enricher set** — nine enrichers silently off, on the profile v3 was about to run |
| `surfaceable` — written by GI, read by **nobody** | unattributable insights published to the UI as somebody's opinion. Shipped by the same agent, one hour after writing the gate |
| the relabel tool wrote **no** `voice_type` at all | every voice gate INERT across the corpus. Two causes: never set, *and* `format_diarized_screenplay_with_offsets` **rebuilds** each segment dict and drops unknown keys — so setting it *before* formatting would have looked right and done nothing |
| the **eval harness** never passed `transcript_segments` to `build_artifact` | every voice gate dead in every eval run. The head-to-head that was about to decide prod-v3 would have scored a pipeline **neither model would ever run under** (2026-07-14) |

**What catches this:** mutation testing (re-break the fix, does anything go red?) and *wiring
contracts* that drive the real chokepoint rather than calling the unit under test. Both were added
this arc. The registry and enricher cases were caught by tests that **already existed and were red**
— see §F.

**And the trap inside the cure — a guard that is not a guard.** The first test written for the eval
wiring bug asserted the *source text* of `run_experiment.py` mentioned `gi_text`. Under mutation it
**survived**: replacing `gi_text = raw_text if segments else text` with `gi_text = text` — the exact
bug — still passed, because the name was still there. A test that greps for a symbol proves the
symbol exists, not that it carries the right value.

The fix was structural, not a better assertion: make the bug **unrepresentable**. `build_artifact`'s
text and its segments are one decision, so they are now returned by **one call**
(`gi_transcript_and_segments`) and cannot be mismatched by a caller. Then all four mutations went
red. *When a guard survives its own mutation, do not sharpen the guard — remove the degree of
freedom it was trying to police.*

### B. The most-trusted signal is the easiest one to poison

| signal | why it was trusted | who exploits it |
| --- | --- | --- |
| `_self_intros_by_voice` — "a voice that says *I'm X* **is** X" | the strongest per-voice evidence we have | an **ad narrator**, whose entire job is to read its own name aloud |
| the LLM speaker detector's name list | it returns `success: True` | qwen3.5:35b, naming Elon Musk as a *speaker* of an episode he is being sued in |
| "the host is the voice that opens the episode" | true of every podcast | the **pre-roll ad**, which opens it first |
| an interview cue in the description | that is how guests are introduced | an unbounded `.*?` gap, letting one cue introduce every name in the paragraph |

**The lesson generalises:** whenever a heuristic is labelled "most trusted", ask what an adversary
optimising for exactly that signal would look like. For speaker naming, the adversary is an
advertisement — and it is *already in the audio*, in every episode, for free.

**What catches this:** adversarial fixtures where the trap and the real thing appear **in the same
input** (`tests/fixtures/speaker_detection_traps.yaml`). A toy string with the cue next to the name
passes any implementation, including the broken one.

### C. Thresholds that don't survive contact with real data

Every one of these was a rule that was *correct in spirit* and wrong at the boundary.

| rule | how it broke |
| --- | --- |
| QA span softmax computed within each window's top-k | every window claimed ~1.0 → `"Codex"` at `qa_score=1.000`, and `qa_score_min` gated on nothing |
| ad voice = "under 90s of talk, only at the edges" (absolute) | on a 3-minute fixture **every** voice is near an edge → the entire cast typed as advertising |
| ad voice = "**zero** turns outside the edge window" | pyannote mis-assigned **one 2-second turn** → the ad narrator was cleared, took a host slot, and the real co-host was pushed out and given the guest's name |
| co-host = "clears `CO_HOST_INTRO_SHARE` in the first 90s" | Hard Fork's second host carries 38% of the episode and does **not** clear it |
| host count capped by the number of host *names* available | with `known_hosts` empty the cap collapsed to **one** |

**Pattern:** absolute thresholds break on scale; exact thresholds (`== 0`) break on noise. Prefer
**fractions of the thing itself** (share of talk, fraction of a voice's speech at the edges), and
always fixture **both** ends — the short episode *and* the noisy one.

### D. Eval ≠ production

| divergence | consequence |
| --- | --- |
| the eval dataset reads `.cleaned.txt`; production resolves `.adfree.txt` | the eval measured a transcript production never sees |
| the eval harness re-implemented the pipeline's provider wiring | its numbers could never match production, by construction |
| `prod-v2` (built by a commit that exists nowhere on this machine) used to answer questions about *current* code | conclusions retracted |
| the corpus-battle pilot compared gemini-on-v2-transcripts vs qwen-on-v3-transcripts | measured **transcript × LLM together**; cannot isolate the model. Its own `caveat` field says so |
| the eval harness never handed GI the diarized segments | the ad-drop, `surfaceable` and person-node gates were **all dead in eval** while live in production (see §A) |
| **offsets belong to a string, not to a file** | `char_start`/`char_end` index the RAW screenplay; eval preprocesses the text first. Feed GI the cleaned text with raw offsets and only **0–8%** of segments still resolve — every voice lookup lands on whichever speaker the shift happened to hit |

**What catches this:** a parity assertion — for every shipped profile, the resolved eval config and
the resolved production config must agree on the stack. Exists now
(`TestEvalMatchesProduction`). It does **not** yet cover which transcript *variant* is read. That
is a gap.

**The generalisation, and it is the sharpest lesson of the arc:** *the instrument must run the thing
it is measuring.* The eval was about to answer "is qwen good enough to rebuild the corpus" by
scoring a pipeline with none of the quality gates that make the corpus good. Ad copy is **written**
to be quotable — it is the most fluent, most confident false insight available — so an ungated
harness does not merely omit the gate, it actively rewards whichever model hallucinates most
eagerly on advertisements.

**A mis-attributing gate is worse than an absent one.** Hence the loader refuses segments whose
offsets no longer resolve (>=95% must hit) rather than silently trusting them. A gate that is off
is a known gap; a gate that is confidently wrong is a lie with a green check next to it.

### E. Content contamination

| thing | status |
| --- | --- |
| **house ads / cross-promos** — no sponsor language, so `_AD_PATTERNS` scores **0 hits** | **9 of 10 feeds** miss their pre-roll. Fixed structurally in the roster (`_edge_ad_voices`), not by keywords |
| **mid-roll house ads** (`Jonathan Knight`, NYT Games, 7/10 episodes) | **STILL OPEN** — sits mid-episode, so the edge rule cannot see it (#1188) |
| `anonymize_speakers` missed titled speakers (`Dr.`, `Prof.`, `Sen.`) | the *guests* — the one name a summary must never parrot — leaked while the hosts were scrubbed |
| `cleaning_v4` anonymises speakers, which is right for summarisation and fatal for GI | GI attributed **nobody**, silently. `cleaning_v5` exists for this |

### E2. "Unnamed" is four different things, and collapsing them costs a genre

Every voice we could not put a name to used to render `SPEAKER_NN`. That single bucket hid a
distinction the product depends on:

| voice_type | what it means | renders as | grounded? | surfaceable? |
| --- | --- | --- | --- | --- |
| `person` | a named human | the real name | yes | yes |
| `unidentified` | a real person **nobody in the episode ever names** — the vox-pop, the tape | "Unidentified speaker" | **yes** | no |
| `commercial` | an advertisement | "Advertisement" | **never** | no |
| `cameo` | a brief incidental voice | "Brief speaker" | yes | no |
| `unknown` | a person we **FAILED** to name | `SPEAKER_NN` | yes | no — **and counted, because a defect that costs nothing gets fixed by nobody** |

Measured over 160 episodes / 9 shows: `person` **77.45%** of talk, `unidentified` **17.57%**,
`commercial` 1.09%, `cameo` 0.37%, and `unknown` **3.52%** — the honest defect number, no longer
inflated by voices nobody could ever have named.

**Why `unidentified` must stay grounded.** On Planet Money and The Daily the tape **is the story** —
36–40% of the episode. Dropping unnamed speech outright would gut the narrated shows to protect them
from a problem they do not have. So: a fact is still a fact and stays eligible for CONNECT; only a
**STANCE** needs a name, because an unattributed stance is not a stance — nobody holds it and nobody
can disagree with it.

**And do not mint a Person for a non-person.** 19.1% of the Person nodes in the shipped corpus (69
of 361) were named `SPEAKER_NN`. #1167 filtered them back out downstream — a **mop, not a gate**, and
one that worked *only because the id happened to be ugly*. Give those voices a friendly label and it
breaks. The roster already knows the voice is not a person; the graph must not be told otherwise.

**Fixtures must cover:** a narrated episode (host + tape + ad), so that dropping `unidentified`
visibly guts it, and a `SPEAKER_NN` that must **not** become a Person node.

### F0. The instrument invented the number

I reported that **5.8%** of the corpus was "unambiguously recoverable" and set out to fix it. The
diagnostic (`attribution_ceiling.py`) used a **looser regex than the roster actually ships**, so it
counted as recoverable a great deal the pipeline could never have caught.

Honest figures: **0.88%** recoverable (not 5.8%), **6.39%** genuinely nameless (not 1.5%), ceiling
**92.15%** (not 97%).

**The rule:** a diagnostic must **import the shipped code path**, never re-implement it. A measuring
script that re-states the rule in its own words is measuring its own words. `attribution_ceiling.py`
now imports the exact `_GUEST_INTRODUCED_BY_HOST` regex the roster ships, so the diagnostic
*cannot* disagree with what we deploy.

Related, same family: quoting a number my own tool produced without asking what the tool assumed. Any
figure that motivates work must be traceable to shipped code or to a frozen artifact.

### F. Suite rot — the thing that made all of it possible

`test_openai_detector` installs a `MagicMock` into `sys.modules["openai"]` at import and can never
remove it (restoring the real module re-imports a C extension and **segfaults the interpreter**).
The mock had no `__spec__`, so every later `importlib.util.find_spec("openai")` died — **33 tests**
across unrelated modules failed on a full run and passed in isolation.

A suite that is red only when run whole is a suite people stop reading. **And it was concealing a
real bug**: `test_profile_sets` was failing because `experiment_dgx_only` had no enrichers. The test
that would have saved us was already there, already red, and already ignored.

> If you take one thing from this note: **a red suite is not a nuisance, it is an unread alarm.**

---

## What our fixtures cover today, and the blind spot

| tier | what it is | status |
| --- | --- | --- |
| 0 | **unit traps** — adversarial synthetic inputs (`speaker_detection_traps.yaml`, roster ad fixtures, GI invariants) | good, and they now carry the *real* production text |
| 1 | **mutation testing** — re-break each fix, assert something goes red | done **ad hoc**, in a scratch script. Not repeatable. |
| 2 | **wiring contracts** — drive the real chokepoint, assert the guard is *reached* | thin. One for speakers, one for GI. |
| 3 | **golden diarization** — real pyannote clusters + hand-labelled ground truth | **MISSING. This is the blind spot.** |
| 4 | **corpus-level invariants** — cross-episode checks | **MISSING** |
| 5 | **eval↔production parity** | partial (providers yes, transcript variant no) |

**The blind spot is tier 3, and it is where every roster bug lived.** Both roster failures — the
stray-turn escape and the guest-name-on-a-host cascade — were invisible to synthetic fixtures and
appeared *immediately* on real clusters. Synthetic diarization is too clean: it has no
mis-assignments, no 2-second strays, no 8-voice episodes where one voice is 0.3% of the talk.

---

## The proposal: the v4 fixture ladder

### Tier 3 — golden diarization fixtures (highest value, do first)

Freeze the **real** pyannote output for a handful of episodes and hand-label the truth.

- **Input:** the actual diarization turns (`start`, `end`, `speaker`) — already on disk, no GPU to
  regenerate — plus each voice's own text.

- **Ground truth:** for every voice: `host | guest | advertisement | cameo`, and the correct name.
- **Assert:** the roster reproduces it exactly.
- **Cost:** zero GPU, milliseconds to run, and it is the only tier that would have caught the two
  bugs that shipped.

Seed it with the cases we already know are hard:

1. **Hard Fork ep1** — pre-roll ad (2 voices), mid-roll ad, 2 hosts, 2 guests, one 2-second
   mis-assigned turn. This single episode is a better test than the entire synthetic suite.
2. An episode with **no guests** (news round-up) — the roster must not invent one.
3. An episode where a **guest out-talks the host** (#1169) — the swap case.
4. A **panel** (3+ real speakers).
5. A **short** episode (<10 min) — the scale guard.

### Tier 4 — corpus-level invariants

Some truths are only visible **across** episodes. These would have caught the ads instantly.

- **An advertisement repeats; editorial does not.** Near-identical text blocks recurring across
  episodes of a feed are ads. This is the proposed fix for the mid-roll (#1188), and it is exact:
  `Jonathan Knight` reads the *same copy verbatim in 7 of 10 episodes*.

- **A host recurs; a guest usually does not.** A "host" appearing in exactly one episode is
  suspicious. `Amy Lawrence` appeared in **10/10** with **0.3% talk share** — a host who never
  speaks is a contradiction, and it is checkable.

- **A person's talk share should be consistent with their role.** A `host` under 5% of talk, or a
  `guest` over 50%, is a smell.

- **No corpus-wide near-duplicate names.** `Kevin Roos` / `Kevin Russo` / `Kevin Roose` are one man.

Run these as a **corpus audit** over the rebuilt 10 episodes and fail the build on violations.

### Tier 1 — make mutation testing a real target

`make mutants` — a checked-in list of `(file, correct_code, broken_code, test_selection)`, run in
CI on demand. It found **6 unguarded fixes out of 11** and it must not stay in a scratch file.

### Tier 5 — close the eval↔production gap

Extend the existing parity test to assert the eval dataset reads the **same transcript variant**
production resolves (`.adfree.txt`, not `.cleaned.txt`).

---

## Acceptance gate for v4

Before a 100-episode build is authorised:

1. Tier-3 golden diarization fixtures exist and pass, including the Hard Fork ep1 case.
2. The tier-4 corpus audit runs over the 10-episode rebuild with **zero** violations —
   specifically: no ad narrator named, no host under 5% talk share, no near-duplicate person names.
3. `make mutants` is green.
4. `make test-unit` and `make test-integration` are green **run whole**, not just in isolation.
5. The mid-roll ad (#1188) is either fixed or explicitly accepted, in writing, with its blast radius
   measured.

---

## The anatomy of a real episode — and why our fixtures are lying to us

Our synthetic fixtures are two speakers alternating politely for three minutes. **No real episode
looks like that**, and every bug we shipped lived in the difference.

This is Hard Fork ep1, measured from the actual diarization (2-minute blocks, dominant voice):

```text
  0:00 – 0:30    PRE-ROLL AD        two voices, neither ever heard again, both self-introducing
  0:30 – 30:00   HOSTS              Kevin/Casey alternating every 2–6 min: banter, then news
 ~36:00          MID-ROLL AD        one voice, ~50s, self-introducing, same copy in 7/10 episodes
 30:00 – 52:00   GUEST INTERVIEW    the guest holds the floor for 22 CONTIGUOUS MINUTES
 52:00 – 56:00   HANDOFF            back to hosts
 56:00 – 70:00   SECOND GUEST       a different guest, a different segment
 70:00 – 72:00   OUTRO + CREDITS    hosts, then producer names nobody speaks
```

Eight voices. Two of them are adverts. Two are hosts. Two are guests. One is a 15-second cameo. One
is a mid-roll advertiser. **A two-speaker toy fixture cannot express a single one of the failures
this arc produced.**

### What each structural feature does to us

| feature | what it breaks |
| --- | --- |
| **pre-roll ad opens the episode** | "the host is the opening voice" hands the show to the advertiser |
| **ad narrators self-introduce** | the most-trusted naming signal is the one the ad is built to trigger |
| **a guest holds the floor for 22 min** | the guest out-talks the host over any short window → intro-share co-host detection swaps them (#1169) |
| **hosts alternate in 2–6 min blocks** | neither host dominates the intro window → a real co-host fails `CO_HOST_INTRO_SHARE` and falls through to *guest* naming |
| **rapid banter, tiny backchannel turns** ("Yeah." "Right.") | dozens of sub-second turns; one mis-assigned 2-second turn was enough to defeat the ad rule |
| **mid-roll ad, 35–40% in** | outside any edge window — invisible to a structural edge rule (#1188, still open) |
| **credits name people who never speak** | producers/editors are perfect NER bait |
| **the same show is sometimes a guest feed** (ep6 is an Ezra Klein cross-post) | the "hosts" of the episode are not the hosts of the feed |

### Why this is *the* grounding problem, not just a roster problem

Speaker attribution is derived from the **character offset** of the grounded quote. So the accuracy
of grounding is bounded by the accuracy of the turn boundaries, and the danger is not uniform across
the episode:

- **Inside the 22-minute guest block** — attribution is *forgiving*. An offset can be wrong by a
  paragraph and still land on the right speaker.

- **In the banter** — attribution is *brutal*. Turns are seconds long, so an offset that is wrong by
  one sentence flips Kevin into Casey. This is where an insight silently changes mouths.

- **At a handoff** (52:00) — the host and guest are adjacent, so an off-by-one-turn error swaps a
  guest's claim onto a host. This is the single most dangerous position in the episode and we do not
  test it at all.

- **Inside an ad** — a quote grounded here is *always* wrong, and worse, it is fluent, confident,
  and quotable ("If you play our games, you probably know there's something a bit different about
  them"). Ad copy is written to be memorable. It is the perfect false insight.

> The grounding invariant we actually want: **a quote may never be grounded inside an ad region, and
> a quote adjacent to a turn boundary must be attributed conservatively or not at all.**

### The fixture we should build: one realistic show, not twenty toy ones

Build a single **canonical episode fixture** with the structure above, and derive the variants from
it. It costs nothing to run (it is timings + text, no audio, no GPU) and it exercises every failure
mode at once:

```text
episode: hardfork_canonical
  0:00–0:30    AD_1, AD_2      self-introduce; never return          -> must be "Advertisement"
  0:30–30:00   HOST_A, HOST_B  alternate in 2–6 min blocks           -> both must be hosts
  ~36:00       AD_3            self-introduces; ~50s; mid-roll       -> must NOT be a guest  (#1188)
  30:00–52:00  GUEST_1         22 contiguous minutes                 -> must be a guest, named
  52:00–56:00  HOST_A          handoff                               -> boundary attribution
  56:00–70:00  GUEST_2         second segment                        -> not introduced in description
  70:00–72:00  HOST_A, HOST_B  outro + credits naming non-speakers   -> credits must not become guests
```

Assertions it should carry, all of which we have shipped bugs against:

1. `AD_1`, `AD_2`, `AD_3` are never named, and never appear as `person` / `host` / `guest`.
2. `HOST_A` and `HOST_B` are both hosts — even with `known_hosts` empty, even though neither
   dominates the intro window.
3. `GUEST_1` gets the guest name; **no host ever wears it**.
4. `GUEST_2` is *not* in the episode description → the corroboration gate drops the name → the voice
   must fall back to its own self-introduction, or stay `SPEAKER_NN`. It must **not** inherit
   `GUEST_1`'s name.
5. Every producer named in the credits is absent from the roster.
6. A quote taken from inside `AD_3` is refused by grounding.
7. A quote taken 1 second either side of the 52:00 handoff is attributed correctly, or not at all.

Then perturb it — this is where the real bugs were:

- **`+noise`**: reassign one 2-second turn from `AD_1` to a random mid-episode position. (This exact
  perturbation, in real pyannote output, put the guest's name on a host.)

- **`+short`**: the same show cut to 4 minutes. (Guards the scale rule.)
- **`+swap`**: the guest answers at length inside the first 90 seconds. (#1169.)
- **`+solo`**: no guests at all — a news round-up. (The roster must not invent one.)
- **`+panel`**: three real guests.
- **`+crosspost`**: the episode is another show's feed (ep6) — the feed's hosts are not the episode's.

### How this changes what we ask the LLM for

The same realism argument applies to the *dialogue* we feed the extraction prompt. Today the GI
prompt sees a flat screenplay. It does not know:

- **where the ads are** (so it can quote one),
- **who is the host and who is the guest** (so it cannot tell a question from a claim), or
- **which segment it is in** (so it cannot tell "we're doing the news" from "we're interviewing an
  expert").

A host asking *"So is prescribing via chatbot actually unsafe?"* and a guest answering *"Prescribing
via chatbot is unsafe"* produce nearly identical sentences — and only one of them is a STANCE. That
is not a prompt-tuning problem, it is a **missing input** problem. Feeding the extractor the segment
structure (`ad` / `banter` / `interview` / `outro`) and the speaker's *role*, not just their name,
is the highest-leverage change available to insight quality — and we already compute all of it in
the roster.

---

---

## G. Identity is READ, not inferred — and the fixtures must force that

Added after the roster was rebuilt a third time. This is the failure I repeated most, so it gets its
own section: **five times in three days I ignored the metadata and built a statistical rule instead.**

### What went wrong

I inferred hosts from talk share and span — "a host talks a lot and is present from the first minute
to the last". That is not a property of podcasts. It is a property of **one** podcast, and it
inverts by format:

| show | who talks most | who actually hosts |
| --- | --- | --- |
| Invest Like the Best | the **guest**, 82% | Patrick O'Shaughnessy, **17%** |
| Latent Space | the **guest**, 84.5% | Brandon, **8.6%** |
| Hard Fork | the hosts, 26–39% | the hosts |
| Hard Fork | the episode is **opened by a pre-roll ADVERT** | not a host at all |

Any rule keyed on "who talks most" or "who opens" is tuned to whichever show it was written against.
Meanwhile **the feed says who hosts it, in plain English** — 7 of our 10 state it outright:

```text
Hard Fork      "journalists Kevin Roose and Casey Newton explore..."
The Journal    "Hosted by Ryan Knutson and Jessica Mendoza."
No Priors      "co-hosts Elad Gil and Sarah Guo talk to..."
Odd Lots       "Bloomberg's Joe Weisenthal and Tracy Alloway explore..."
Invest Like…   in the TITLE: "Invest Like the Best with Patrick O'Shaughnessy"
```

### The contract the fixtures must now enforce

| source | answers | must never |
| --- | --- | --- |
| **metadata** (feed title/description/authors; episode description) | **WHO** the hosts are, and **HOW MANY** | be overruled by a statistic |
| **the conversation** (transcript) | **WHICH VOICE** each one is; and the whole answer when the feed is silent | promote a voice that performs no role |
| **talk share** | **ad vs person** only (30 seconds vs 20 minutes — that gap does not invert) | ever separate host from guest |

The role is **performed**: the host welcomes you to the show and introduces the guest; the guest says
thanks for having me. That is format-independent, and it is what finally worked.

### The cases the fixtures must simulate

Every one of these is real, and each is a distinct trap. A fixture suite that omits any of them will
let the same class of bug back in.

| # | case | real example | what it breaks |
| --- | --- | --- | --- |
| 1 | **guest out-talks the host 5:1** | Invest Like the Best (82% / 17%) | any talk-share host rule |
| 2 | **host holds <10% of the episode** | Latent Space (Brandon, 8.6%) | "the host talks a lot" |
| 3 | **the labels are SWAPPED** | NVIDIA — the cluster labelled `Nicolas Cerisier` says *"I'm Noah Kravitz. My guest is Nicolas Serissier"* | everything downstream; only the conversation reveals it |
| 4 | **the feed names NO host** | Planet Money, Latent Space, NVIDIA (3/10) | metadata-only designs |
| 5 | **the description lists PAST GUESTS** | Latent Space — Bret Taylor, Chris Lattner, George Hotz | NER-over-description. A better NER does **not** fix it: they *are* people |
| 6 | **the show's own name looks like a person** | "At **Planet Money**, we explore..." | naive capitalised-run extraction |
| 7 | **publisher possessive glued to the name** | "**Bloomberg's** Joe Weisenthal" | name cleaning |
| 8 | **the host is only in the TITLE** | "Invest Like the Best **with Patrick O'Shaughnessy**" | description-only parsing |
| 9 | **an ad narrator self-introduces** | "I'm Paul Tenorio. I cover soccer for The Athletic." | the most-trusted naming signal |
| 10 | **a host's turn merges into a guest's cluster** | Hard Fork — briefly produced a THIRD host | uncapped conversation-derived roles |
| 11 | **rotating/narrated hosts, 13 voices** | Planet Money | two-speaker assumptions |
| 12 | **a cross-post: the feed's hosts are not the episode's** | Hard Fork ep6 is an Ezra Klein show | feed-level `known_hosts` applied blindly |

### How to build them: one golden fixture per SHOW, plus a metadata axis

The tier-3 golden fixtures (real diarization + hand-labelled truth) need a **second half**: the
metadata each episode was resolved against. A roster fixture without its feed metadata is testing
half the system, and it is the half that was never the problem.

Each show's fixture should therefore carry:

```yaml
show: hard_fork
feed:
  title: "Hard Fork"
  description: "...journalists Kevin Roose and Casey Newton explore..."
  authors: ["The New York Times"]          # an ORG — must not become a host
expected_hosts_from_feed: ["Kevin Roose", "Casey Newton"]   # and HOW MANY: 2

episode:
  description: "...a trial against Elon Musk... Dr. Adam Rodman, of Harvard, returns..."
expected_guests_after_gate: ["Dr. Adam Rodman"]             # Musk/Altman rejected

diarization:            # the REAL pyannote turns, frozen. No audio, no GPU.
  - {start: 0.0,   end: 15.0,  speaker: SPEAKER_04}
  - ...
voice_texts:            # each cluster's own words — where the role is PERFORMED
  SPEAKER_04: "I'm Paul Tenorio. I cover soccer for The Athletic."
  SPEAKER_07: "I'm Kevin Russo and this is Hard Fork."          # ASR mangles the name

expected_roster:
  SPEAKER_04: {role: advertisement, named: false}
  SPEAKER_07: {role: host,  name: "Kevin Roose"}               # snapped to the FEED's spelling
  SPEAKER_03: {role: guest, name: "Dr. Adam Rodman"}
```

**The perturbations, and which case each one guards:**

| perturbation | how | guards |
| --- | --- | --- |
| `+no_feed_hosts` | blank the feed description | 4 — forces the conversation path |
| `+swapped` | exchange two clusters' texts | 3 — the NVIDIA bug |
| `+quiet_host` | shrink the host's turns to <10% | 2 |
| `+loud_guest` | grow the guest's turns to >80% | 1 |
| `+stray_turn` | move one 2-second turn of an ad voice to mid-episode | the cascade that put a guest's name on a host |
| `+guest_soup` | add "Past guests include ..." to the feed description | 5 |
| `+merged_cluster` | append a host sentence to the guest's `voice_texts` | 10 — the third-host bug |
| `+short` | cut the episode to 4 minutes | the scale rule |
| `+solo` | remove the guest entirely | the roster must not invent one |
| `+crosspost` | swap in another show's hosts | 12 |

Each perturbation is a **one-line edit to a frozen fixture**, costs no GPU, and reproduces a bug that
actually shipped. That is the whole point of doing it this way rather than adding more toy strings.

### The assertion that would have caught everything

> **No name may appear in the roster that is not either (a) stated in the feed, (b) stated in the
> episode description and corroborated, or (c) spoken in the transcript by the voice it is assigned
> to.**

`Amy Lawrence`, `Paul Tenorio`, `Elon Musk`, `Sam Altman`, `Tim Cook` — every single one fails that
test. It is checkable, it is cheap, and it belongs in the corpus audit (tier 4).

---

## Open items

- **#1188 mid-roll house ads** — `Jonathan Knight` still enters the corpus as a *guest* of Hard Fork
  with `GUESTS_ON` edges. He does not displace a real guest and no longer poisons host detection, so
  it is corpus pollution rather than misattribution. Proposed fix: cross-episode repetition (tier 4).

- **Under-attribution is now the failure mode.** The corroboration gate drops guests whose names
  appear only in the transcript, not the description (`David Duvenaud`, `Andrew Marantz`). The
  roster's per-voice self-intro is the safety net. This trade is deliberate (#876 — a wrong name is
  worse than no name), but it should be **measured**, not assumed.

- **The eval harness needs the same parity assertion the profiles have** (§D). `TestEvalMatchesProduction`
  compares *configs*; it did not notice that the harness passed no `transcript_segments` at all, so
  every voice gate was dead in eval while live in production. Fixed for GI
  (`gi_transcript_and_segments`, guarded by mutation-tested wiring contracts). **KG branch audited
  (2026-07-16): no parity gap.** `kg.build_artifact` has no `transcript_segments` parameter and
  production KG (`metadata_generation.py`) never passes one — KG extracts entities/topics from the
  full transcript text plus `detected_hosts`/`detected_guests` names, and has no per-speaker voice
  gate like GI's evidence grounding. So the eval KG branch correctly passes no segments; there is
  nothing to wire, and adding the argument would be dead weight. Item closed.

- **The 6.39% nobody-names-them bucket.** Now honestly classified as `unidentified` rather than
  counted as our defect (§E2). It is a real ceiling, not a bug — but it is worth revisiting whether
  a narrated-show fixture could recover any of it from the narrator's own framing ("*a school
  teacher in Ohio told us…*").
