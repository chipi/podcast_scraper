# Evolving the fixtures for corpus v4 — every failure we actually hit

**Status:** WIP note. Not authoritative. Written 2026-07-14, at the end of the v3 speaker/ads arc.

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
|---|---|
| evidence align read `summary_provider` from args, never from the profile | silence — production grounded with a model no eval ever used |
| `model_copy` does not re-run validators, so eval cells lost their grounder | **513 insights, 0 quotes** |
| GI had no detected-person list, so `build_named_turns` never ran | every quote shipped `speaker_id: None` |
| `corroborate_guests` — thoroughly tested, never called by the pipeline | suite stayed green with the call **deleted** |
| GI invariants — thoroughly tested, never called by the stage | suite stayed green with the call **deleted** |
| registry raised `ValueError`, which `_resolve_profile` catches to mean "not a preset" | a typo would silently disable the whole registry |
| renaming `prod_dgx_only` → `experiment_dgx_only` dropped it out of a prefix match | **empty enricher set** — nine enrichers silently off, on the profile v3 was about to run |

**What catches this:** mutation testing (re-break the fix, does anything go red?) and *wiring
contracts* that drive the real chokepoint rather than calling the unit under test. Both were added
this arc. The registry and enricher cases were caught by tests that **already existed and were red**
— see §F.

### B. The most-trusted signal is the easiest one to poison

| signal | why it was trusted | who exploits it |
|---|---|---|
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
|---|---|
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
|---|---|
| the eval dataset reads `.cleaned.txt`; production resolves `.adfree.txt` | the eval measured a transcript production never sees |
| the eval harness re-implemented the pipeline's provider wiring | its numbers could never match production, by construction |
| `prod-v2` (built by a commit that exists nowhere on this machine) used to answer questions about *current* code | conclusions retracted |
| the corpus-battle pilot compared gemini-on-v2-transcripts vs qwen-on-v3-transcripts | measured **transcript × LLM together**; cannot isolate the model. Its own `caveat` field says so |

**What catches this:** a parity assertion — for every shipped profile, the resolved eval config and
the resolved production config must agree on the stack. Exists now
(`TestEvalMatchesProduction`). It does **not** yet cover which transcript *variant* is read. That
is a gap.

### E. Content contamination

| thing | status |
|---|---|
| **house ads / cross-promos** — no sponsor language, so `_AD_PATTERNS` scores **0 hits** | **9 of 10 feeds** miss their pre-roll. Fixed structurally in the roster (`_edge_ad_voices`), not by keywords |
| **mid-roll house ads** (`Jonathan Knight`, NYT Games, 7/10 episodes) | **STILL OPEN** — sits mid-episode, so the edge rule cannot see it (#1188) |
| `anonymize_speakers` missed titled speakers (`Dr.`, `Prof.`, `Sen.`) | the *guests* — the one name a summary must never parrot — leaked while the hosts were scrubbed |
| `cleaning_v4` anonymises speakers, which is right for summarisation and fatal for GI | GI attributed **nobody**, silently. `cleaning_v5` exists for this |

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
|---|---|---|
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

## Open items

- **#1188 mid-roll house ads** — `Jonathan Knight` still enters the corpus as a *guest* of Hard Fork
  with `GUESTS_ON` edges. He does not displace a real guest and no longer poisons host detection, so
  it is corpus pollution rather than misattribution. Proposed fix: cross-episode repetition (tier 4).
- **Under-attribution is now the failure mode.** The corroboration gate drops guests whose names
  appear only in the transcript, not the description (`David Duvenaud`, `Andrew Marantz`). The
  roster's per-voice self-intro is the safety net. This trade is deliberate (#876 — a wrong name is
  worse than no name), but it should be **measured**, not assumed.
