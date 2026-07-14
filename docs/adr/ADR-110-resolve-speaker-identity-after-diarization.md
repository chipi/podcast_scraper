# ADR-110: Ask who speaks AFTER we can hear them

**Status:** Proposed
**Date:** 2026-07-14
**Deciders:** Marko
**Related:** #876 (Elon Musk named as a speaker), #1169 (speaker/person quality), ADR-109

## Context

### The pipeline asks the identity question before the answer exists

```text
processing.py:827         _detect_speakers_for_episode()   <-- "who are the speakers?"
episode_processor:2082    download_media_for_transcription()    media not downloaded yet
episode_processor:1613    transcribe_with_segments()            no transcript yet
episode_processor:1632    apply_diarization_to_result()         no voices yet
```

And the interface cannot take a transcript even if we wanted to give it one:

```python
def detect_speakers(self, episode_title, episode_description, known_hosts)
```

So we ask an LLM *"who speaks in this episode?"* while showing it **only the show notes**, before a
single word of audio has been transcribed. Nobody could answer that from that evidence. The model
does the only thing available to it: it returns the people the notes **mention**. That is how
`Elon Musk` — named in a Hard Fork description solely as the man *suing* OpenAI — was returned as a
speaker, assigned to a voice cluster, and published as the author of a real guest's words (#876).

### The gate we built to stop it is circular

`corroborate_guests` checks the model's proposal **against the same show notes the proposal was
derived from**. Validating a claim against the evidence it was guessed from can only catch names
invented from nowhere; it cannot distinguish a person who *speaks* from a person who is *discussed*,
because the text being checked does not say. So it falls back to a regex — "is there an interview
cue next to this name" — and desk shows never write one.

**Measured, via the detector prod actually runs (gemini-2.5-flash-lite), over 50 episodes / 5 shows:**

| | |
| --- | --- |
| mean unattributed talk per episode | **24.0%** |
| episodes with >= 25% of talk attributable to nobody | **21 / 50** |
| names gemini proposed that corroboration DELETED | **70** |
| ...of which whole first+last names | **69 of 70** |

Who we deleted:

```text
5x Sierra Juarez      3x Alex Goldmark     2x Marianne McCune    1x Joe Leahy
4x Rob Armstrong      3x Jess Jiang        2x Jimmy Keeley       1x Robin Wigglesworth
3x Cena Loffredo      3x Emma Peaslee      2x Jay Powell         1x Richard Waters
3x Robert Rodriguez                        1x Kevin Warsh
```

**Rob Armstrong is the co-host of FT Unhedged.** Alex Goldmark is Planet Money's executive producer;
Jess Jiang, Sierra Juarez, Marianne McCune and Emma Peaslee are its reporters. They are in the
episode, talking. The gate also correctly deletes Jay Powell and Kevin Warsh, who are discussed and
absent — but it cannot tell the two groups apart, so it deletes the newsroom to catch the Fed chair.

**The extractor was never the bottleneck.** 69 of 70 proposed names are whole and correct. We throw
them away.

### And the thing the gate was holding back

The gate has to be that strict because downstream, `_name_guest_voices` **paints surviving names
onto voices positionally, in talk-time order**. *That* is the invention mechanism — a name lands on a
voice with no evidence tying it there. The gate is a sandbag in front of that hole.

## Decision

**Move the identity decision to where the evidence is: after diarization.**

1. **Keep the pre-transcription detector as a cheap PROPOSAL.** Metadata-only is fine for the job it
   can actually do — listing the people the episode names. It is good at it (69/70 whole names).
   Nothing about its interface changes.

2. **Add a post-diarization RESOLUTION step** (`resolve_speakers_from_conversation`), given what the
   question requires:
   - the names the metadata stated (detector + feed + config hosts),
   - the first ~60 seconds of **each voice's own turns**,

   and asked the question that has an answer:

   > For each VOICE, which stated person is it — and which stated people never speak?

   The model must point at a voice and justify it from that voice's words. It cannot paint a name on
   a voice it has not heard.

3. **Never bind a name to a voice without evidence.** After resolution, a name attaches to a voice
   only via: its own self-introduction, an on-air introduction by the host, the LLM resolution above,
   or a **forced** match (exactly one candidate name and exactly one candidate voice — no choice, so
   no guess). **Delete positional talk-time painting.** Anything unbound stays `unknown`: an honest
   defect, never a wrong name.

4. **The regex path stays, as the no-LLM fallback.** `airgapped`, `local`, `dev` and
   `reprocess_dgx_no_llm` run `speaker_detector_provider: spacy` and keep the deterministic cue
   matcher exactly as it is. LLM where there is an LLM; deterministic cues where there is not —
   the same split diarization and summarization already use.

## Alternatives considered

### A. Relax the corroboration regex

Add more cue patterns until the reporters survive. **Rejected.** It treats the symptom: the gate is
being asked to decide, from show notes, something show notes do not state. Every new cue widens the
door for `Jay Powell` at the same rate as it admits `Rob Armstrong`, because to a regex they look
identical. The precision/recall trade has no good point on it.

### B. The "anchor" rule — one confirmed guest vouches for the people named beside him

*Built, measured over 160 episodes, and removed on 2026-07-14.* It admitted 8 names: 3 real guests
(Qasar Younis, Dan Gural, Marc Andreessen) and 5 who were never in the room — including **HB Reese**,
the founder of Reese's, *discussed* by a Planet Money episode, **dead since 1956**, given a voice.
Plus a bare "Marc" landing on a second voice in the Marc Andreessen episode, and "Bill" / "Er" on
voices with 0.0% of the talk.

That is #876 rebuilt. It is preserved as a **counter-example test**
(`test_the_metadata_NEVER_names_a_voice_that_did_not_say_it`) so nobody re-derives it.

### C. Swap spaCy for an LLM as the NER method

**Rejected — this is already the shipped design, and it is not the problem.** `prod_dgx_*` already
run gemini; `experiment_dgx_only` runs ollama; spaCy is already the airgapped option. The LLM is
already producing near-perfect names. Changing the proposer cannot help when the loss happens after
the proposal.

## Consequences

**Good.** The identity question is asked where the answer exists. The corroboration regex stops
deleting real hosts. Positional painting — the mechanism behind every wrong name we have shipped —
is gone. `unknown` becomes a true defect count because the alternative to a name is now silence, not
a guess.

**Cost.** One extra LLM call per episode, after diarization (~160 calls per corpus rebuild; at
flash-lite prices this is cents). Airgapped profiles pay nothing and lose nothing.

**Risk.** The LLM could still misbind a voice. Mitigated by: it must choose from the STATED names
only (it cannot invent), self-intro and on-air introduction still win over it where they exist, and
the corpus warrant audit (`scripts/audit/corpus_speaker_audit.py`) replays the whole 160-episode
corpus with zero GPU on every rule change — every name it admits gets read before it ships. That
audit is what caught HB Reese.

**Not addressed here.** Cold opens (an episode that starts with a guest teaser clip defeats
"the opening voice is the host"). Separate issue, separate fix.
