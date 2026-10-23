# Naming ladder — the rungs, the decision trace, and how to reconstruct any episode

| | |
| --- | --- |
| Doc version | 37 |
| Last updated | 2026-10-23 |
| Trace schema | `TRACE_VERSION = 1` (`src/podcast_scraper/providers/ml/diarization/naming_trace.py`) |
| Implemented by | `448151fab` (roster trace), `dbed527d2` (rung tests), `b5923aab4` (LLM answers), `c8bda243d` + `11e75c7ab` (inside the helpers) — #2276. Rules since: `93f49482c` (`introduced` source), `c05cc0273` + `2b7aab46e` (one person, one entry), `aee17d79e` (credential without a comma), `64654a8c9` (hailed guest), `b3fb1c941` (narrator seat), `66eaedd0c` (Phase C), `ee4d19961` (K8), `2fd6411dc` (F7 storyteller), `19da42c29` (new promos, stand-in, opener hail), `c2ac9fcfa` (commercial talk-share guard), title spelling references (v23), promo rule fixes (v24), question-lead host seat (v25), voice identity rung + talk floor + split-host record source (v26), transcript naming (v27), transcript naming shadow mode (v28), voice identity shadow mode (v29), E1d + S30s rescues (v30), transcript naming in the bundled calls (v31), naming passes (v32), split-voice merge (v33), candidate widening (v34), review fixes (v35, v36), widening name words (v37) |
| Known issues | [Naming ladder — known issues and limits](NAMING_KNOWN_ISSUES.md); how changes are judged: [naming gate runbook](NAMING_GATE_RUNBOOK.md) |
| Code content hash | `7d76fa48bb8f` (see below) |
| Status | Phase 1 **deployed**: prod sidecars written since the deploy carry `decision_trace` (114 of 2,451 served episodes with diagnostics on 2026-10-08; older ones were never re-ingested). `roster_replay --trace-out` reconstructs the rest offline |

**How to tell whether this doc matches the code.**

- `tests/unit/podcast_scraper/providers/ml/diarization/test_naming_trace.py` reads the trace schema
  row above. It fails when that row differs from `TRACE_VERSION`. A schema change cannot land
  without this doc.
- The content hash is the first 12 hex digits of the sha256 of `naming_trace.py`, `roster.py` and
  `pipeline.py`, concatenated in that order:

  ```sh
  cat src/podcast_scraper/providers/ml/diarization/{naming_trace,roster,pipeline}.py | shasum -a 256 | cut -c1-12
  ```

  A different value means the ladder or the recorder has changed since this doc was last checked
  against it. Re-read the "Recorded" and "NOT recorded" sections before you rely on them.

**Changelog**

- **v37, 2026-10-23.** Candidate widening reads a name with an inner capital or a hyphen whole.
  Its word pattern allowed only lowercase after a word's first letter, so "Jade McGlynn" was read
  as "Jade Mc" and "Aled Maclean-Jones" as "Aled Maclean-", and those fragments passed the publish
  gate onto the candidate list. A word now takes one inner capital ("McCormick", "O'Brien",
  "DeShawn") and hyphenated parts, never a trailing hyphen. Found by the DGX-block test of R6, on
  the held-out set before any run. Still off by default.
- **v36, 2026-10-23.** Fixes from the second branch review; no measured rule changes.
  - **Candidate widening: "joins / returns / featuring" present a person only when nothing else
    is said about them.** v35 still added "Why Elon Musk" from "Why Elon Musk joins the AI
    race", and "Taylor Swift" from "Taylor Swift returns to Nashville" and "featuring Taylor
    Swift songs". Now "joins / returns / is back" count only with no object or the show as the
    object ("joins us", "joins the show"); "featuring" only when the name closes the clause or a
    list; and a phrase starting with a question or sentence opener ("Why", "How", "Is", "Meet"…)
    is no name. Still off by default and unmeasured.
  - **Transcript naming cannot fail the episode.** The ask is wrapped as the voice rung is: if it
    raises, the ladder's names stand and the sidecar records `ok: false` with the error. Before,
    this held on prod's provider by inspection, not by construction.
  - **The voice rung counts the siblings it read and skipped** (`siblings_read`,
    `siblings_skipped` in its report), and logs one warning per episode when any were
    unreadable, so a feed whose siblings all fail to parse no longer reads as "no voted group".
  - **An E1d rescue from the voice's own introduction is self-introduced in the speaker record.**
    E1d keeps its source `e1d_rescue`; its trace step's `cue: self_intro` is the same evidence as
    `self_intro`, so a rescued host whose name is the show's keeps the eponymous-host exception
    (`metadata_generation._self_introduced_voices`).
  - **`voice_identity.MIN_VOICE_TALK_S`** (10 s) names the threshold `decide` already used; the
    shadow report reads it to tell judged voices from unjudged ones.
- **v35, 2026-10-23.** Fixes from the branch review; no rule's measured behaviour changes.
  - **Voice identity reads siblings whose metadata stores an absolute path.** Prod metadata
    often stores `transcript_file_path` as written at ingest (`/app/output/feeds/<feed>/<run>/
    transcripts/x.txt`). On a corpus mounted elsewhere (a snapshot copy, a restore) those
    siblings were skipped silently, and one skipped sibling can cost a whole group (a group needs
    two episodes). `voice_identity.sibling_diagnostics_path` now takes the part after the run
    directory's name, as the migrations' `_run_relative` does. In the prod container the paths
    resolved already, so served shadow records are unaffected.
  - **A side namer cannot fail the episode.** A sibling sidecar of the wrong shape (not an
    object, a non-numeric fingerprint) is skipped. If the voice rung raises, the roster before it
    stands and its sidecar says `abstained: "error: <type>"`. If the side namers' sidecar entries
    cannot be built, they are not written and the episode is published as without them.
  - **Candidate widening: a bare "with" counts only where it opens the text or a clause.** v34
    said text ABOUT a person is refused, but a bare "with" anywhere matched, so "his feud with
    Elon Musk" added the name. Now a bare "with" counts at the start of the text or after
    `- — – : | ( , ;`, or after a conversation word ("a conversation with", "talks to", "sits
    down with", "catches up with"). "Guest", "joined by", "featuring" and "<name> joins" are
    unchanged. A title "The Future of AI with Ana Ruiz" no longer adds the name: a wrong name is
    worse than a hidden one. Still off by default and unmeasured.
  - **`transcript_naming: publish` in a bundled mode publishes nothing more than `shadow`** until
    the bundled write-back exists (v31); the config field now says so.
  - **Cost label.** Gemini's `complete_text` defaulted to stage `speaker_resolution`; every
    provider now defaults to `speaker_detection`. Since d47a256ae every provider records the
    naming call's cost (v28 said "OpenAI-compatible providers and Gemini").
  - **The voice sidecar's `ladder` column is the roster published before the rung.** With
    `transcript_naming: publish` and `voice_identity_rung: shadow` it would hold the LLM pass's
    names; no profile sets that combination.
  - **Bundled shadow and the summary text.** In a bundled mode with transcript naming on, the
    bundled call carries the speakers field, and `take_speakers` re-serialises the parsed JSON
    into `summary` after taking it out: the same content, different bytes than the model wrote.
    The bundled prompt itself also differs from off (v31). No prod profile is bundled.
  - **Removed:** `transcript_naming.transcript_block` and `parse_answer`, used only by tests.
    The layout and prompt tests now read the prompt `fit_prompt` builds, the one production
    sends.
- **v34, 2026-10-23.** `widen_candidates_from_metadata` (default off) adds to the closed candidate
  list the names the episode's own title and description present as people on it, which
  detection missed (`providers/ml/diarization/candidate_widening.py`).
  - **What counts as "on it":** a name after "with", "guest", "joined by", "featuring",
    "interview with", "conversation with", "talks to / with", "sits down with", or followed by
    "joins" / "returns".
  - **What is refused:**
    - text ABOUT a person (#876);
    - a phrase starting with "The", "Our" and the like;
    - a name the list already has, by spelling or by surname plus first-name start;
    - a name the publish gate refuses in the feed's language;
    - a name `plausible_person_names` drops.
  - **Trace:** the added names go in `decision_trace.inputs.candidate_widening`, only when on.
  - **Why:** of the unnamed nameable voices, 5 were a name in the episode's own title or
    description that detection missed (eval repo NAMING-RETHINK, M2).
  - **Not measured yet;** that is the DGX block.

- **v33, 2026-10-23.** `merge_split_voices` (default off; `merge_split_voices_threshold` 0.8)
  merges one person split into two diarized voices, before alignment and naming. No rung changed.
  - **The rule.** Two voices of one episode, each speaking >= 10 s, whose fingerprints are at
    least the threshold similar (cosine) are grouped. A chain of matches is one group. Each group
    becomes its voice that talks most, so every rung sees one voice
    (`providers/ml/diarization/split_voices.py`).
  - **The record.** The sidecar key `split_voices` lists every pair's similarity, merged or not,
    so the threshold can be chosen from a run without re-diarizing; also what was merged into
    what, or why it abstained.
  - **Why:** 12 of 68 unnamed Gate 1 voices were a second voice of a named person (eval repo
    NAMING-RETHINK, M5).
  - **Not measured yet:** the threshold and the effect. That is the DGX block.

- **v32, 2026-10-23.** `speaker_naming_passes` (default 1, at most 5) sets how many times each
  naming LLM call is asked: the ladder's resolution call and the staged transcript-naming call.
  No rung changed.
  - **Why:** vLLM is not deterministic even at temperature 0. The same episode came back with
    different names across passes.
  - **With more than one pass**, a voice keeps a name only when every pass gave it the same one
    (case-insensitive), and its role only when every role agreed.
  - **Resolution call:** the first pass's report is the trace input, and the other passes are
    under `passes`. `passes_agreed` lists the names dropped by disagreement.
  - **Transcript naming:** a failed pass is left out; `report.passes_agreed` says how many
    answered.
  - **Bundled calls are asked once.**
  - **Measured** (eval repo NAMING-RETHINK, 3 passes, held out):
    - ladder all-agree: wrong ~36 -> 29 on tuning, about 17 more unnamed;
    - LLM all-agree: wrong 3.1%, unnamed 17.1%.
  - **Off by default:** each extra pass costs one more call of each kind.

- **v31, 2026-10-23.** Transcript naming also runs inside the bundled calls (`bundled`,
  `mega_bundled`, `extraction_bundled`). This is the operator's option A. No rung changed.
  - **The question.** The bundled transcript is the named one (`Name: text`, no voice ids), so
    the call is asked who each speaker LABEL really is (`shared/speaker_naming/bundled_field_v1`).
    This is a check of the ladder's labels, not the measured voice-id prompt.
  - **Mapping.** Labels map back to voices through the sidecar's `resolved_name`. A label two
    voices share is left out.
  - **No second call.** In a bundled mode whose provider has that call, the staged standalone
    call is skipped.
  - **When it is added.** Only with transcript naming on and diarized voices on disk; otherwise
    every bundled prompt is byte-identical.
  - **The answer stays out of the summary.** It is taken out of the summary JSON before the
    summary parser sees it.
  - **The record.** It is written into the sidecar's `transcript_naming` key with
    `call: bundled | mega_bundled | extraction_bundled`, in the standalone entry's shape, with
    `applied: false`. A bundled call never changes the published names yet. The model side gets
    the same host-or-guest rule, publish gate and talk floor; it does not get operator renames,
    which the metadata stage does not hold.
  - **Every provider:** OpenAI family, Gemini, Anthropic, Mistral, Grok; Ollama for `bundled`,
    as it has no mega or extraction bundle.

- **v30, 2026-10-23.** Two rescues after the ladder, behind one flag, `ladder_rescue_rules`
  (default off): `providers/ml/diarization/ladder_rescues.py`. They are ports of the rules
  measured in podcast-scraper-eval-data (`variants/e1d_third_person_introduced_strict.py`,
  `split_rule.py` with `strict=True`). The operator's decision was to ship them flag-gated and
  watch their fires.
  - **S30s** (`s30_split_merge`): one host split into two voices.
    - The conditions: exactly two speaking voices (>= 150 words each); one named N as host and
      the other unnamed; the LLM names nobody but N; no stated name other than N; and the unnamed
      voice holds >= 30% of the two voices' words.
    - The unnamed voice becomes N, host.
  - **E1d** (`e1d_rescue`): an unnamed voice whose LLM name the third-person guard refused takes
    that name when no voice holds it and the conversation introduces it to that voice. The voice
    says "I'm X" / "I am X" / "my name is X", or another voice welcomes or thanks X.
  - **Order:** S30s, then E1d (the measured arm's order). Both run before transcript naming,
    the voice rung, the talk floor and renames, so the floor still guards a rescued name.
  - **Trace:** per voice `named` under the rung `s30_split_merge` / `e1d_rescue`; E1d records the
    `cue`.
  - **Measured:** tuning E1d 19 right / 0 wrong, S30 +10 right / 0 wrong. Held out, E1d + split
    merges 9 fires, 8 right; the 1 wrong was plain S30, which is not shipped.

- **v29, 2026-10-23.** `voice_identity_rung` now has three modes: `off` (the default), `shadow` and
  `publish`, the same as `transcript_naming`. No rung changed.
  - **`shadow`.** The ladder publishes. The fingerprint rung's answer goes beside it in the
    speaker-diagnostics sidecar under the key `voice_identity`.
  - **Per voice:**
    - `ladder`;
    - `fingerprint_result` (the roster the rung gives after the same talk floor and renames);
    - `match`: the best voted group, its similarity and the voice's talk seconds;
    - `outcome`.
  - **Per episode:**
    - `version`, `mode`, `outcomes`;
    - `report`: `history_voices`, `groups`, the embedding `model`, or `abstained` with the reason
      (no fingerprints, no feed workspace, no episode id, no voted group).
  - **Same shape as `transcript_naming`.** The two shadow keys compare voice by voice.
  - **The feed directory.** The rung reads the feed workspace that a real run's `output_dir` is
    (`corpus_feed_output_dir`). A stage-level test proves it
    (`test_voice_identity_reads_the_feed_workspace.py`); the "suspected bug" in the review was
    not one.

- **v28, 2026-10-23.** `transcript_naming` now has three modes: `off` (the default), `shadow` and
  `publish`. No rung changed.
  - **`shadow`.** The ladder publishes. The model's answer is written beside it in the speaker
    diagnostics sidecar under the key `transcript_naming`, and nothing else the stage publishes
    changes (a unit test compares the stage's output in shadow and off).
  - **`publish`.** The v27 behaviour, with one change: a name the model gives the voice the ladder
    already gave keeps the ladder's `source`. A self-introduced host named like the show
    (`self_intro`) is therefore still placed in the record and the graph.
  - **The sidecar key, per voice:**
    - `ladder`: the name, role and source the ladder publishes;
    - `model`: the model's name, role, confidence and evidence;
    - `model_result`: the roster the answer gives after the same talk floor and renames;
    - `outcome`: `agree`, `adds`, `removes` or `changes`.
  - **The sidecar key, per episode:**
    - `version`, `mode`, `call` (`standalone`), `ok`, `voices_asked`, `outcomes`;
    - `report`: the prompt version, provider and model, the clip (window, turns, turns dropped,
      prompt tokens), the raw answer, and `error` or `skipped`.
  - **The prompt** lives in the prompt store as `shared/speaker_naming/standalone_v1` and
    `instructions_v1`; a test pins it byte for byte to the measured prompt.
  - **Clipping.** A transcript too long for the provider's window loses turns from the middle,
    marked `[... N turns omitted for length ...]`, so the opening introductions and the closing
    sign-offs stay. The window comes from `served_context_tokens()` (OpenAI-compatible providers)
    or `ollama_num_ctx`; on other providers it is unknown and the transcript is sent whole.
  - **Providers.** Every provider's `complete_text` takes `max_tokens` (default 800) and `stage`.
    The call asks for 3,000 output tokens, and its cost is recorded as `speaker_naming` where the
    provider records cost (OpenAI-compatible providers and Gemini). Ollama's `complete_text` now
    passes `num_ctx`, which it did not, so long resolver prompts are no longer cut to Ollama's
    2,048-token default.
  - **Voice identity.** A name with source `transcript_naming` does not vote in the
    `voice_identity` feed bank.
  - **Trace.** In `publish` the per-voice trace steps are as in v27; the answer itself is in the
    sidecar key, not in `decision_trace.inputs`.

- **v27, 2026-10-22.** One stage added after the ladder, OFF by default: `transcript_naming`
  (`transcript_naming`, `providers/ml/diarization/transcript_naming.py`). The LLM reads the WHOLE
  diarized transcript (with the feed and episode metadata) and names every voice with at least 40
  words; a voice it calls host or guest takes that name and role, every other voice is unnamed;
  when the call fails or the answer cannot be read, the ladder's roster stands. It can move a name
  from one voice to another, which no rung can. Measured with the production model on 181 held-out
  two-voice episodes (podcast-scraper-eval-data, `docs/wip/NAMING-RETHINK-2026-10-22.md`, "Bet 1"):
  unnamed nameable voices 9.4% -> 2.8%, wrong names 6.6% -> 3.8% of named voices. The prompt is the
  measured one (a test pins it). Order at the end: non-regression, `transcript_naming`,
  `voice_identity`, `talk_floor`, operator renames.
- **v26, 2026-10-22.** Two rungs added at the end of the ladder, both OFF by default, and one
  record rule fixed. Measured in podcast-scraper-eval-data (`docs/wip/NAMING-RETHINK-2026-10-22.md`,
  `NAMING-GATES-2026-10-22.md`):
  - `voice_identity` (`voice_identity_rung`, threshold `voice_identity_threshold` 0.7): a voice that
    matches a fingerprint group of the feed's OTHER episodes (named by vote: >= 2 episodes and more
    than half of the group's named members) takes that name; the name leaves a voice clearly not
    that person (similarity < 0.5). The rung's own names never vote. Design:
    `docs/wip/VOICE-IDENTITY-DESIGN-2026-10-12.md`, stage 3. Needs stored fingerprints (4% of the
    10-17 corpus copy has them), so it acts on new episodes as feed banks grow.
  - `talk_floor` (`speaker_name_min_talk_share`, 0.05 as measured): after every rung, a named voice
    holding less than that share of the episode's talk (aligned segments' duration) is unnamed. A
    person's name on such a voice is almost always an ad, promo or clip voice: replayed on 601
    labelled episodes, 51 such names removed plus 3 spurious and 2 wrong, against 12 right names and
    2 role errors lost (briefly speaking hosts, guests of multi-guest compilations); on held-out
    sets 9 removed, 0 right lost.
  - record (`metadata_generation._build_speaker_record`): a person placed on two voices is
    `self_intro` when ANY of its voices introduced itself (it took the first listed voice's
    source, and the show-name check then dropped a split, self-introduced eponymous host).
  Order at the end of the ladder: non-regression, `voice_identity`, `talk_floor`, operator renames.
- **v25, 2026-10-15.** One host seat step added (#2366): `host_seat_step` `7_question_lead`. When
  every step above seated nobody and the pool holds exactly one publishable name, the voice that
  asks questions at least twice as often as every other voice with 10% or more of the non-ad talk
  (`_question_lead_voice`, "?" per 1,000 words, a transcript with at least 20 full stops) takes the
  seat, under step 6's guards; the forced naming then gives it the pool's one name. Measured in
  `docs/wip/HOST-SEAT-BY-QUESTIONS-2366.md`: the voice picked is the host in 47/48 (dev_v1),
  177/186 (val_v1), 56/64 (val_v2). Snapshot replay: 6 voices change, all better.
- **v24, 2026-10-15.** The promo rule (`_self_introduced_promo_voices`, in `inputs.ad_voices`),
  two fixes (#2364): it reads each voice with one space between words (two segments joined at
  "on  Patreon" were no cue), and its exemption needs the voice to introduce ITSELF as a stated
  person (`_introduces_itself_as_stated`, any spelling the stated-name snap accepts). Saying a
  stated name in the third person no longer exempts a voice: StarTalk's supporter shout-outs
  ("I support StarTalk on Patreon. This is StarTalk with Neil deGrasse Tyson") were guests.
- **v23, 2026-10-15.** One reference widened (#2363): the final ADR-130 snap
  (`recover_stated_names`) also takes the people in the episode TITLE as spelling references
  (`_title_person_spellings`: every two- or three-word capitalised run with a person's shape; a
  bracket, a function word or an all-caps token ends a run). They come LAST, after the feed's
  hosts and the metadata's people, and they never name a voice: they only respell a name a voice
  already carries ("Matt Iglesias" -> "Matthew Yglesias", Capitalisn't). `resolve_speaker_roster`
  takes `episode_title`; `roster_replay.py` passes it only to a variant that accepts it.
- **v22, 2026-10-15 (catch-up: changes on main after v21 that this doc had not recorded).**
  - Phase C (`66eaedd0c`, #2347): feed-stated hosts stay in the pool even when heard in an ad
    (the ad-reader filter applies only to names guessed from the transcript);
    `plausible_person_names` drops group and institution phrases at the roster's entry; step
    `3_opener` seats an opener beside an already seated host only when its first 3 turns name the
    show or a stated person in full (`_opener_presents`); a lone unnamed host seat with two
    unclaimed stated hosts is the one it does not keep addressing (`_seat_by_cohost_address`,
    trace note `host_naming_by_cohost_address`).
  - K8 (`ee4d19961`, #2338): a last forced-guest rung, `forced_by: stated_guest_owns_the_talk`.
    Exactly one metadata-stated full name is unplaced, its surname is spoken, and exactly one
    unnamed non-host voice holds at least half the talk; never when the episode calls that person
    another show's host (`_named_as_another_shows_host`).
  - F7 (`2fd6411dc`): step 4 takes a substantial voice present across the episode as a candidate
    when no opening-window candidate exists and no guest is present (`_storyteller_candidates`);
    two unnamed seats that the strict pair refuses are named by each seat's dominant address,
    both or neither (`_pair_by_dominant_address`, inside `host_naming_pair`).
  - `19da42c29`: a voice under 90 s that introduces itself AND speaks a promo's lines is an ad
    voice (`_self_introduced_promo_voices`, in `inputs.ad_voices`), unless it says a person this
    episode states; "I'm <role> <stated name>" names a stand-in; the opener's hail at a turn start
    counts.
  - `c2ac9fcfa`: the shared cleaning types a voice `commercial` by ad overlap only below 25% of
    the episode's talk (`COMMERCIAL_MAX_TALK_SHARE`).
  - #2324 (non-English onboarding) made the per-language cue rows reach every reader (D1-D28 in
    `docs/wip/NONENGLISH-REFERENCE-SHOWS-2026-10-09.md`); English behaviour unchanged.
- **v21, 2026-10-12.** Two guards (#2304 Track B audit, K13): `title_subject` does not declare the
  title's OPENING guest while a host seat holds over half the talk (as the tail already did), and the
  forced host name skips a seat under 5% of the talk beside a voice of 20% or more
  (`_is_a_sliver_seat`).
- **v20, 2026-10-12.** One rule widened (#2304, F2d): `title_subject` accepts a title's person
  (opening or tail, now also "ft. / feat. / featuring" tails) when a host voice seated on its own
  evidence says the name IN FULL and also ADDRESSES them by first name ("...with me, Tim."), with no
  all-caps token ("Surge AI", "NVIDIA AI"). Before, only the metadata, the description ("speaks
  with") or a first-name greeting vouched.
- **v19, 2026-10-12.** One rule changed (#2304): `self_intro` replaces a voice's unpublishable
  self-introduction ("I'm Apple just announced...") with the same voice's next publishable one
  ("I'm Ryan Knutson") (`_publishable_intros`). Never dropped when there is none: a lone first
  name ("Hi, it's Kenny") is still evidence for the host snap and the guest role.
- **v18, 2026-10-12.** One rule changed (#2304): the forced guest name's third-person refusal
  (`says_that_name_in_the_third_person`) no longer fires when every mention of the surname follows
  a clearly DIFFERENT given name (`_surname_only_of_someone_else`): The Living Philosophy's Layman
  Pascal talking about "Blaise Pascal". A near spelling (two edits, "Laman") or a title (Dr., Mr.,
  Professor) still counts as the same person, and so does a bare surname.
- **v17, 2026-10-12.** Two rules changed (#2304, F2c). `title_subject` also accepts the person
  the title ENDS with ("[Dr. Jeff Beck]", "(Dan Hendrycks)", "| Josh Elman", "— Brad Carson",
  "with Steve McConnell", read in the first 160 characters of title + description), when the
  metadata states that person or a host voice greets them by first name ("Steve, welcome to the
  podcast"); only beside a host seated on its own evidence (self-introduced, presenting,
  performing the host role, or introducing others; not the opener seat, not the model's verdict
  alone) that holds at most half the talk, and never a short form of a name a voice already
  carries ("Chris" for "Christine"). `host_introduction_harvest` snaps a heard name to the
  title's spelling ("Matt McGinnis" -> "Matt MacInnis").
- **v16, 2026-10-11.** `title_subject` also accepts the title's person when the episode's own
  description introduces them ("X speaks with / is joined by / interviews <the same person>"), for
  episodes whose metadata extraction stated nobody.
- **v15, 2026-10-11.** One rule changed (#2304, F5): `self_intro` snaps a self-introduction said
  INSIDE the host's own formula ("Welcome to X... I'm N", "My name is N, and I'm joined by",
  "And I'm N") onto the one unclaimed stated host whose surname is within one edit (two for
  surnames over five letters), on a voice with at least 5% of the talk
  (`_snap_hosts_by_their_formula`): misheard given names ("Alexander Carpi", "Jill Weisenthal").
- **v14, 2026-10-11.** Two rules changed (#2304): `guest_pool` adds the person the episode title
  opens with ("Neil Kinnock: ...", "Brett Wilson on ...") when the metadata states that same person
  and a host is seated on evidence (`title_subject` name step, F2); `intro_reader` skips (reason
  `not_a_person`) a heard name that is the show or ends in a word no surname is ("New York City",
  "Consumer AI Ads", F12).
- **v13, 2026-10-11.** One rule changed (#2304, F12): `intro_reader` skips (reason `sponsor_read`)
  a voice under 5% of the words whose own text is a sponsor read ("Today's episode is brought to you
  by", "sponsored by", "promo code"): the next speaker after "my guest today is X" was the ad.
- **v12, 2026-10-11.** Two rules changed (#2304, F2): the English introduction cue reads "I'm
  delighted / very happy to be chatting with[, in person,] X" (Conversations with Tyler's opening),
  and a forced guest name is not refused "in the third person" when the voice owns most of the words
  and says the name only in direct address that opens its sentence ("Steven Pinker, thank you very
  much.": the host's close bled into the guest's cluster, `_only_addressed_in_a_dominant_voice`).
- **v11, 2026-10-11.** One rule changed (#2304, F4a): before the model's names are merged, the name
  for a voice that itself INTRODUCES that person ("Today, my guest is Jacob Warwick", "I'm here with
  Eric Alabest", read by `guests_introduced_by_the_host`, matched by the stated-name snap) is dropped:
  `llm_merge` `skipped` with reason `introduced_by_this_voice`. The model's host/guest verdict stands.
- **v10, 2026-10-11.** One rule changed (#2304, F5a): `guest_pool` drops a declared guest name that
  the bounded host snap (`_canonicalize_to_known_host`) maps onto a stated host: it is that host in
  the ASR's spelling ("I'm joined by RJ Hanaki" for Latent Space's co-host RJ Honicky), not a spare
  guest name for one-name-one-voice to force (`_declared_guests_not_hosts`).
- **v9, 2026-10-11.** One rule changed (#2304, F4b2): `inputs.conversation_roles` no longer marks
  "host" a voice that owns at least half the talk (`_DOMINANT_SHARE`) when every host act in it is
  half of a bled pair: within 200 characters (`hosts.BLEED_PAIR_CHARS`) of a guest reply that opens
  its own sentence ("Welcome to the show. Thank you for having me."). Gated on the share because the
  text cannot say which side owns the pair: applied to every bled voice it un-named Hard Fork's two
  hosts on the corpus replay.
  Only a voice the model NAMED (a role fix, never an open guest seat): on an unnamed voice the
  one-name-one-voice arithmetic put the wrong stated guest on it (gold val_v1 v226).
- **v8, 2026-10-11.** One rule changed (#2304, F4c): `host_seat_step` `5_llm_host_no_pool` no longer
  seats a voice that replies to being invited ("Thank you for inviting me", "thank you for the
  invitation", `roster._INVITED_REPLY`). Read ONLY there: as a general guest act the same phrase,
  bled into the host's cluster at a turn boundary, un-named two hosts on the corpus replay.
- **v7, 2026-10-11.** Two rules changed (#2304, failure inventory F4b1):
  - conversation roles: "welcome to" said as a permission or idiom ("you're welcome to share",
    "listeners are welcome to", "welcome to the club") is no longer a host speech act
    (`hosts.host_speech_act_starts`);
  - `llm_merge`: an introduced name that is nobody the episode states, on a voice the model matched
    to a stated person nobody else carries, yields to the model's name
    (`over_unstated_introduction`). Found by the corpus replay of the first change: a mid-interview
    mention ("I'm talking to Justin Welber, the former archbishop") named the stated guest's voice
    once that voice stopped performing the host role.
- **v6, 2026-10-11.** No rule changed (same content hash). Every rung in the table below is now
  reached by a scenario, so `NOT_YET_EXERCISED` is empty. Two reasons I gave in v5 were wrong:
  `host_naming_pair` did not need the feed's co-presence history (the address pair alone says the
  co-host is present; the scenario's second host first spoke outside the intro window), and the ad
  rung was not a harness gap (the harness runs the cleaning step; the scenario's sponsor read was too
  weak for the pre-roll detector and its episode too short for the edge-ad share). The harness can now
  feed the feed's earlier episodes (`feed_history`), and the F4/F5/F6 failure inventory from prod is in
  the library as strict-xfail scenarios.
- **v5, 2026-10-11.** Three rules changed (#2304 naming plan):
  - `guest_naming`: a forced name refused "in the third person" is kept when another voice hails that
    first name and this voice answers next (`_refuted_unless_hailed`);
  - `host_naming_forced_gates`: `seat_owns_the_talk` steps aside for a narrator below 70% of the talk
    among at least two more substantial voices than the feed has hosts;
  - the canonical name drops a trailing capitalised credential written without a comma ("Paul Hoffman
    SC").
  The ladder is now also tested as one component: `tests/integration/naming_ladder/` (see "Scenario
  tests" below).
- **v4, 2026-10-08.** In line with the code again: the `host_naming_pair` rung and the
  `address_pair` seat naming were missing; the `introduced` source (`93f49482c`) is no longer
  pending; the trace is deployed. The one-person-one-entry rules (`c05cc0273` and the follow-up)
  are described at the rungs where they act ("Same-person rules" below), and the record-level
  check and m0025, which happen outside the trace, are listed under NOT recorded.
- **v3, 2026-10-04.** v2 called phase 1 "Done"; it was not — the helpers' decisions (host-seat
  steps, forced names, guest variants, one-name groups, voice-type reasons) were listed as phase 1
  in v1 and not recorded. Now recorded, plus both LLM answers in full (moved forward from phase 3),
  a test per rung, an end-to-end run with the real LLM, and an old-vs-new behaviour gate.
- **v2, 2026-10-04.** Phase 1 implemented and reviewed. The design text is replaced by what the
  code records, the real sidecar shape, what is NOT recorded, and the purity-gate result.
- **v1, 2026-10-04.** Design, written from the 31-rung inventory.

## Goal

The operator set this goal on 2026-10-04. Every episode's sidecars must hold enough to reconstruct
how each voice got its name and role: which rung proposed what, what won, what was overridden, and
why. With that we can:

- measure the priority order instead of arguing about it;
- find patterns across the corpus;
- design new solutions from data.

## Why: what the inventory found

The ladder has **31 rungs**, from feed host detection to the published roster (inventory of
2026-10-04). Before phase 1, the diagnostics sidecar
(`<transcript>.speakers.diagnostics.json`) recorded only three things:

- the **inputs** (`tried`);
- the **final** per-voice result (`voices[]`);
- `resolution_attribution`.

The `source` field is **lossy**. `self_intro` is the catch-all for every intro entry that is not
from the publisher or the LLM. Downstream code (the metadata show-name filter, m0014, the
copresence prior) reads `self_intro` as "the voice said it", which is not always true.

Worked example: Conversations with Tyler, "Julia Ioffe on…", gold dev c087.

1. The rules named the 2,735 s voice "Julia Ioffe" from her own self-introduction. This is correct.
2. The LLM then added "Julia Ioffe" as HOST on Tyler's 380 s voice. The show states no host.
3. Corroboration had already dropped her from the guest list, because "X on …" in a title is not an
   accepted cue.

The old sidecar showed none of this chain.

## Phase 1: what is recorded (code as of this doc version)

### Mechanism

`resolve_speaker_roster(..., trace: Optional[NamingTrace] = None)`. The helpers that make the
decisions take the same optional `trace`; `resolve_speaker_roster` passes its own.

- **Observer only.** Recorders are plain calls on a `NamingTrace`; with none given, a `NullTrace`
  (`enabled = False`) records nothing, so the ladder carries no `if trace` branches.
  `_select_host_voices` sits at the complexity cap (mccabe 25), so its recorders are straight-line
  calls only.
- **One decision-code refactor**, in `_name_host_voices`: the forced-name condition was one `and`
  chain; it is now named booleans (`one_name_one_seat`, `seat_owns_talk`, `presenter_elsewhere`,
  `addressed`) evaluated in the same order with the same short-circuits, so the trace can say which
  gate held. Gated by the old-vs-new replay below.
- **Failure isolation.** A recorder's exception is swallowed and sets `degraded: true`;
  `to_dict()` never raises and returns a JSON-safe copy (sets → sorted lists); an unserialisable
  trace (NaN) becomes `{version, degraded: true, error}` so the sidecar is still written. NOT
  guarded: arguments the roster builds at a call site (`sorted(...)`, `dict(...)`).
- **The talkative-host retry runs untraced** (`_name_host_voices` a second time, one seat's answer
  taken); its `talkative_host` diff records the result.
- **Pipeline.** Traces the SHIPPED pass. Adds `inputs.baseline_without_llm` (rules-only roster per
  voice), `non_regression` / `restored` steps, `inputs.llm_resolution` and `inputs.detection`.
  Written as `speaker_diagnostics.decision_trace`.
- **Offline replay.** `scripts/measure/roster_replay.py --trace-out PATH` replays every stored
  episode plain and traced; summary counts `traced_roster_differs`, `traced_only_error`,
  `trace_degraded`, `trace_bytes_max`; exits 1 if any of the first three is non-zero; refuses a
  roster variant without a `trace` parameter.

### The two LLM answers, in full

- **`inputs.detection`** — ingest-time speaker detection, recorded on the `Episode`
  (`speaker_detection_report`) by `_detect_speakers_for_episode` and carried to every
  `apply_diarization_to_result` call site: `detector`, `raw` (the model's answer; each LLM
  provider's parser keeps its last one, cleared before every call; capped at 8k chars, `raw_chars`
  keeps the length), `returned` (speakers / hosts / succeeded), `dropped_placeholders`,
  `dropped_non_persons`, `hosts_from_description`, `hosts`, `proposed_guests`,
  `corroborated_guests`, `corroboration_rejected` (`{name, reason}`, reason `is_a_host` /
  `no_interview_cue`), and `outcome` / `reason` (a skipped detection says so). `null` when the call
  site had no record (e.g. a reprocess that skipped detection).
- **`inputs.llm_resolution`** — the post-diarization resolver: `provider`, `stated_names`,
  `prompt_chars`, `raw` (capped at 16k, `raw_chars` uncut), `verdicts` (per voice the model named:
  `said_voice`, mapped `voice`, `name`, `role`, `matched` stated name, `outcome` = `accepted` /
  `invented` / `third_person` / `duplicate` / `unmapped_voice` / `role_only` / `abstained`),
  `complement` (`two_voice` / `swap`), `error`, or `skipped` with the reason it was never asked.

### Step semantics

- `name` is the name a step SET (or refused at the publish gate / intro reader); a proposal not
  taken is `proposed`.
- **Set a name:** `named` / `renamed` (name diffs), `set` / `changed` (roster diffs; `changed`
  carries `previous`, `cleared` lists fields that became null), `accepted` (LLM merge),
  `named_from_earlier_rung` (host or guest naming kept a name an earlier rung put on the voice —
  that rung's own step says which), `forced_pool_name`, `forced_name`, `prefix_stripped`, `restored`.
- **Other:** `seated`, `stated`, `heard`, `typed`, `unnamed`, `forced_name_vetoes`,
  `bare_first_name_joins`; did-not-take: `skipped`, `refused`, `refused_spelling`, `excluded`,
  `never_names`, `removed`, `dropped`.

### Rungs recorded, in ladder order

| Rung (trace key) | Recorded as | What it captures |
| --- | --- | --- |
| `inputs.detection`, `inputs.llm_resolution` | input | both LLM answers in full (above) |
| `inputs.*` | input | `known_hosts`, `detected_guests`, `metadata_named`, `stated_voice_names`, `llm_voice_names`, `llm_voice_roles`, `ad_voices`, `conversation_roles`, `host_pool` |
| `guest_host_pool` | episode note | a guest host the episode text added to the pool |
| `publisher_label` | per voice `stated`, then name diff | every person label the publisher states; where it replaced a name |
| `self_intro`, `show_mononym_filter` | name diff | names voices gave themselves; show-mononyms removed |
| `intro_reader` | per voice `heard` (`heard`, `spelling`) / `skipped` (`ad_voice` / `named_itself`) / `refused_spelling` (`name`, `resembles`), plus name diff | every introduction read, what it became, what was refused and whom it resembled |
| `llm_merge` | per voice `accepted` (`llm_said`) / `skipped` (`proposed`, `reason`, `kept`) / `over_unstated_introduction` (`introduced`, `llm_said`) | every surviving LLM name and what happened to it; the third: an introduced name that is nobody the episode states yields to the model's stated name for that voice |
| `cohost_formula` | name diff | names from the co-host formula |
| `presenter_evidence` | episode note | branded / introducer / copresenter voices; conversation guests |
| `host_pool` | input, plus per name `never_names` | the pool with each name's source |
| `host_seat_guards` | episode note | every veto set the seat steps consult, and the cap |
| `host_seat_step` | per voice `seated` (`step`) | WHICH step seated it: `1_named_as_a_stated_host`, `1b_presents_on_own_evidence`, `1c_unnamed_introducer`, `2_performs_host_role`, `3_opener`, `4_count_fill`, `5_llm_host_no_pool`, `6_stand_in_interviewer`, `7_question_lead`. `3_opener` beside an already seated host needs the opener to present (`_opener_presents`); `4_count_fill` may draw a storyteller candidate (`_storyteller_candidates`) |
| `host_seat_opener`, `host_seat_step_4` | episode notes | the opener; step 4's arithmetic (empty seats, unclaimed / said-absent hosts, fillable, candidates, guest present, third-party excess, co-host said present) |
| `host_seats` / `host_seat` | episode note, per voice `order` | final seats in order |
| `host_naming_forced_gates` | episode note | spare names before/after the feed-history and said-absent filters, unnamed seats, guest-hosted episode, one-name-one-seat, seat owns the talk (not when the seat narrates: below 70% of the talk with at least two more substantial voices than the feed has hosts — The Flip, `b3fb1c941`), better presenter elsewhere |
| `host_naming_pair` | episode note (`performs_guest_act`, `named`) | two unnamed seats and two unclaimed pool names, each seat addressing the other's host by name (`_pair_by_address`, The Rest Is History); when that strict pair refuses, each seat's dominant address, named only when both resolve to different hosts (`_pair_by_dominant_address`, F7) |
| `host_naming_by_cohost_address` | episode note (`seat`, `named`, `performs_guest_act`) | one unnamed seat, two unclaimed stated hosts: the seat is the host it does not keep addressing (`_seat_by_cohost_address`, Phase C, Hard Fork) |
| `host_naming` | per seat `named_from_earlier_rung` / `forced_pool_name` (`by: address_pair` when the pair rule named it) / `unnamed` (`intro_name_taken`), plus `forced_name_vetoes` (guest act, greeted by that name, rescued from bleed, forced) | how each seat was named or why not |
| `host_introduction_harvest`, `two_voice_interview` | per name `added` / `refused` | guest names harvested from the host's introduction |
| `guest_pool` | per name `excluded` (reason), episode note | the guest pool and why declared names left it |
| `guest_naming_forced` | episode note | spare names, candidates above the cameo floor, forced name, `forced_by` = `one_name_one_voice` / `only_voice_left_below_cameo_floor` / `dominant_unassigned_voice` / `host_elimination` / `stated_guest_owns_the_talk` (K8), forced voice |
| `guest_naming` | per voice `named_from_earlier_rung` / `forced_name` (`forced_by`) / `unnamed` (`role`, `intro_name_refused`, `forced_name_refused`, `role_evidence`) | how each remaining voice was named or why not. `forced_name_refused: says_that_name_in_the_third_person` is NOT given when another voice hails the name's first name and this voice answers the next turn, unless this voice hails it itself or says the full name (`_refuted_unless_hailed`, `64654a8c9`) |
| `talkative_host`, `ad_voice_placeholder`, `stated_spelling_snap`, `recover_stated_names` | role diff | changes only |
| `publish_gate` | per voice `prefix_stripped` / `refused`, per name `refused` | the final plausibility gate. Every published name is also canonical (`canonical_person_name`): a trailing credential after a comma, or written in capitals without one (`SC`, `QC`, `MD`, `PhD`; not `Ma`, `Do`, nor `KC`, a Nepali surname), is dropped (`aee17d79e`) |
| `one_name_per_person` | role diff, per voice `bare_first_name_joins`, episode note per group (`unified`: names, roles, kept name + `kept_because`, role + `role_reason`; or `kept_apart`) | one person, one name, one role |
| `nameable` | episode note | nameable voices, leftover / unbound names |
| `voice_types` | per voice `typed` (`voice_type`, `reason` = `edge_ad` (+`name_demoted`) / `named` / `mostly_inside_ads` / `brief` / `no_source_names_them` / `a_name_existed_and_we_failed`, `talk_s`, `classified_by`) | every voice's type and why. `commercial` by ad overlap needs 60% of the voice's talk in ads AND under 25% of the episode's talk (`c2ac9fcfa`) |
| `non_regression`, `inputs.baseline_without_llm` | pipeline | names the additive contract restored; the rules-only roster |
| `transcript_naming` | per voice `named` / `kept` / `removed` (`name`, `said_role`), in `publish` only; the answer and the comparison with the ladder are in the sidecar key `transcript_naming` (v28), in `shadow` and `publish` | the whole-transcript naming pass and what it did to each voice (off by default) |
| `s30_split_merge` | per voice `named` (`name`) | a split host's unnamed half named as the host (`ladder_rescue_rules`, off by default) |
| `e1d_rescue` | per voice `named` (`name`, `cue` = `self_intro` / `introduced`) | an introduced name the third-person guard refused, put back on its voice (`ladder_rescue_rules`, off by default) |
| `voice_identity` | per voice `named` / `removed` (`name`), in `publish` only; the rung's answer against the ladder is in the sidecar key `voice_identity` (v29), in `shadow` and `publish` | the feed fingerprint match: a voice given a group's name, or a name taken off a voice that is clearly not that person (off by default) |
| `talk_floor` | per voice `removed` (`name`, `talk_share`) | a name taken off a voice under `speaker_name_min_talk_share` of the episode's talk (off by default) |

**Sources a voice's name can carry** (`voices[].source`). `_NAME_EVIDENCE_RANK` ranks four of them,
used when one person's voices disagree: `self_intro`, `introduced`, `publisher_transcript` = 3,
`llm_resolution` = 2.

- `publisher_transcript` — the publisher's own speaker label.
- `llm_resolution` — the post-diarization LLM's closed-list match.
- `introduced` — the HOST's words, read by the introduction reader (`93f49482c`). Counts as
  host-copresence evidence, not as the voice's own words.
- `self_intro` — every other name in the intro map: the voice's own introduction (opening,
  sign-off, or a weak one the metadata vouches for or anchors) and the co-host formula's names.
- A host seat named from the pool keeps the pool entry's source (`feed`, `known_hosts`, …).
- `e1d_rescue` — E1d: an introduced name the third-person guard refused (v30).
- `s30_split_merge` — S30s: the second voice of a host split in two (v30).
- `transcript_naming` — the whole-transcript LLM pass, in `publish` only (v27). Does not vote in
  the voice identity bank.
- `voice_identity` — the feed fingerprint match, in `publish` only (v26), and every name
  migration 0035 applies, respellings included. Does not vote in the voice identity bank.
- `talk_floor` — unnamed by the talk floor (v26; `named` is false).
- `forced` — placed by arithmetic. `raw` — unnamed.

### Same-person rules, and the rungs they act at

One person is one entry (2026-10-08, `c05cc0273` and its follow-up). Two predicates, by scope:

- **Across sources** — `_same_person`: same surname, and a matching given name or initial (or a
  title + surname). A missing generation ("Jr.") is not a disagreement.
- **Within one episode** — `_same_person_on_one_episode`: titles stripped, whole-name similarity
  >= 0.91, or one differing token with similarity >= 0.70 (`_TOKEN_RESPELLING_SIMILARITY`); a
  generation difference IS a different person.

Where each acts:

| Rung | Rule |
| --- | --- |
| `self_intro`, `host_seat_guards` (`stated_non_host`) | A leading title is not the given name (`_core_name_tokens` drops it while a given name and a surname remain). `_snap_near_identical_host` maps a self-introduction to the stated host when they differ only by a title, by one letter (5+ letter surname), or by two adjacent swapped letters ("Wiesenthal" / "Weisenthal"). A voice that says "I'm Professor Hannah Fry" on Hannah Fry's show is therefore not a stated non-host. |
| `recover_stated_names` | A guest never takes a stated HOST's spelling (N1, "Kevin Ross" / "Kevin Roose") — unless the episode also names that person as a participant (`detected_guests` / `metadata_named`): then the host pool is the polluted source ("Lucas Kaiser" -> Lukasz Kaiser, guest, The a16z Show). When the fuzzy canonicaliser finds nothing, `_stated_participant_spelling` gives the voice the ONE stated participant it is a respelling of by the one-episode rule, if no other voice holds that name ("Scholastic Gatobu" -> "Schola Gatobu", "Charming Lai" -> "Chiamin Lai"). A name the episode already states is never re-spelt. The references, in priority order: feed hosts, detected guests, metadata people, then the people in the episode title (`_title_person_spellings`, v23), which only respell and never name. |
| `guest_pool` (`same_person_as_a_named_voice`) | A declared guest name is not spare when it is the same person as a voice's name by `_same_spoken_person` OR the one-episode rule ("Andy Ratcliffe" beside "Andy Rachleff", The Long Run). |
| `one_name_per_person` | Voices grouped by the one-episode rule take one name and one role. |
| record (`metadata_generation._build_speaker_record`) | A person placed on two voices takes `self_intro` as its source when any of its voices introduced itself, so the eponymous-host exception to the show-name check holds for a split host. Not a roster rung. |
| record (`metadata_generation._unplaced_speakers`) | An unplaced entry is dropped when it is the same person as a placed or earlier entry by EITHER predicate. Not a roster rung: not in the trace. |

## Scenario tests: the ladder as one component

`tests/integration/naming_ladder/` runs the whole naming stage, from feed and episode metadata,
the detection output, optional injected LLM verdicts and anonymous diarized turns to the speaker
record. Each `scenarios/*.json` states the CORRECT answer (name, role, or names it must never be;
optionally the source and the trace steps that must decide). `test_ladder_doc_alignment.py` fails
when a rung in the table above is reached by no scenario and is not in its `NOT_YET_EXERCISED`
list, so this document and the library cannot drift apart. Known-wrong outcomes are strict
`xfail` with the reason.

| Rung | Scenarios that reach it |
| --- | --- |
| `ad_voice_placeholder` | `ad_narrator_saying_its_name_is_not_a_person` |
| `cohost_formula` | `cohost_formula_from_the_other_side`, `cohost_formula_with_the_first_name_only` |
| `guest_host_pool` | `guest_host_named_in_the_episode_text` |
| `guest_naming` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+43) |
| `guest_naming_forced` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `guest_pool` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_introduction_harvest` | `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_stated_name_stays_on_one_voice`, `introduced_spelling_nobody_stated_is_refused` |
| `host_naming` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_naming_forced_gates` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_naming_pair` | `two_hosts_address_each_other` |
| `host_pool` | `show_name_in_the_host_pool_names_nobody` |
| `host_seat` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_seat_guards` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_seat_opener` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_seat_step` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `host_seats` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `inputs.baseline_without_llm` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_stated_name_stays_on_one_voice`, `an_introduced_stated_guest_keeps_their_voice`, `guest_saying_welcome_to_my_country_is_not_a_host (xfail)`, `guest_saying_you_are_welcome_to_share_is_not_a_host (xfail)` (+9) |
| `inputs.detection` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `inputs.llm_resolution` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `intro_reader` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `an_introduced_stated_guest_keeps_their_voice` (+9) |
| `llm_merge` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_stated_name_stays_on_one_voice`, `an_introduced_stated_guest_keeps_their_voice`, `guest_saying_welcome_to_my_country_is_not_a_host (xfail)`, `guest_saying_you_are_welcome_to_share_is_not_a_host (xfail)` (+9) |
| `nameable` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `non_regression` | `wrong_llm_host_name_on_the_guest_is_not_published` |
| `one_name_per_person` | `credential_written_without_comma`, `stated_spelling_wins_over_self_intro_spelling` |
| `presenter_evidence` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |
| `publish_gate` | `opener_laden_name_is_refused_at_publish` |
| `publisher_label` | `publisher_label_names_the_voice` |
| `recover_stated_names` | `feed_states_its_team_as_a_dash_list`, `publisher_label_names_the_voice` |
| `self_intro` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person`, `ad_read_is_not_a_guest` (+28) |
| `show_mononym_filter` | `show_mononym_is_not_a_person` |
| `stated_spelling_snap` | `stated_spelling_wins_over_self_intro_spelling` |
| `talkative_host` | `talkative_host_named_after_guest_placed` |
| `two_voice_interview` | `host_introduces_guest_on_a_two_voice_show` |
| `voice_types` | `a_cohost_introduced_in_the_asr_spelling_is_the_cohost`, `a_mid_interview_mention_is_not_the_guests_name`, `a_place_after_an_introduction_cue_names_nobody`, `a_sponsor_read_after_the_introduction_is_not_the_guest`, `a_stated_name_stays_on_one_voice`, `ad_narrator_saying_its_name_is_not_a_person` (+51) |

## What is NOT recorded yet

Do not read an absence in a trace as "did not happen".

- **"Considered and declined" for diff-recorded rungs** (`talkative_host`, `stated_spelling_snap`,
  `recover_stated_names`, `ad_voice_placeholder`): no step when nothing changed.
- **Host-seat vetoes per voice.** The veto SETS are recorded once (`host_seat_guards`); which veto
  stopped a given voice at a given step is derivable from them, not recorded per voice. Step 1b's
  presence test (share / span / co-presenter partner) is not recorded.
- **Inside `_voice_named_by_the_introduction`** (complexity 22): its per-cue rules (narrator cue,
  first-name-only, report-verb corroborated resolution) — only what it returned (`heard`).
- **`_guest_voice_by_host_elimination`**: only whether it picked (`forced_by=host_elimination`),
  not why it declined.
- **LLM name reuse** is not flagged; it is visible as an `llm_merge` `accepted` step on a second
  voice plus the first voice's earlier-rung step.
- **Host-pool sub-source** beyond the single `source` tag (feed statement / author tag /
  description / config / recurrence).
- **Detection on non-LLM detectors** (spaCy) has no `raw`; `returned` is its answer.
- **`post_hoc`.** Migrations m0012–m0020 never write the diagnostics sidecar. m0022 and m0025 make
  `voices[]` follow the repaired segments (name, `named`, `source: raw` when unnamed; m0025 also
  `role`), but no migration writes `decision_trace`: a migrated episode's trace describes the
  roster as ingested, not as published. m0025's own record is its receipts
  (`one_person_one_entry.jsonl`: each episode's renames, voice-by-voice swaps, unnamings, dropped
  entries). `scripts/measure/m0025_parity.py` checks, on a copy, that m0025 publishes per voice
  what the fixed roster publishes wherever the stored record is today's code's product.
- **The record's de-duplication** (`_unplaced_speakers`, above) runs in metadata generation, after
  the roster; the trace does not show which unplaced name it dropped as a twin.
- **`source` is less lossy than before**: the introduction reader's names are `introduced`
  (`93f49482c`), but `self_intro` still mixes a voice's own words with metadata-vouched weak
  intros and the co-host formula (see "Sources" above).
- **Offline replays stay lossy for the LLM rung**: `roster_replay` rebuilds LLM inputs only from
  voices whose final `source` is `llm_resolution`. Full LLM answers exist only in traces written
  at ingest (after the deploy).

## Gates and evidence

All runs 2026-10-04. The prod runs are read-only, inside `compose-api-1`, over `/app/output`, with
staged files whose md5s matched the committed ones.

**1. Behaviour gate — the helpers' instrumentation and the `_name_host_voices` refactor change no
decision.** Old roster (`6117aedb5`) vs new (`c8bda243d`, pre-rebase `095a2954d`), every field of every voice (name, role,
named, source, voice_type), same inputs (`--repool --signatures` equivalents):

```text
run_equal.py /app/output   ->  {"episodes": 2319}   EXIT=0   (differ 0; old_error 0; new_error 0)
```

`11e75c7ab` after it renames one step label (string only), covered by the unit tests.

**2. Purity gate — a trace never changes the roster.** New code, traced vs untraced:

```text
run_trace_replay.py --corpus /app/output --repool --signatures --new roster=... --trace-out ...
  -> {"summary": {"episodes": 2319, "trace_bytes_max": 12181}}   EXIT=0   tracebacks 0
```

`traced_roster_differs`, `traced_only_error`, `trace_degraded`, `replay_error` all 0 (absent).
Also, locally, every `resolve_speaker_roster` call in the unit suite (184) run traced and untraced
through a pytest plugin: 0 differ, 0 degraded.

**3. Completeness.** In those 2,319 traces: 4,171 voices published by name, every one with a step
that SET exactly that name (0 without). Trace JSON: p50 2,876 B, p99 7,245 B, max 12,181 B.

**4. End to end with the real LLM.** Laptop, worktree code at `c8bda243d` (pre-rebase `095a2954d`), profile
`prod_dgx_full` (DGX Whisper, pyannote, vLLM Qwen3-30B), No Priors, newest episode:
`EXIT=0`; the written `.speakers.diagnostics.json` holds `decision_trace` v1, `degraded: false`,
with `inputs.detection` (`detector: VLLMProvider`, the model's raw JSON, corroboration result) and
`inputs.llm_resolution` (raw answer; two verdicts, one written `SPEAKER_1` and mapped to
`SPEAKER_01`), seat steps, forced-name gates and voice types; 4.6 KB.
A 5-episode run of one feed (`prod_dgx_full`) was still running at the cut-off push; its result is added when it lands.

**5. Tests.** `test_naming_trace.py` (15), `test_naming_trace_rungs.py` (32: one per rung, mostly
reusing an existing roster test's scenario with a trace injected), `test_llm_answers_are_recorded.py`
(13). Mutation-checked: deleting any of 12 sampled recorders fails a test. Full unit suite with
`PYTHONPATH=src`: 13,139 passed, 55 skipped.

**What the gates do NOT prove.**

- That the decisions are RIGHT: gates 1–2 compare code with code.
- Prod sidecars: nothing is deployed; the first ingest after the deploy is the first prod check.
- Order-dependent unit flakes seen three times in this work (`test_run_manifest`,
  `test_summary_poison_guard`, `test_metadata_generation::test_generate_metadata_dry_run`): each
  failed once inside the parallel full suite and passed alone and on re-run. Not investigated to
  cause.
- `test_cli_profile_subprocess` and two other subprocess tests fail in this worktree unless
  `PYTHONPATH=src` is set (the borrowed venv's editable install points at a removed worktree);
  with it they pass.

## Phases

1. **Trace (roster, helpers, both LLM answers) + `roster_replay --trace-out`.** Code on main;
   deployed: no. Remaining gaps are listed above.
2. **Analysis on existing data.** Per-rung precision against the gold labels; failure chains per
   show across ~2,300 prod episodes. Runs on `roster_replay --trace-out` now; the LLM rung needs
   ingest-time traces.
3. **Ingest-side records** — detection and the resolver are now done (pulled into phase 1); left:
   host-pool sub-source, `post_hoc` migration records.
4. **Then priority and ordering changes**, each gated on dev + validation with the trace explaining
   every voice that moved. Candidates already visible: title/description guest phrases ("X on …",
   "— X", "with X") as corroboration; an LLM name may not take a second voice once the episode bound
   it by self-introduction; an LLM-inferred host name should not count as step 1's "named as a
   stated host" the way a spoken self-introduction does (seen on the first DGX run); the rest of
   the `self_intro` source split (the reader's half is done, `93f49482c`).
