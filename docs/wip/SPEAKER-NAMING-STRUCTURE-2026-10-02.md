# Speaker naming — what is structurally wrong, and one rule set to replace it

Status: DRAFT for advisor + operator review. No code changed for this yet.
Evidence: a read-only code inventory (file:line below, spot-checked by hand) and a read-only
measurement over prod (newest run per episode, 2026-10-02).

## 1. The problem in one paragraph

The pipeline answers two questions — WHO is in this episode, and WHICH voice is each of them — but
it answers each one several times, in several places, with different evidence orders and different
guards. A source that wins early stops the better sources behind it from ever being read; a source
that is computed is sometimes thrown away; a guard added for one incident blocks unrelated real
cases. Since #2075 (2026-09-19) a name that is never tied to a voice is hidden everywhere, so any
failure of the voice step now looks like "no host" in the app.

## 2. What exists today: six ladders and two cross-cutting duplicates

| # | Ladder | Question | Where |
|---|---|---|---|
| L0 | Transcript source | Do we have voices at all? (ASR+diarize / publisher cues with names / cues with no turns / plain text) | `episode_processor.py:4384-4620`, `transcript_formats/cues.py:31-50` |
| L1 | Show hosts | Who hosts the show? | `hosts.py:1644` + `processing.py:784-1360` |
| L2 | Episode metadata | Who does this episode's title/description name? | `processing.py:1773-1998`, provider `detect_speakers`, `corroboration.py` |
| L3 | Host seats | Which voice is a host? (6 steps, docstring says 5) | `roster.py:2812-2976` |
| L4 | Voice names | Which name goes on which voice? (publisher label > self-intro > introduced-by-host > LLM > forced pool name > guest arithmetic) | `roster.py:2978-3490`, `resolution.py:483` |
| L5 | Record + placement | host/guest, `placed` true/false, what the app shows | `metadata_generation.py:1024-1393`, consumers |
| X1 | "Same person?" | ≥8 separate implementations | `speaker_coherence.same_person`, `roster._same_person`, `_canonicalize_to_known_host`, `resolution._surname_variant`, `entity_clusters._are_xep_variants`, … |
| X2 | "Is this a person?" | ≥12 predicates, each applied on a different subset of paths | `has_org_markers`, `is_network_or_org_author`, `looks_like_publisher`, `names_the_show`, `drop_non_person_names`, `is_publishable_speaker_name`, `looks_like_a_person_name`, … |

## 3. Structural defects (each verified in code)

### 3.1 First-hit-wins instead of combining evidence (L1)
- A feed host *statement* that matches returns immediately; author tags are never read
  (`hosts.py:1658-1660`). "Two Carnegie Mellon faculty explore…" beats the author tag
  "Dan Saffer and Nik Martelaro" (AI and Design).
- A statement that matched but was *rejected* returns ∅ and blocks author tags too
  (`hosts.py:1661-1663`).
- The statement path applies only 3 of the ≥12 person guards (`hosts.py:701-722`), which is why a
  junk statement can win at all.

### 3.2 Narrow phrase patterns (L1)
- Title `with NAME$` takes ONE name (`hosts.py:639`): "…with Brené Brown and Adam Grant" → ∅.
- `.search` keeps only the first match per pattern (`hosts.py:693`).
- No "Join X", bare "hosts X", "with X & Y" mid-title, `Ó`-initial names; verb list misses
  tackle/interview/uncover/track. Measured: 32 of 80 prod feeds yield no host; at least 12 of them
  state hosts in plain English.

### 3.3 Three different host-of-show orders (L1)
- Deterministic: statement > author tags > title NER — but NER needs `nlp`, which
  `processing.py:804` never passes.
- LLM provider `detect_hosts`: returns author tags verbatim (`openai_provider.py:1457`); with no
  authors its LLM call can never return a host, because hosts are filtered to `known_hosts=∅`
  (`openai_provider.py:1733`). Dead branch.
- Dry-run preview uses a third order (`processing.py:708-732`).
- Per episode, the host set differs by path: ASR (`hosts_for_episode`), publisher-transcript path
  (raw `cached_hosts`, `processing.py:2088`), relabel/rediarize (bare `detect_hosts_from_feed`).

### 3.4 Evidence computed and thrown away
- Episode-description hosts ("X is joined by Y") are found (`processing.py:1941`) and then only used
  to filter guests; never passed on (`processing.py:1998`), though the comment says they are.
- Vocatives ("Hi, Adam." / "Hey, Brene.") are only ever a veto (`roster.py:1836`), never evidence.
- Plain-text publisher labels ("Nik:", "Dan:") have no parser (`cues.py:32-50`).
- `detect_hosts_from_transcript_intro`, `_merge_intro_guests`, `host_candidates`, `heuristics`,
  `max_names`, the config `known_hosts` fallback (`processing.py:1261`) — unreachable in prod.

### 3.5 Trusting any voice's own words (L4)
- A ≥2-token self-intro from any non-ad voice names it, cameos included (`roster.py:1719, 2686`).
  Promos for other shows ("I'm Phoebe Judge … the podcast Criminal") become placed guests.
- With no known host, the "host hint" voice for intro reading is just the opener
  (`roster.py:3150-3158`), which can be a promo; "journalist Ann Applebaum joins us" then binds a
  third-person mention to the next voice.

### 3.6 Placement only by forcing (L3/L4)
- A host name lands on a voice only when exactly one pool name and exactly one unnamed seat remain
  (`roster.py:1812-1856`); any extra name or seat → host unplaced → hidden.
- Seat step 3 (opener) has no dominance veto, step 4 does (`roster.py:2886` vs `2911`).
- `_GUEST_HOST_EPISODE` is IGNORECASE, so "we are in for a treat" marks the episode guest-hosted
  and refuses the host name (`roster.py:1747-1748`).

### 3.7 No voices → nobody (L0)
- Publisher cue files with no speaker turns are accepted (`require_transcript_speakers` defaults
  False; prod does not set it), so the episode is one voice and every name is unplaced.

## 4. Measured on prod (2026-10-02, read-only)

786 episodes carry the placement record (written since #2075); 1,314 older ones do not.

| Outcome | Episodes |
|---|---|
| host shown | 659 |
| no host: ladder found no host name | 65 (Conversations with Tyler 36, The Flip 10, MLST, Africa Tech Summit, Rest Is History, …) |
| no host: host name known, not tied to a voice | 56 (The Daily 8, a16z 8, Latent Space 7, Hard Fork 6, …) |
| no host: other | 4 |
| no host: publisher transcript with ≤1 voice | 2 |

Published names by source: self-intro 918, per-voice LLM 614, host pool forced 106, guest forced 34.
Hidden (unplaced): 638 show-host names, 175 episode-description guests, 29 hints.

## 5. Proposed rule set (one ladder per question)

### Rule 0 — one person test, one same-person test
- `is_person_name(name, show)` — the ONE predicate every source passes through, at the moment the
  name is proposed (union of today's org / publisher / show-name / role-word / junk checks).
- `same_person(a, b)` — the ONE identity matcher (given-name or surname short form, title strip,
  ASR spelling distance). Every other implementation becomes a call to it.

### Rule 1 — WHO: collect, don't race
Every source proposes candidates with provenance; nothing returns early. Candidates that pass Rule 0
are merged with `same_person`, keeping the best spelling.

| Source | Proposes | Strength |
|---|---|---|
| feed title "with A [and B]" | show host | strong |
| feed description host phrase (hosted by / hosts / co-hosts / join / with / NAMES + verb) | show host | strong |
| RSS author / owner tag (personal, not the show, not a network) | show host | strong |
| episode itunes:author | show host for that episode | medium |
| self-intro recurring across ≥3 episodes | show host | strong |
| episode title/description ("with guest X", "X is joined by Y") | guest (or episode host) | medium |
| in-transcript self-intro / publisher label / introduced-by-host | person in the room | strong for placement |

Conflict rule: a show host stated by TWO independent sources wins over one stated by one; a
candidate failing Rule 0 is dropped individually, never the whole source.

### Rule 2 — WHICH VOICE: evidence per (name, voice)
In order, first evidence wins per voice, each name used once:
1. publisher label on the voice's turns;
2. the voice introduces itself ("I'm X", "this is X");
3. another voice introduces it and it speaks next ("joined by X" → next voice);
4. it is addressed by name by another voice at the open ("Hi, Adam" → the replying voice is Adam);
5. LLM closed-list match (names only from Rule 1);
6. seat rule for show hosts with no direct evidence: the non-guest voice that opens or runs the
   show (one rule, same dominance guard everywhere).

Before any of it: a voice whose own words introduce ANOTHER show ("I'm X and I host the podcast Y",
"we're the hosts of Y") is a promo, not a guest — unless Y is this show.

### Rule 3 — no voices, no guessing (DONE, `3d7c95937`)
A publisher transcript is used only when it says who speaks. A cue file with no speaker turns, or
any plain-text file (no timings, so even `Name:` labels cannot place a quote on the audio), is
refused per episode and the audio is transcribed + diarized (`require_transcript_speakers: true` in
`prod_dgx_full`, operator 2026-10-02). New episodes only. Never publish a name with no voice.

## 6. Case traces under the proposed rules

| Case | Today fails at | Under the rule set |
|---|---|---|
| Curiosity Shop | `hosts.py:639` single-name title pattern; LLM provider returns the org author | Rule 1 title "with A and B" → Brené Brown, Adam Grant; Rule 2.4 vocatives place them; promo rule drops Phoebe Judge / Jon Finer; Ann Applebaum is third-person → never a voice |
| AI and Design | `hosts.py:1658` junk statement beats author tag; plain-text labels unparsed | Rule 0 drops "Two Carnegie Mellon"; author tag → Dan Saffer, Nik Martelaro; Rule 3 transcribes + diarizes; their self-intros ("I'm Nik Martelaro") place them |
| Every | VTT without turns accepted (`episode_processor.py:4456`) | Rule 3 → diarize audio; "Hey, this is Dan Shipper" → Rule 2.2 |
| How I Write | host 15%, guest opens; placement only by forcing | Rule 2.6 seat rule seats the non-guest voice; host introduces the guest ("Andy Weir is on the show today") → it is the host voice |
| Conversations with Tyler (36) | not traced yet | to trace |

## 7. Immediate bugs independent of the redesign
1. `_GUEST_HOST_EPISODE` IGNORECASE (`roster.py:1747`).
2. Statement beats / rejection blocks author tags (`hosts.py:1658-1663`).
3. Single-name title pattern (`hosts.py:639`).
4. LLM provider `detect_hosts` dead branch (`openai_provider.py:1733`).
5. Episode-description hosts dropped (`processing.py:1998`).
6. Provenance mislabel: all `tried.known_hosts` written as `feed_statement` (`metadata_generation.py:1063`).
7. Doc/code divergences: `config.py:4004` vs `episode_processor.py:4594`; `roster.py:2829`; `processing.py:1928`.

## 8. NOT verified / open
- Rule 1 and 2 are a design; nothing is replayed yet. The bar: replay the corpus old vs new and read
  every changed voice (roster_replay), plus unit tests per source and per placement rule.
- Conversations with Tyler (36 no-host) not traced.
- Whether vocative placement (Rule 2.4) mis-binds on panels (more than two voices) — unmeasured.
- Rule 3 cost: DGX transcription for publisher-transcript episodes without turns (83 such episodes exist today; new ones only from now on).
- The 1,314 pre-#2075 episodes carry no placement record and were not measured.
