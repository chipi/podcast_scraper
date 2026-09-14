# Speaker attribution: how a name becomes a role, and how to debug it

Written 2026-09-14 after the #2065 / #2056 arc, for whoever next has to answer "why is this person
labelled wrong?"

The operator's original report was: *"I click on insights and there's always the same name listed
on all insights"* and *"we never actually show who is the guest. It's always a contributor."* Ten
distinct root causes were behind that. **Most of them lived in the seams between layers, not inside
any one layer** — which is why single-function tests were all green while the product was wrong.

---

## 1. The nine layers

A person's name travels this path. Each arrow is a place it can be lost, replaced, or
mis-attributed.

| # | layer | artifact / field | written by |
| --- | --- | --- | --- |
| 1 | **FEED** | RSS author, title, episode description | — |
| 2 | **HINT** | `detected_hosts` / `detected_guests` | `speaker_detectors/hosts.py` |
| 3 | **AUDIO** | `*.segments.json` → `speaker: SPEAKER_NN` | diarization provider |
| 4 | **ROSTER** | `content.speakers`, `speakers_source`, `diarization_num_speakers` | `providers/ml/diarization/roster.py` |
| 5 | **GRAPH** | `kg.json` → `Person.properties.role` | `_speaker_lists_for_graph` → `kg/pipeline.py` |
| 6 | **QUOTES** | `gi.json` → `SPOKEN_BY`, `speaker_id` | `gi/speakers.py` |
| 7 | **MENTIONS** | `gi.json` → `MENTIONS_PERSON` | `gi/relational_edges.py` |
| 8 | **IDENTITY** | `person:<slug>` / `person:unresolved-<name>-<ep>` | `identity/bare_name_scope.py`, `identity/intra_episode_merge.py` |
| 9 | **SURFACES** | cards, search, related-people | `server/app_kg_index.py`, `search/corpus_graph.py`, `server/cil_queries.py` |

### The two facts that explain most confusion

**`SPOKEN_BY` is in `gi.json`, NOT `kg.json`.** Measured across 287 production artifacts: `kg.json`
carries only `HAS_EPISODE` / `MENTIONS` / `HOSTS` / `GUESTS_ON`; 259 of the `gi.json` siblings carry
`SPOKEN_BY`. A guard that looked for it in the KG payload was a silent no-op on every real
artifact — and its unit test passed, because the fixture invented an edge type that shape never
has.

**The two layers mint DIFFERENT ids for one human.** The roster names a voice from ASR
(`person:aaron-levy`); the extractor reads the text (`person:aaron-levie`). Same episode, same
person, two nodes. Match across layers with `kg.speaker_coherence.same_person`, never by id alone.

---

## 2. Symptom → which artifact to open

| symptom | look at | likely layer |
| --- | --- | --- |
| everyone is "contributor" / no guests | `kg.json` roles vs `content.speakers` | 4→5 seam |
| the show's name is a host | `content.speakers` + `feed.title` | 2 or 4 |
| "Host" is a person | `kg.json` for `person:host` | 8 |
| the same name on every insight | `gi.json` quote `speaker_id` | 6 |
| a quote attributed to the wrong speaker | `gi.json` `SPOKEN_BY` + the transcript's `Name:` markers | 6 |
| two nodes for one person, same episode | `gi.json` + `kg.json` person nodes together | 8 |
| two nodes for one person, different episodes | `kg/entity_clusters.build_entity_id_map` | 9 |
| the graph is right but the app is wrong | the caches — see §5 | 9 |

Fetch any artifact from production:

```bash
GET /api/corpus/text-file?relpath=<path>     # X-API-Key header; serves .json too
```

The three siblings share a stem: `X.metadata.json`, `X.kg.json`, `X.gi.json`, and the transcript
plus `X.segments.json` under `transcripts/` relative to the **run root** (the metadata file's
grandparent).

---

## 3. Commands

```bash
make speaker-coherence CORPUS_DIR=<corpus>            # is the corpus self-consistent now
make speaker-migration-preview CORPUS_DIR=<corpus>    # what m0009 would do, + role transitions
make upgrade-undo-roles CORPUS_DIR=<corpus>           # roll the role changes back
python scripts/ops/undo_speaker_roles.py --corpus-dir <c> --show   # read the ledger, change nothing
```

---

## 4. Traps — things that LOOK like evidence and are not

This is the section worth reading twice. Every serious mistake in the #2065 arc was of this shape:
**a check that could not observe what it claimed to observe.**

### 4.1 A metric that shares the migration's own predicate

`check_no_show_as_speaker` and m0009's demotion both use `names_the_show`. So if the migration
wrongly demotes a real host, the coherence count scores that as a violation **FIXED**. The
headline "82 → 56 violations, 0 introduced" is structurally blind to the damage it was supposed to
rule out.

**Use the role-transition table instead** (`make speaker-migration-preview`). It reports
`host → mentioned: 21` per node, using no predicate at all.

### 4.2 A field that cannot distinguish a measurement from a fallback

```python
speakers      = diarized_speakers or _build_speakers_from_detected_names(...)   # roster or HINT
num_speakers  = len(raw_ids) or (len(named_order) or None)                      # count or FALLBACK
```

Both collapse two very different things into one field. `content.speakers_source` (#2070) now
labels the first on new ingests; `unknown` on an older artifact means **unknown**, not "diarized".

For the count, **read the segments sidecar** — distinct non-null `speaker` values are the real
measurement. 40/40 sampled production episodes had a reachable sidecar.

### 4.3 A fixture shape production never takes

Four separate times in one session a test passed because the fixture was wrong:

- a `kg.json` containing `SPOKEN_BY` (never happens)
- a roleless Person node absent from every fixture (happens in production)
- `_speaker_lists_for_graph` handed dicts when it reads `getattr(sp, "name")` — returns empty for
  everything, so every comparison silently compares nothing
- a staging corpus with no `*.segments.json` at all, making an undo's "0 refused" meaningless

**Before trusting a green test, confirm the fixture has the shape the artifact really has.** Grep a
real artifact for the key you are asserting on.

### 4.4 Editing source while a long suite runs

`test_generate_episode_metadata_installs_the_episode_fuse` uses `inspect.getsource()`. Edit the
file mid-run and it reads shifted lines. More broadly: a 55-minute suite measured against a moving
tree is void, not green.

### 4.5 A guard with no evidence to act on

`names_the_show(candidate, feed_title)` returns `False` on an empty title *by design* — "no title,
no opinion". A caller that omits `feed_title` therefore disables the guard it is calling. Measured:
without the title, **every** show name is kept.

**Pattern:** for each guard, feed it the signal it depends on and assert the answer CHANGES. A
guard whose output is identical with and without its input is inert, whatever it returns.

---

## 5. The graph is right but the app is wrong

Corpus-derived projections are cached in-process. After a migration or a re-enrich:

| cache | token |
| --- | --- |
| `app_kg_index`, catalog, momentum, top-persons, per-artifact loader | `perf_cache.corpus_mtime` — max mtime of `corpus_run_summary.json` / `corpus_manifest.json` / `upgrade_ledger.json` |
| `search/corpus_graph.get_corpus_graph` | same token (added #2069) |
| `server/cil_queries._cil_entity_id_map` | same token (added #2069) |

m0009 writes `*.kg.json` and the upgrade ledger; the ledger is in the token, so the caches do
invalidate. **Restart the API anyway** — it is the only thing that guarantees every in-process
cache is gone.

`get_corpus_graph(reconcile_hosts=True)` also demotes hosts with no attributed speech **at serve
time**, so that surface can disagree with `kg.json` legitimately.

---

## 6. Repairing existing artifacts

Two different breakages, two different repairs — a migration cannot fix the second:

| | on-disk state | repair |
| --- | --- | --- |
| **A** | roster is correct, graph never got it | `make upgrade-corpus` (m0009) — no LLM, no GPU |
| **B** | the roster is **itself** wrong (`['Host']`, an org) | `pipeline_stage=relabel_only` — re-resolves names on the frozen diarization, cascades GI/KG |

`relabel_only` needs no audio, no re-ASR, no re-diarization, and reindexes incrementally on its
own. It is exposed as an operator job:

```text
POST /api/jobs?pipeline_stage=relabel_only[&feed=<feed>][&profile=<dgx profile>]
```

**Undo before a re-enrich, or not at all.** The undo refuses any episode whose file hash changed
since the migration wrote it — and `rederive_only` / `rediarize_only` rewrite `kg.json`, so after
step 3 those refusals are correct: the re-enriched answer is the better one.

Full sequence: `docs/wip/POST-DEPLOY-SPEAKER-ATTRIBUTION-2026-09-14.md`.

---

## 7. Known-imperfect predicates

Do not assume these are sound; they are the best available, and each has a recorded failure mode.

| predicate | known failure |
| --- | --- |
| `names_the_show` | matches a title PREFIX, so a host whose name leads their own show (`Lex Fridman Podcast`, `Rich Roll Podcast`) reads as the show. Zero occurrences across the 55 production feeds today; m0009 reports these as `suspect_demotions` rather than guarding, because `SPOKEN_BY` inherits the roster's own mistakes. |
| `_are_xep_variants` | of 68 variant pairs over 287 artifacts, roughly half of those held apart only by `same_show_required` are different people (`Alex Bregman` / `Lex Friedman`). `same_show_required=True` is load-bearing — do not relax it. |
| `same_person` | token-subset means `John` matches `John Smith`; m0009 treats a multi-hit as AMBIGUOUS and refuses rather than guessing. |
| one-token rule | a person's name may drift in only ONE token. Costs the real pair `Alexander Carpi` / `Alexandra Karppi`; prevents `Albert Einstein` / `Bert Vogelstein`. Deliberate: a false split is clutter, a false merge reassigns quotes. |

---

## 8. Related

- `docs/wip/POST-DEPLOY-SPEAKER-ATTRIBUTION-2026-09-14.md` — the production repair sequence
- `src/podcast_scraper/kg/speaker_coherence.py` — the invariants, each with the defect it caught
- `src/podcast_scraper/upgrade/role_ledger.py` — what m0009 changed and how to reverse it
- `src/podcast_scraper/identity/roster_provenance.py` — measurement vs guess
- #2065 (the arc), #2056 (duplicate people), #2069 (reversibility), #2070 (provenance)
