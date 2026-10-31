---
id: s5b_person_language
class: full
status: open
issue: none (RFC-131 S5, L1–L5; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/identity/person_match.py
  - src/podcast_scraper/kg/entity_clusters.py
  - src/podcast_scraper/languages.py
  - src/podcast_scraper/server/corpus_catalog.py
  - src/podcast_scraper/quality/name_report.py
  - src/podcast_scraper/speaker_detectors/naming_vocabulary.py
---

# s5b_person_language — the person rule reads each name's language: per-language titles, nicknames and folds; a name of unknown language joins only when equal

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3 step S5, second
half: items L1, L2, L4 and L5. L3 is covered under Diagnosis.

## 1. Intake

- Plan V3, "Languages", items L1–L5 (eval-data repo).
- `s5a_person_variants.md`: the rule this extends; this change ships in the same deploy (D2).
- `languages.py` (`normalize_language_tag`, `primary_language`) and `server/corpus_catalog.py`
  `_episode_language`: the served rule for an episode's language, which is the episode's tag, else
  the feed's.
- `speaker_detectors/naming_vocabulary.py`: the per-language vocabulary module (de, en, es, fr, it,
  pt rows) that L1 names as the home.
- Measured before this record, on snapshot-prod-20261030:
  - all 80 feeds and all 3,069 KG episodes are `en`;
  - all 10,444 person candidates and all 720 D1 person joins come from `en` shows;
  - reading every episode's metadata for its language costs 0.4 s (39 MB) on that snapshot.

## 2. Outcome and baseline

- **Prod:** no non-English or untagged show exists, so this change has no prod population. Its
  baseline is S5a's ledger joins on the snapshot corpus, built from the corpus itself and not from the
  S4 export, because the export carries no languages.
- **Fixture:** the app-validation corpus v3 has non-English feeds (p10–p14). Its ledger joins by
  language are measured before and after as a test, not as evidence (memory: fixtures are not
  evidence).

## 3. Diagnosis by loss

No counted prod loss: there are no cases on prod. The items are rules for shows the corpus will
gain:
- **L1:** titles and nicknames are English lists, applied to every language. A German "Herr" or "Frau"
  is not stripped, and English nicknames apply to German first names.
- **L2:** a name whose show declares no language gets the English fuzzy rules. A missing vocabulary
  should split, never merge.
- **L3:** phonetic matching (Soundex) is not used anywhere in the identity home. Its only users are the
  roster's own rules (`providers/ml/diarization/roster.py`, `speaker_detectors/name_canonicalization.py`),
  which belong to #2403 (the roster's same-person rules), outside this arc. Nothing to change here.
- **L4:** cleaning folds `ü` to `u`, so `Müller` and `Mueller` differ by one letter and join only
  through the fuzzy ratio. German writes the umlaut as `ue`, `oe`, `ae`, and `ß` as `ss`.
- **L5:** the family-only reference ("Carney" ~ "Mark Carney") matches the LAST word. In
  family-name-first languages (zh, ja, ko, hu, vi) the family name comes first.

## 4. Prediction

Dated: 2026-10-31

The rule:
- **A candidate's languages:** the primary subtags of its episodes' languages, read with the catalog's
  rule (the episode's tag, else the feed's), now one public function in `languages.py`.
- **A pair's language:** the one language both candidates share, when there is exactly one; otherwise
  none.
- **`en`:** exactly today's rule. The title and nickname lists move to the vocabulary module
  unchanged.
- **de, es, fr, it, pt:** that row's titles (`HONORIFIC_TITLES`), no nicknames (no table exists),
  and the same edit-ratio rules. `de` adds a fold before cleaning: ä→ae, ö→oe, ü→ue, ß→ss.
- **zh, ja, ko, hu, vi:** the family-only reference matches the FIRST word. Their titles: none.
- **Who gets a language (corrected after the fresh read):**
  - `language=None` keeps today's rule exactly, so every caller that does not pass a language is
    unchanged;
  - the language-aware rules run only when the person join is passed a language: `en`, a supported
    code, or `und` for an episode with no tag, set by candidate collection;
  - organisations never get language rules; their join stays mode `cluster` with no language.
  - Callers that pass no language and keep today's rule exactly:
    - `identity/intra_episode_merge.py`, both calls;
    - `search/corpus_graph.py`, `host_variants`;
    - the org path in `kg/entity_clusters.py` `_JoinTest._match`;
    - any direct `cluster_variants` call.
  - Callers that get the candidates' languages: both paths that build from
    `collect_entity_candidates`. One is the ledger (`identity/ledger.py` `build_ledger`). The other
    is the read-time `build_entity_id_map`, used through `cached_entity_id_map` by the viewer,
    player and cil reads when not served from the ledger.
- **Unknown or unsupported language, `und` or a code with no vocabulary row (L2):** two names join only when they are equal after
  normalisation, or differ only in spacing or hyphens. No title strip, nickname, family-only reference,
  fuzzy ratio, hyphen retry or middle initial. The guards apply as before.
- **`RULE_VERSION`:** stays `/3`, shared with S5a. The version is bumped once per deploy, so D1's
  `/2` ledger stays one step back (`ok_previous_rule`) when D2 deploys.

Predicted:
- **Snapshot corpus (prod):** the full ledger join list (`d1/d1_ledger_joins.py`) is byte-identical
  to S5a's, built with S5a's code from a git worktree and with this change's code, both over
  snapshot-prod-20261030. Every candidate there is `en`. Any difference stops the change.
- **Counts on snapshot-prod-20261030** (eval-data `s5/langcount.py`):
  - untagged episodes: 0 of 3,069 (`en` 3,069);
  - person ids in more than one language: 0 of 10,444.
  So no prod candidate is `und` or mixed.
- **D2 step 10:** prod reports the same two counts on the live corpus, including the 22 episodes
  added since the snapshot, before D2's ledger rebuild. A non-zero count is listed and read before the
  rebuild.
- **The read-time map** (`build_entity_id_map`) is byte-identical on the snapshot before and after, as
  well as the ledger.
- **Unit scenarios (invented names):**
  - `de`: Müller ~ Mueller joins exactly, and "Herr" is stripped;
  - `en`: unchanged;
  - an unknown language: "Korin Vael" / "Korin Vaal" do not join, "Korin Vael" / "korin vael" do;
  - `zh`: a one-word family name joins the full name that starts with it;
  - a candidate in two languages, with no single shared one: neutral rules.
- **Fixture corpus (a test, not evidence):**
  - 55 episodes, of which 15 are non-English (es, it, fr, de, pt, 3 each); 0 untagged; 49 person ids;
  - person joins on the non-English feeds change only by L1 and L4. Every changed join is listed.
- **What a listener sees:**
  - on prod, nothing: the person rows in `/api/corpus/persons` and the viewer's merged-entity counts
    are unchanged, because the map is identical;
  - for a future non-English show: an umlaut pair (Müller / Mueller) joins by the `de` fold rather
    than the fuzzy ratio, a `de` title is stripped before comparing, and an untagged show's names join
    only when equal.
- **Also unchanged on the snapshot:** the blocked joins and the join count in the ledger payload.
- **Scenario test:** the within-episode merge of two spellings ("Levy" / "Levie" class) still merges.
- **Scenario test:** an org of unknown language still fuzzy-joins.
- **Must not move:**
  - the org rule (org joins byte-identical on the snapshot);
  - the catalog's served episode language: the function moves without a behaviour change, and its
    tests pass unchanged.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, about 1.5 minutes: "Not
complete". It found that the first draft's L2 ("no language means equal only") would have changed
every caller that passes no language today:
- the within-episode merge;
- hosts;
- organisations;
- the read-time map.

The design is corrected: None keeps today's rule, `und` is explicit, and people only. Rows added: the
callers; the org path; the untagged and multi-language counts (both 0 on the snapshot); a step-10
check on prod's live counts; the read-time map identical; what a listener sees.

Spot checks for step 8:
- blocked joins and the join count unchanged;
- `build_entity_id_map` identical on the snapshot;
- the within-episode merge scenario still merges;
- one fixture non-English changed join read in the after-state;
- the `_episode_language` tests pass unchanged.

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- No prod data measures any non-English rule, because prod has no non-English show. The rules are
  judged by tests and by the fixture, which is not evidence.
- `gi/repair.py` reads the language inline, without normalising it. It is left alone, because moving
  it to the shared function would change its behaviour for unnormalised tags.
- L3 (phonetic matching) has no user in the identity home; the roster's uses are #2403.
