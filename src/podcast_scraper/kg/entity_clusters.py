"""Corpus-wide entity canonicalization — cross-episode spelling drift (#852).

The same person/org appears under different slugs across episodes of a show
(`Cargil`/`Cargill`, `Data Bricks`/`Databricks`, `Tracy`/`Tracey Alloway`) — ASR
proper-noun drift the within-episode fix (#851) and the per-episode bridge can't
catch. This builds a corpus-wide ``variant_id → canonical_id`` map, the missing
analog to topic clustering (RFC-075) for entities.

Design (evidence: ~95% of drift is same-show, frequency-dominant canonical):

- **Conservative, string-based** — names drift by *spelling*, not semantics, so we
  cluster by string similarity, not embeddings.
- **Same-show required** — two variants merge only if they share a podcast (cuts
  false merges; 95% of drift is same-show anyway).
- **Canonical = highest frequency** — the spelling in the most episodes wins
  (`Odd Lots` ×13 over the 1-episode garbles).
- **Guards (the balanced check):** acronym guard (`UPS` ≠ `USPS`); version /
  distinguishing-token guard (`Claude` ≠ `Claude 3`); a differing *content* token
  that is not itself a spelling variant blocks the merge (`Bloomberg Audio` ≠
  `Bloomberg Media Studios`).

The map is applied at read-time via ``CorpusGraph.build(identity_map=…)`` —
reversible, no artifact rewrite. Threshold precision tuning is deferred to
autoresearch. See issue #852.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..builders.bridge_builder import strip_layer_prefixes
from .filters import _clean_entity_name, _is_acronymish

logger = logging.getLogger(__name__)

ENTITY_CLUSTERS_SCHEMA_VERSION = "1.0"

# Thresholds tuned in #853 (data/eval/runs/baseline_entity_canon_v1).
# Silver eval: 190 candidate pairs from a real prod corpus
# (`.test_outputs/manual/my-manual-run-10`), Sonnet 4.6 silver labels
# (49 SAME / 134 DIFFERENT / 7 BORDERLINE).
#
# Baseline (token=0.78, overall=0.85): P=1.00, R=0.31, F1=0.47
# Tuned    (token=0.65, overall=0.70): P=1.00, R=0.49, F1=0.66
# +60% recall at preserved 100% precision. Caught: Bessent/Bessett,
# Tracy Alloway/Allaway, Joe Weisenthal/Wassenthal, Tim Geithner/Geidner,
# Henry Blodget/Blodgett, etc. The recall ceiling is structural (token-
# count mismatches like "Dr. Elena Fischer" vs "Elena Fischer") — predicate
# redesign tracked in #904, not threshold tuning.
_TOKEN_RATIO = 0.65  # per-aligned-token spelling-variant floor
_OVERALL_RATIO = 0.70  # whole-string floor
_VERSION_TOKEN_RE = re.compile(r"\d")  # a differing token containing a digit blocks merge

# Nicknames where the canonical pair differs by too much for the ratio test
# (Michael→Mike, Robert→Rob/Bob) but the people are the same. Lowercase →
# lowercase, bidirectional (both directions added at module load). Tuned in
# #904 from the #853 silver-eval miss catalogue. NOT meant to be exhaustive —
# the catch is "the long-tail of recall improvements at preserved precision":
# add a pair only after you've seen the miss class in real prod data.
_NICKNAME_MAP: dict[str, set[str]] = {
    "michael": {"mike"},
    "nicholas": {"nick"},
    "elizabeth": {"liz", "beth", "betsy"},
    "jerome": {"jerry", "jay", "j."},
    "emmanuel": {"manny"},
    "richard": {"rich", "rick", "dick"},
    "robert": {"rob", "bob", "bobby"},
    "william": {"will", "bill", "billy"},
    "thomas": {"tom", "tommy"},
    "joseph": {"joe", "joey"},
    "anthony": {"tony"},
    "patrick": {"pat"},
    "stephen": {"steve"},
    "steven": {"steve"},
    "matthew": {"matt"},
    "jonathan": {"jon"},
    "christopher": {"chris"},
    "samuel": {"sam"},
    "alexander": {"alex", "al"},
    "daniel": {"dan", "danny"},
    "edward": {"ed", "eddie", "ted"},
    "andrew": {"andy", "drew"},
    "benjamin": {"ben"},
    "katherine": {"kate", "katie", "kathy"},
    "catherine": {"cathy", "kate"},
    "rebecca": {"becky"},
    "susan": {"sue"},
    "deborah": {"debbie", "deb"},
    "barbara": {"barb"},
    "patricia": {"pat", "patty", "trish"},
    "jennifer": {"jen", "jenny"},
    "margaret": {"maggie", "meg", "peggy"},
}
# Symmetric closure so lookup works either direction.
_NICKNAME_PAIRS: set[tuple[str, str]] = set()
for _full, _nicks in _NICKNAME_MAP.items():
    for _n in _nicks:
        # Strip trailing dot so "j." stored as "j" — `_nickname_token_equiv` does
        # the same dot-strip on input, so lookups round-trip.
        _n_norm = _n.rstrip(".")
        _NICKNAME_PAIRS.add((_full, _n_norm))
        _NICKNAME_PAIRS.add((_n_norm, _full))

# Title / honorific prefixes the predicate strips before comparing — handles
# `Dr. Elena Fischer` vs `Elena Fischer`, `Ayatollah Ali Khamenei` vs
# `Ali Khamenei`. Lowercase, trailing dot optional.
_TITLE_PREFIXES: frozenset[str] = frozenset(
    {
        "dr",
        "dr.",
        "mr",
        "mr.",
        "mrs",
        "mrs.",
        "ms",
        "ms.",
        "prof",
        "prof.",
        "professor",
        "sir",
        "ayatollah",
        "rabbi",
        "imam",
        "father",
        "sister",
        "brother",
        "president",
        "senator",
        "rep",
        "rep.",
        "gov",
        "gov.",
        "judge",
        "justice",
        "captain",
        "lieutenant",
        "lt",
        "lt.",
        "general",
        "gen",
        "gen.",
        "ambassador",
    }
)


@dataclass
class EntityCandidate:
    """One canonical entity id observed across the corpus."""

    id: str
    kind: str  # "person" | "org"
    name: str  # representative display name (most frequent)
    episodes: Set[str] = field(default_factory=set)
    shows: Set[str] = field(default_factory=set)
    _name_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def freq(self) -> int:
        return len(self.episodes)


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _strip_title_prefix(tokens: List[str]) -> List[str]:
    """Drop a leading title token like 'Dr.' or 'Ayatollah'. Returns the rest."""
    if tokens and tokens[0].rstrip(".") in {p.rstrip(".") for p in _TITLE_PREFIXES}:
        return tokens[1:]
    return tokens


def _nickname_token_equiv(x: str, y: str) -> bool:
    """True if x and y are the same first-name modulo nickname (case-insensitive).

    Tolerates trailing-dot variants on both sides so `J.` and `j` and `j.`
    all canonicalise to the same lookup ("j" → set containing "jerome",
    plus "j." → same set after rstrip).
    """
    xn, yn = x.rstrip("."), y.rstrip(".")
    if xn == yn:
        return True
    return (xn, yn) in _NICKNAME_PAIRS


def _token_count_tolerant_match(a: str, b: str, kind: str) -> bool:
    """Handle token-count mismatches the strict aligned predicate rejects.

    Covers three patterns from the #853 silver-eval miss catalogue (#904
    scope):
    1. **Title-prefix strip** — `Dr. Elena Fischer` ↔ `Elena Fischer`,
       `Ayatollah Ali Khamenei` ↔ `Ali Khamenei`. After stripping a leading
       title from either side, retry the strict predicate.
    2. **Family-only reference** — `Carney` ↔ `Mark Carney`, `Trump` ↔
       `Donald Trump`. The single-token side must match the LAST token of
       the multi-token side EXACTLY (no fuzzy on last name alone — too
       risky, e.g. `Brown` vs `Mark Brown` is too thin a match).
    3. **First-name-only alias** — `Liam` ↔ `Liam Verbeek`. Symmetric to
       (2): single-token first name matches FIRST token of multi-token name
       exactly. Restricted to ``kind == "person"`` because the same pattern
       on orgs (`Adobe` ↔ `Adobe Creative Cloud`) is more likely a
       distinct sub-product than an alias.

    NOT covered (out of scope, would need richer signal):
    - Both sides multi-token but with different counts (e.g.
      ``Joe Eisenthal House`` vs ``Joe Weisenthal`` — Whisper artefact
      that inserts a spurious token mid-name).
    """
    ta, tb = a.split(), b.split()
    if len(ta) == len(tb):
        return False  # caller already handled equal-count case
    # 1. Title-prefix strip on either side (or both).
    sa, sb = _strip_title_prefix(ta), _strip_title_prefix(tb)
    if (sa != ta or sb != tb) and len(sa) == len(sb) and sa == sb:
        return True
    if kind == "person":
        short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        if len(short) == 1 and len(long_) >= 2:
            # 2. Family-only reference: short token matches long's LAST token exactly.
            #    e.g. `Carney` ↔ `Mark Carney`, `Trump` ↔ `Donald Trump`.
            if short[0] == long_[-1]:
                return True
            # NOTE: a first-name-only-merge rule (short matches long's first
            # token) was attempted in an earlier #904 draft but removed —
            # it can't distinguish ASR aliases (`Liam` ↔ `Liam Verbeek`,
            # same person) from organic same-first-name pairs (`Marco` ↔
            # `Marco Bianchi`, different people, the v2 two-Marcos test).
            # Both have identical predicate shape; differentiating needs
            # external signal (shared-episode evidence, same-show speaker
            # role, or LLM escalation). Tracked for follow-up; the Liam
            # alias case is fixture-only today and real-prod first-name-only
            # references rarely arise from ASR.
    return False


def _are_xep_variants(name_a: str, name_b: str, kind: str) -> bool:
    """Conservative cross-episode variant test (spelling drift, not distinct names)."""
    a, b = _clean_entity_name(name_a), _clean_entity_name(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Up-front title-prefix normalization so `President Trump` vs `Donald Trump`
    # (equal token counts after stripping a title from one side) routes through
    # the tolerant family-only-reference path instead of being rejected by the
    # token-ratio test. We compare both as-is AND title-stripped versions —
    # whichever pair matches.
    ta_full, tb_full = a.split(), b.split()
    ta_stripped, tb_stripped = _strip_title_prefix(ta_full), _strip_title_prefix(tb_full)
    if ta_stripped != ta_full or tb_stripped != tb_full:
        stripped_a, stripped_b = " ".join(ta_stripped), " ".join(tb_stripped)
        if stripped_a and stripped_b and stripped_a == stripped_b:
            return True
        # If only one side had a title prefix, recurse into the tolerant path
        # so the now-shorter name can match by family-only-reference.
        if (ta_stripped == ta_full) != (tb_stripped == tb_full):
            short, long_ = (
                (ta_stripped, tb_full) if ta_stripped != ta_full else (tb_stripped, ta_full)
            )
            # Family-only-reference only (no first-name-only-merge — see the
            # NOTE in `_token_count_tolerant_match` for why).
            if kind == "person" and len(short) == 1 and len(long_) >= 2 and short[0] == long_[-1]:
                return True
    # Spacing variants: "Data Bricks" == "Databricks", "Chat GPT" == "ChatGPT".
    if a.replace(" ", "") == b.replace(" ", "") and a.replace(" ", ""):
        return True
    ta, tb = ta_full, tb_full
    # Different token counts → tolerant matcher (title-prefix, family-only,
    # first-name-only). Runs BEFORE the acronym guard because short proper
    # nouns like `Trump`/`Liam`/`Brown` look acronym-ish to that guard but
    # are valid family-only-reference candidates when they appear as a
    # complete token in the other (longer) name. The tolerant matcher uses
    # exact-token equality, not fuzzy similarity, so the UPS-vs-USPS concern
    # the acronym guard exists to address doesn't apply on this path.
    # #904 lift: this branch was unconditional reject before #904.
    if len(ta) != len(tb):
        return _token_count_tolerant_match(a, b, kind)
    # Acronyms never fuzzy-merge (UPS != USPS) — guard applies to the
    # equal-token-count branch where we'd otherwise apply ratio similarity.
    if _is_acronymish(name_a, a) or _is_acronymish(name_b, b):
        return False
    if _ratio(a, b) < _OVERALL_RATIO:
        # One escape hatch: same token-count, otherwise-identical names, but
        # one token differs by nickname (Mike Selig vs Michael Selig). The
        # ratio test rejects because Mike/Michael are too dissimilar — but
        # they ARE the same first name.
        if (
            kind == "person"
            and len(ta) == len(tb)
            and sum(1 for x, y in zip(ta, tb) if x != y) == 1
        ):
            diff_pairs = [(x, y) for x, y in zip(ta, tb) if x != y]
            if diff_pairs and _nickname_token_equiv(*diff_pairs[0]):
                return True
        return False
    # Token-aligned: every differing token pair must be a spelling variant, not a
    # distinct content word (audio vs media) or a version token (3, v2).
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if _VERSION_TOKEN_RE.search(x) or _VERSION_TOKEN_RE.search(y):
            return False  # numeric/version distinction
        if _ratio(x, y) < _TOKEN_RATIO and not _nickname_token_equiv(x, y):
            return False  # distinct words, not a spelling variant
    return True


def collect_entity_candidates(corpus_dir: Path | str) -> Dict[str, EntityCandidate]:
    """Aggregate person/org entities corpus-wide with episode frequency + shows."""
    from .corpus import load_kg_artifacts, scan_kg_artifact_paths

    out: Dict[str, EntityCandidate] = {}
    for _path, data in load_kg_artifacts(scan_kg_artifact_paths(Path(corpus_dir))):
        episode_id = str(data.get("episode_id") or "")
        show = ""
        for node in data.get("nodes") or []:
            if isinstance(node, dict) and node.get("type") == "Episode":
                props = node.get("properties") or {}
                show = str(props.get("podcast_id") or props.get("feed_id") or "")
                break
        for node in data.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            nid = strip_layer_prefixes(str(node.get("id") or ""))
            kind = nid.split(":", 1)[0]
            if kind not in ("person", "org"):
                continue
            props = node.get("properties") or {}
            name = str(props.get("name") or props.get("label") or "").strip()
            if not name:
                continue
            cand = out.get(nid)
            if cand is None:
                cand = EntityCandidate(id=nid, kind=kind, name=name)
                out[nid] = cand
            if episode_id:
                cand.episodes.add(episode_id)
            if show:
                cand.shows.add(show)
            cand._name_counts[name] = cand._name_counts.get(name, 0) + 1
    # Representative display name = most frequent surface form.
    for cand in out.values():
        if cand._name_counts:
            cand.name = max(cand._name_counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return out


def _pick_canonical(members: List[EntityCandidate]) -> EntityCandidate:
    """Highest frequency wins; tie → longest name, then lexical."""
    return sorted(members, key=lambda c: (-c.freq, -len(c.name), c.name.lower()))[0]


def build_entity_canonical_map(
    candidates: Dict[str, EntityCandidate],
    *,
    same_show_required: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Cluster variants and return ``(entity_clusters_payload, variant→canonical map)``."""
    by_kind: Dict[str, List[EntityCandidate]] = {}
    for cand in candidates.values():
        by_kind.setdefault(cand.kind, []).append(cand)

    id_map: Dict[str, str] = {}
    clusters_out: List[Dict[str, Any]] = []

    for kind, items in sorted(by_kind.items()):
        # High-frequency first so the dominant spelling seeds (and wins) each cluster.
        items_sorted = sorted(items, key=lambda c: (-c.freq, c.name.lower()))
        clusters: List[List[EntityCandidate]] = []
        for cand in items_sorted:
            for cluster in clusters:
                if same_show_required and not any(cand.shows & m.shows for m in cluster):
                    continue
                if any(_are_xep_variants(cand.name, m.name, kind) for m in cluster):
                    cluster.append(cand)
                    break
            else:
                clusters.append([cand])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            canonical = _pick_canonical(cluster)
            members_json = []
            for m in sorted(cluster, key=lambda c: (-c.freq, c.id)):
                if m.id != canonical.id:
                    id_map[m.id] = canonical.id
                members_json.append({"id": m.id, "name": m.name, "episode_count": m.freq})
            clusters_out.append(
                {
                    "canonical_id": canonical.id,
                    "canonical_name": canonical.name,
                    "member_count": len(cluster),
                    "members": members_json,
                }
            )

    payload = {
        "schema_version": ENTITY_CLUSTERS_SCHEMA_VERSION,
        "same_show_required": same_show_required,
        "entity_count": len(candidates),
        "cluster_count": len(clusters_out),
        "merged_variants": len(id_map),
        "clusters": clusters_out,
    }
    return payload, id_map


def build_entity_id_map(
    corpus_dir: Path | str, *, same_show_required: bool = True
) -> Dict[str, str]:
    """Convenience: corpus → ``variant_id → canonical_id`` map for ``CorpusGraph``."""
    candidates = collect_entity_candidates(corpus_dir)
    _payload, id_map = build_entity_canonical_map(candidates, same_show_required=same_show_required)
    return id_map


def id_map_from_clusters_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """Reconstruct the ``variant_id → canonical_id`` map from a saved payload."""
    out: Dict[str, str] = {}
    for cluster in payload.get("clusters") or []:
        canonical = cluster.get("canonical_id")
        if not canonical:
            continue
        for member in cluster.get("members") or []:
            mid = member.get("id")
            if mid and mid != canonical:
                out[mid] = canonical
    return out


__all__ = [
    "EntityCandidate",
    "build_entity_canonical_map",
    "build_entity_id_map",
    "collect_entity_candidates",
    "id_map_from_clusters_payload",
]
