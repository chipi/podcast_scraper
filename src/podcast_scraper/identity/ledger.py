"""The identity ledger: every cross-episode identity decision, recorded (ADR-166, RFC-131).

A deterministic step builds the ledger from corpus text and writes it next to the other derived
corpus files. The server applies it at read time. The ledger IS the decision, not a description
written beside it, so what is recorded and what is served cannot disagree.

Failure mode (ADR-166): when the ledger is missing, unreadable, or written by another schema or
rule version, there are NO cross-episode joins. Every person id stands alone, exactly as on the
episode it came from. The in-memory unguarded build is never a serving fallback: it is the very
map the guards (ADR-167) exist to correct.

A corpus newer than the ledger (an ingest, a migration or a relabel ran since) is served from the
ledger as it is. New episodes have no joins yet; a changed episode keeps its previous joins until
the next build. ``ledger_status`` reports that gap.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The ledger and its change log, at the corpus root beside the other derived corpus files.
LEDGER_NAME = "identity_ledger.json"
CHANGES_NAME = "identity_ledger_changes.jsonl"
#: The last valid ledger before the current one: the undo. ``restore_previous`` puts it back. A
#: missing ledger means no cross-episode joins at all (below), so the undo is never "delete it".
PREVIOUS_NAME = "identity_ledger.previous.json"

#: The file's shape. A reader refuses any other (ADR-166).
LEDGER_SCHEMA_VERSION = "identity-ledger/1"

#: The rules that decided the joins. Bump it whenever a rule or guard changes what joins, and move
#: the old value to ``PREVIOUS_RULE_VERSION``.
RULE_VERSION = "cluster-person-no-bridge-guarded-publisher-verified/1"

#: The rule version one step back, still served (state ``ok_previous_rule``) until the rebuild.
#: Without it a deploy that bumps the rule serves no joins between the new api starting and the
#: ledger rebuild, and a rebuild before the deploy is refused by the api still running (RFC-131
#: S5a). Any older version is refused: no joins.
PREVIOUS_RULE_VERSION = "cluster-person-no-bridge-guarded/3"

#: The person rule the ledger's joins use (RFC-131 S4): mode ``cluster_person`` (one-word names
#: join only as one spelling), no one-word bridge between full names, and the guards (ADR-167).
PERSON_MODE = "cluster_person"
PERSON_LINKAGE = "no_one_word_bridge"

#: Where the ledger's publisher evidence comes from (record ``l_publisher_verified_mode``):
#: ``extracted``, the diagnostics' extracted names as every build before it, or
#: ``publisher_verified``, checked against and supplemented from the publisher's own text, with
#: its host-seat joins. Shipped ``extracted`` on 2026-11-05, switched on by the operator on
#: 2026-11-06; the first refresh after the deploy respells the files.
#: A build is stamped with its own mode's version, whichever mode ships.
EVIDENCE_MODE = "publisher_verified"
RULE_VERSION_BY_EVIDENCE = {
    "extracted": "cluster-person-no-bridge-guarded/3",
    "publisher_verified": "cluster-person-no-bridge-guarded-publisher-verified/1",
}

#: Ledger states, for logs and the state gauge.
STATE_OK = "ok"
STATE_OK_PREVIOUS_RULE = "ok_previous_rule"
STATE_MISSING = "missing"
STATE_UNREADABLE = "unreadable"
STATE_VERSION_MISMATCH = "version_mismatch"
LEDGER_STATES = (
    STATE_OK,
    STATE_OK_PREVIOUS_RULE,
    STATE_MISSING,
    STATE_UNREADABLE,
    STATE_VERSION_MISMATCH,
)

#: Changes listed per build in the change log; the counts are always complete.
_MAX_LISTED_CHANGES = 200


def corpus_token(root: Path | str) -> Dict[str, Any]:
    """The corpus state a ledger was built from: content mtime and the newest-run kg count.

    The content mtime excludes the ledger itself (``perf_cache.corpus_content_mtime``): a token
    that included the file it is written into could never match after the write.
    """
    from podcast_scraper.kg.corpus import newest_run_artifact_paths, scan_kg_artifact_paths
    from podcast_scraper.perf_cache import corpus_content_mtime

    root = Path(root)
    kg = newest_run_artifact_paths(root, scan_kg_artifact_paths(root), ".kg.json")
    return {"corpus_mtime": corpus_content_mtime(root), "newest_kg_count": len(kg)}


def build_ledger(root: Path | str, evidence_mode: Optional[str] = None) -> Dict[str, Any]:
    """Build the ledger for a corpus root, from corpus text only. Writes nothing.

    The joins are the cluster rule's for organisations. For people: rule mode ``PERSON_MODE``,
    linkage ``PERSON_LINKAGE``, and the identity guards inside the join test (ADR-167); each
    refused join is recorded in ``blocked_joins``. The reviewed near-spelling list (record s5d)
    applies as the build's ``pair_rule``. A group keeps the canonical id the served
    ledger gives it while that id is still a member (RFC-131 0041).

    ``evidence_mode`` (default ``EVIDENCE_MODE``): ``publisher_verified`` builds with the
    publisher's own text as evidence and its host-seat joins (record l_publisher_verified_mode),
    stamped with that mode's rule version.
    """
    import podcast_scraper
    from podcast_scraper.identity.guards import (
        candidate_guard,
        collect_episode_evidence,
        host_join_pair_rule,
        load_reviewed,
        reviewed_pair_rule,
    )
    from podcast_scraper.kg.entity_clusters import (
        build_entity_canonical_map,
        collect_entity_candidates,
    )

    mode = evidence_mode or EVIDENCE_MODE
    if mode not in RULE_VERSION_BY_EVIDENCE:
        raise ValueError(f"unknown evidence mode {mode!r}")
    root = Path(root)
    candidates = collect_entity_candidates(root)
    evidence = collect_episode_evidence(root, mode=mode)
    served, _state = read_ledger(root)
    pair_rule = reviewed_pair_rule(evidence, load_reviewed())
    if mode == "publisher_verified":
        pair_rule = host_join_pair_rule(evidence, pair_rule)
    payload, _id_map = build_entity_canonical_map(
        candidates,
        guard=candidate_guard(evidence),
        person_mode=PERSON_MODE,
        linkage=PERSON_LINKAGE,
        keep_canonical=_canonical_ids(served),
        pair_rule=pair_rule,
    )
    _add_display_spellings(payload, candidates, evidence)
    blocked = payload.pop("blocked_joins", [])
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "rule_version": RULE_VERSION_BY_EVIDENCE[mode],
        "code_version": getattr(podcast_scraper, "__version__", "unknown"),
        "corpus_token": corpus_token(root),
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "clusters_payload": payload,
        "blocked_joins": blocked,
    }


def _canonical_ids(ledger: Optional[Dict[str, Any]]) -> set:
    """The canonical ids a served ledger gives; a rebuild keeps each one that is still a member."""
    if ledger is None:
        return set()
    return {
        str(c.get("canonical_id"))
        for c in (ledger.get("clusters_payload") or {}).get("clusters") or []
        if c.get("canonical_id")
    }


def _add_display_spellings(payload: Dict[str, Any], candidates: Any, evidence: Any) -> None:
    """Each person cluster's shown spelling and its evidence (RFC-131 S6a; ADR-166 decision 1).

    Publisher text first: the member that the most episodes' publisher text names
    (``guards.EpisodeEvidence.publisher_named``), then episode count, then the shorter name, then
    the name. With no member named by publisher text, the cluster's canonical (frequency).
    Measured on the truth set's held-out shows: 9 of 33 spellings wrong against frequency's 13
    (eval-data ``s6/RESULT-2026-10-31.md``).

    **The publisher's own text is the reference spelling** (operator, 2026-11-01; record
    ``s6e_publisher_spelling``): where an episode holding a member states the person in its title,
    description or feed, that spelling is shown, voted by episodes (:func:`_publisher_spellings`).
    The rule above applies only to a group no publisher text names.

    A one-word name never outranks a member's full name (m0017: a name reduced to one word is never
    a repair): publisher text picks among the full names, and with none named there, the most
    frequent full name (0041, eval-data ``d2i9/spelling_full_name.py``: 5 of 475 groups move, the
    truth set's scores unchanged).

    **A tie is not chosen** (operator, 2026-11-02; record ``s6g_tie_spelling_not_chosen``): when no
    publisher text names the group and the two most frequent spellings of its pool (the full names,
    else every member; folded) have equal episode counts, no spelling is set and the evidence is
    ``{"rule": "tie"}``. Each episode keeps its own spelling; the card shows the canonical's name.

    **No majority, no choice** (record ``s6h_publisher_ties_and_pluralities``): when the most
    frequent spelling holds at most half of the pool's counted episodes, the evidence is
    ``{"rule": "plurality"}``; when the publisher's own text names the group two ways equally often,
    ``{"rule": "publisher_tie"}``. Neither sets a spelling, as a tie.
    """
    from collections import Counter

    named = Counter(pid for ev in evidence.values() for pid in ev.publisher_named)
    stated = _publisher_spellings(payload, evidence)

    def full(i: str) -> bool:
        return len(candidates[i].name.split()) >= 2

    for cluster in payload.get("clusters") or []:
        canonical = str(cluster.get("canonical_id") or "")
        if not canonical.startswith("person:"):
            continue
        members = [m["id"] for m in cluster.get("members") or [] if m.get("id") in candidates]
        if canonical in stated:
            spelling, episodes = stated[canonical]
            if spelling is None:
                cluster.pop("display_name", None)
                cluster["display_evidence"] = {"rule": "publisher_tie"}
                continue
            cluster["display_name"] = spelling
            cluster["display_evidence"] = {"rule": "publisher", "episodes": episodes}
            continue
        fulls = [i for i in members if full(i)]
        pool = fulls or members
        by_pub = [i for i in pool if named[i]]
        if by_pub:
            pick = max(
                by_pub,
                key=lambda i: (
                    named[i],
                    candidates[i].freq,
                    -len(candidates[i].name),
                    candidates[i].name,
                ),
            )
            cluster["display_name"] = candidates[pick].name
            cluster["display_evidence"] = {"rule": "publisher", "episodes": named[pick]}
        elif (undecided := _undecided(pool, candidates)) is not None:
            cluster.pop("display_name", None)
            cluster["display_evidence"] = {"rule": undecided}
        elif fulls and canonical in candidates and not full(canonical):
            pick = max(
                fulls,
                key=lambda i: (candidates[i].freq, len(candidates[i].name), candidates[i].name),
            )
            cluster["display_name"] = candidates[pick].name
            cluster["display_evidence"] = {"rule": "frequency", "full_name": True}
        else:
            cluster["display_name"] = cluster.get("canonical_name")
            cluster["display_evidence"] = {"rule": "frequency"}


def _undecided(pool: List[str], candidates: Any) -> Optional[str]:
    """Why ``pool``'s spellings choose none, or None when a majority decides (records s6g, s6h).

    A spelling's count is the sum of its members' episode counts, spellings folded, so an episode
    holding two spellings counts for both. ``"tie"``: the two highest counts are equal.
    ``"plurality"``: the highest is at most half the sum, which is no majority."""
    from collections import Counter

    per: Counter = Counter()
    for i in pool:
        per[_fold(candidates[i].name)] += candidates[i].freq
    top = per.most_common(2)
    if len(top) == 2 and top[0][1] == top[1][1]:
        return "tie"
    if top and top[0][1] * 2 <= sum(per.values()):
        return "plurality"
    return None


def _pair(pairs: Any, a: str, b: str) -> bool:
    """``(a, b)`` is a V2 pair, either way round (``publisher_text.fold``)."""
    if not pairs:
        return False
    from podcast_scraper.identity.publisher_text import fold

    fa, fb = fold(a), fold(b)
    return (fa, fb) in pairs or (fb, fa) in pairs


def _fold(name: str) -> str:
    """Case, spacing and accents folded: how two spellings of one name are compared."""
    import unicodedata

    text = unicodedata.normalize("NFKD", " ".join(str(name).split()).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _full_name(name: str) -> bool:
    """Two or more words once leading titles are set aside ("Professor X" is one name)."""
    from podcast_scraper.speaker_detectors.hosts import HONORIFIC_TITLES

    toks = str(name).split()
    while toks and toks[0].lower().strip(".,") in HONORIFIC_TITLES:
        toks = toks[1:]
    return len(toks) >= 2


def _publisher_spellings(
    payload: Dict[str, Any], evidence: Any
) -> Dict[str, Tuple[Optional[str], int]]:
    """``{canonical: (the publisher's spelling, episodes)}`` for each person group publisher text
    names (record ``s6e_publisher_spelling``, prototype eval-data ``d2i9/spelling_publisher_text``).

    Per episode holding a member, a publisher name counts when it is the name of one of the
    group's Person nodes IN THAT EPISODE (folded), or a full name the roster's own predicates match
    to one; never when it is another person's name there. A name only ``detected_guests`` states
    counts on an exact match only. Two names that fit, neither exactly, give that episode no vote.
    The spelling is the publisher's own text; a bare one never beats a full one. When the top two
    spellings have equal votes the spelling is None (record s6h): the text does not decide.
    """
    from collections import Counter

    from podcast_scraper.providers.ml.diarization import roster as R

    group_of: Dict[str, str] = {}
    member_names: Dict[str, set] = {}
    for cluster in payload.get("clusters") or []:
        canonical = str(cluster.get("canonical_id") or "")
        if not canonical.startswith("person:"):
            continue
        for m in cluster.get("members") or []:
            if m.get("id"):
                group_of[str(m["id"])] = canonical
                member_names.setdefault(canonical, set()).add(str(m.get("name") or ""))
    votes: Dict[str, Counter] = {}
    forms: Dict[str, Counter] = {}
    for ev in evidence.values():
        if not ev.publisher_names:
            continue
        by_group: Dict[str, set] = {}
        for pid, name in ev.person_names.items():
            if pid in group_of:
                by_group.setdefault(group_of[pid], set()).add(name)
        ner = {_fold(n) for n in ev.ner_only}
        for canonical, here in by_group.items():
            others = {
                _fold(n) for pid, n in ev.person_names.items() if group_of.get(pid) != canonical
            }
            hits: List[Tuple[str, bool]] = []
            for p in ev.publisher_names:
                fp = _fold(p)
                if fp in others:
                    continue
                exact = any(_fold(n) == fp for n in here)
                fuzzy = (
                    fp not in ner
                    and _full_name(p)
                    and any(
                        _full_name(n)
                        and (
                            R._same_person(p, n)
                            or R._same_person_on_one_episode(p, n)
                            # V2: the publisher's text spells this person so (empty unless the
                            # evidence mode is publisher_verified)
                            or _pair(ev.publisher_pairs, p, n)
                        )
                        for n in here
                    )
                )
                if exact or fuzzy:
                    hits.append((p, exact))
            if any(e for _p, e in hits):
                hits = [h for h in hits if h[1]]
            elif len({_fold(p) for p, _e in hits}) > 1:
                continue  # two names fit: possibly two people; no vote
            if hits:
                p = hits[0][0]
                votes.setdefault(canonical, Counter())[_fold(p)] += 1
                forms.setdefault(canonical, Counter())[p] += 1
    out: Dict[str, Tuple[Optional[str], int]] = {}
    for canonical, v in votes.items():
        has_full = any(_full_name(n) for n in member_names.get(canonical, ()))
        cand = {fp: k for fp, k in v.items() if _full_name(fp) or not has_full}
        if not cand:
            continue
        top = sorted(cand.values(), reverse=True)
        if len(top) > 1 and top[0] == top[1]:
            out[canonical] = (None, top[0])  # a tie in the publisher's own text: none (s6h)
            continue
        best = max(cand.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]
        spelling = max(
            (f for f in forms[canonical] if _fold(f) == best),
            key=lambda f: (forms[canonical][f], f),
        )
        out[canonical] = (spelling, cand[best])
    return out


def ledger_id_map(ledger: Dict[str, Any]) -> Dict[str, str]:
    """The ``variant -> canonical`` map a ledger records."""
    from podcast_scraper.kg.entity_clusters import id_map_from_clusters_payload

    return id_map_from_clusters_payload(ledger.get("clusters_payload") or {})


def read_ledger(root: Path | str) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(ledger, state)``. The ledger is None unless the state is ok or ok_previous_rule."""
    path = Path(root) / LEDGER_NAME
    try:
        with open(path, encoding="utf-8") as fh:
            ledger = json.load(fh)
    except FileNotFoundError:
        return None, STATE_MISSING
    except (OSError, ValueError):
        return None, STATE_UNREADABLE
    if not isinstance(ledger, dict):
        return None, STATE_UNREADABLE
    rule = ledger.get("rule_version")
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION or rule not in (
        RULE_VERSION,
        PREVIOUS_RULE_VERSION,
    ):
        return None, STATE_VERSION_MISMATCH
    if not isinstance(ledger.get("clusters_payload"), dict):
        return None, STATE_UNREADABLE
    return ledger, STATE_OK if rule == RULE_VERSION else STATE_OK_PREVIOUS_RULE


def served_id_map(root: Path | str) -> Dict[str, str]:
    """The identity map the server applies: the ledger's, or no joins (ADR-166)."""
    ledger, state = read_ledger(root)
    _observe_state(state)
    if ledger is None:
        logger.warning(
            "identity ledger %s at %s: serving no cross-episode joins (ADR-166)",
            state,
            Path(root) / LEDGER_NAME,
        )
        return {}
    return ledger_id_map(ledger)


def ledger_status(root: Path | str) -> Dict[str, Any]:
    """State, versions and staleness of a corpus's ledger, for reports and checks."""
    ledger, state = read_ledger(root)
    out: Dict[str, Any] = {"state": state}
    if ledger is None:
        return out
    current = corpus_token(root)
    built = ledger.get("corpus_token") or {}
    out.update(
        {
            "built_at": ledger.get("built_at"),
            "rule_version": ledger.get("rule_version"),
            "rebuild_needed": state == STATE_OK_PREVIOUS_RULE,
            "code_version": ledger.get("code_version"),
            "corpus_token_at_build": built,
            "corpus_token_now": current,
            "corpus_newer_than_ledger": (current.get("corpus_mtime") or 0)
            > (built.get("corpus_mtime") or 0),
            "kg_episodes_since_build": (current.get("newest_kg_count") or 0)
            - (built.get("newest_kg_count") or 0),
        }
    )
    return out


def write_ledger(
    root: Path | str, ledger: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Write ``ledger`` atomically and append what changed to the change log. Returns the change.

    The ledger it replaces, when valid, is kept as ``PREVIOUS_NAME`` first (the undo).

    Atomic: a temp file in the corpus root, fsync, then ``os.replace``, so a reader never sees a
    half-written ledger (it would read as unreadable, which means no joins). When run as root the
    files take the corpus root's owner, as every corpus writer must.
    """
    root = Path(root)
    previous, _state = read_ledger(root)
    change = _change_entry(previous, ledger)
    change.update(extra or {})
    if previous is not None:
        _write_atomic(root, root / PREVIOUS_NAME, previous)
    _write_atomic(root, root / LEDGER_NAME, ledger)
    _append_change(root, change)
    return change


def refresh_ledger(root: Path | str, *, after: str, log: Optional[logging.Logger] = None) -> bool:
    """Build and write the ledger as the last stage of a corpus-writing path (plan V3, D1).

    Called after a successful ``enrich`` run (the post-pipeline chain queues one after every
    ingest and reprocess batch), an ``upgrade run`` that applied a migration, and an
    ``enrich-edges`` pass that rewrote artifacts. True when the ledger was written. On failure the
    previous ledger stays as it was (joins as of the last build, never none) and the error is
    logged; the caller reports it in its exit code, so it is never silent.

    Before the write, the episode files take the build's group spellings (``identity.respell``,
    RFC-131 0041); the counts go into the change log under ``respell``.
    """
    log = log or logger
    try:
        built = build_ledger(root)
        # Respell the files to this build's spellings BEFORE the write: the write moves
        # corpus_mtime, which invalidates every cache, so no cache holds old files under the new
        # token (RFC-131 0041). A respelling that fails or is deferred never blocks the ledger.
        respell = respell_for(root, built, log)
        change = write_ledger(root, built, extra={"respell": respell})
    except Exception:  # noqa: BLE001 - any build or write failure leaves the previous ledger
        log.exception("identity ledger refresh after %s failed; the previous ledger stays", after)
        return False
    log.info(
        "identity ledger refreshed after %s: %d joins (%d added, %d removed, %d moved)",
        after,
        change["joins"],
        change["joins_added"],
        change["joins_removed"],
        change["canonical_changed"],
    )
    return True


def respell_for(root: Path | str, built: Dict[str, Any], log: logging.Logger) -> Dict[str, Any]:
    """The ledger step's respelling (``identity.respell``); its failure is counted, not raised."""
    from podcast_scraper.identity.respell import respell_files

    try:
        return respell_files(root, built, log=log)
    except Exception as exc:  # noqa: BLE001 - the files lag; read time keeps the surfaces right
        log.exception("ledger respell failed; the files keep their spelling until the next build")
        return {"failed": f"{type(exc).__name__}: {exc}"}


def restore_previous(root: Path | str) -> Dict[str, Any]:
    """Put the previous valid ledger back (the undo of the last write). Returns the change.

    Refuses (``FileNotFoundError`` / ``ValueError``) when there is no valid previous ledger, rather
    than leave the corpus with none: no ledger means no cross-episode joins.
    """
    root = Path(root)
    path = root / PREVIOUS_NAME
    with open(path, encoding="utf-8") as fh:
        previous = json.load(fh)
    if (
        not isinstance(previous, dict)
        or previous.get("schema_version") != LEDGER_SCHEMA_VERSION
        or not isinstance(previous.get("clusters_payload"), dict)
    ):
        raise ValueError(f"{path} is not a valid identity ledger; nothing restored")
    current, _state = read_ledger(root)
    change = _change_entry(current, previous)
    change["restored_from"] = PREVIOUS_NAME
    _write_atomic(root, root / LEDGER_NAME, previous)
    _append_change(root, change)
    return change


def _write_atomic(root: Path, target: Path, payload: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".identity_ledger.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        _match_corpus_owner(root, Path(tmp))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append_change(root: Path, change: Dict[str, Any]) -> None:
    changes = root / CHANGES_NAME
    with open(changes, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(change, ensure_ascii=False, sort_keys=True) + "\n")
    _match_corpus_owner(root, changes)


def _change_entry(previous: Optional[Dict[str, Any]], ledger: Dict[str, Any]) -> Dict[str, Any]:
    """What this build changes against the previous valid ledger: joins added, removed, moved."""
    before = ledger_id_map(previous) if previous else {}
    after = ledger_id_map(ledger)
    added = sorted(k for k in after if k not in before)
    removed = sorted(k for k in before if k not in after)
    moved = sorted(k for k in after if k in before and after[k] != before[k])

    def listed(keys: List[str], src: Dict[str, str], dst: Dict[str, str]) -> List[Any]:
        return [[k, src.get(k), dst.get(k)] for k in keys[:_MAX_LISTED_CHANGES]]

    return {
        "built_at": ledger.get("built_at"),
        "code_version": ledger.get("code_version"),
        "rule_version": ledger.get("rule_version"),
        "corpus_token": ledger.get("corpus_token"),
        "previous_built_at": previous.get("built_at") if previous else None,
        "joins": len(after),
        "joins_added": len(added),
        "joins_removed": len(removed),
        "canonical_changed": len(moved),
        "added": listed(added, before, after),
        "removed": listed(removed, before, after),
        "changed": listed(moved, before, after),
    }


def _match_corpus_owner(root: Path, path: Path) -> None:
    """Give ``path`` the corpus root's owner when running as root; a no-op otherwise."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    st = root.stat()
    os.chown(path, st.st_uid, st.st_gid)


_GAUGE: Dict[str, Any] = {"done": False, "gauge": None}


def _observe_state(state: str) -> None:
    """Set ``podcast_identity_ledger_state{state}`` (1 for the current state), when enabled."""
    if os.environ.get("PODCAST_METRICS_ENABLED", "").strip().lower() not in {"1", "true", "yes"}:
        return
    if not _GAUGE["done"]:
        _GAUGE["done"] = True
        try:
            from prometheus_client import Gauge
        except ImportError:
            return
        _GAUGE["gauge"] = Gauge(
            "podcast_identity_ledger_state",
            "Identity ledger state the server read (1 = current); ok and ok_previous_rule serve "
            "joins (the second until a rebuild), any other state means no joins.",
            ["state"],
        )
    gauge = _GAUGE["gauge"]
    if gauge is None:
        return
    for s in LEDGER_STATES:
        gauge.labels(state=s).set(1 if s == state else 0)
