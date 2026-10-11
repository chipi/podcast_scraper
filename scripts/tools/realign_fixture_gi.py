#!/usr/bin/env python3
"""Re-ground the app validation corpus's insights in what is said (dry run by default). ADR-165.

The committed GI files carried the builder's defects, measured 2026-10-11 over 55 episodes:

* quotes handed to insights ROUND-ROBIN, so 75 of 162 pipeline insights had only a neighbour's
  quote and 12 had none;
* 37 sponsor reads and 37 greetings as quotes;
* host questions, greetings, thanks and sponsor reads as "insights" (90 of 162);
* authored quotes stamped at 0 ms, 7 of them paraphrases rather than transcript text.

The builder is fixed (``build_gi`` grounds each insight in the segment that says it; insights are
the first lines that are claims; ``authored_quote_anchor`` anchors claims). A rebuild does not
reproduce the committed corpus (later edits live only there), so this applies the SAME rules to it
in place, per episode:

1. pipeline insights that are not claims are removed (questions in any language; greetings, thanks
   and sponsor reads by the English rules, applied to a non-English line through its translation);
2. the episode is filled back to its count with the next lines that are claims, as the builder now
   picks them;
3. authored quotes are anchored (times; a paraphrase gets its verbatim line);
4. every pipeline insight is SUPPORTED_BY the one segment that contains its sentence (an existing
   quote node with that text is reused, else one is made from the segment);
5. quotes no insight is supported by any more are removed, with their edges; edges are written in
   a stable order, so a second pass changes nothing.

A non-English episode's insights are in its source language, so they are grounded in the
source-language segments (``<stem>.<lang>.segments.json``); the translation is read only to say
what an opening line means.

    .venv/bin/python scripts/tools/realign_fixture_gi.py            # report
    .venv/bin/python scripts/tools/realign_fixture_gi.py --apply    # write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_app_validation_corpus import (  # noqa: E402
    _insight_density_envelope,
    _insight_sentiment_envelope,
    authored_quote_anchor,
    has_failure_mode,
    is_ad_copy,
    speaker_balanced,
)
from build_synthetic_validation_corpus import (  # noqa: E402
    _route_and_tag,
    about_topic_ids,
    is_greeting_or_filler,
    short_hash,
)

#: Speaker labels that are not people (an ad read is not anyone's claim); the builder's list.
_NOT_A_PERSON = {"ad", "ads", "sponsor", "announcer", "narrator", "promo"}
_NAME = re.compile(r"\b[A-ZÀ-Þ][\w'-]{2,}")
#: Insight types, spread by position as the builder spreads them.
_INSIGHT_TYPES = ("claim", "observation", "recommendation")


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[\w']+", s.lower()))


def is_not_a_claim(text: str, english: Optional[str] = None) -> bool:
    """A host question, an opening/closing pleasantry or a sponsor read — not a takeaway.

    The greeting and sponsor rules are English. For a non-English line, ``english`` is its
    translation when one is confidently paired; a question mark is a question in any language.
    """
    t = text.strip()
    if t.endswith("?") or "? " in t:
        return True
    probe = (english or t).strip()
    return probe.endswith("?") or is_greeting_or_filler(probe) or is_ad_copy(probe) or is_ad_copy(t)


def trusted_prefix(segments: list[dict[str, Any]], turns: dict[str, list[str]]) -> int:
    """How many opening source segments pair with the translation turn of the same index.

    The source file has a few segments the translation does not, so index pairing drifts. It holds
    from the start until the first pair that shares no proper name (the guest's, the host's, the
    sponsor's); from there on it is not trusted. One shared name further in proves nothing: a
    talking-point brand ("RockShox") recurs across many lines and paired a Spanish line about
    braking with an English one about risk (2026-10-11).
    """
    k = 0
    while k < len(segments) and paired_english(k, str(segments[k].get("text") or ""), turns):
        k += 1
    return k


def paired_english(index: int, source_text: str, turns: dict[str, list[str]]) -> Optional[str]:
    """The English of the translation turn at ``index`` when a proper name both texts share
    confirms the pair (see :func:`trusted_prefix` for where pairing is trusted at all)."""
    english = " ".join(turns.get(f"t{index:04d}", []))
    if not english:
        return None
    # Capitalised words in both: names and brands survive translation ("Beatriz", "Strava"); the
    # sentence-opening words of two different languages ("Este", "This") practically never match.
    return english if set(_NAME.findall(source_text)) & set(_NAME.findall(english)) else None


class _Episode:
    """One episode's GI and the transcript its insights are written against."""

    def __init__(
        self,
        gi: dict[str, Any],
        segments: list[dict[str, Any]],
        transcript: str,
        turns: Optional[dict[str, list[str]]],
    ) -> None:
        self.gi = gi
        self.segments = segments
        self.transcript = transcript
        self.turns = turns
        self.trusted = trusted_prefix(segments, turns) if turns else 0
        self.stats = dict.fromkeys(
            (
                "insights_removed",
                "insights_added",
                "authored",
                "regrounded",
                "unsaid",
                "quotes_removed",
            ),
            0,
        )

    # -- helpers --------------------------------------------------------------------------------
    @property
    def nodes(self) -> list[dict[str, Any]]:
        return self.gi["nodes"]

    @property
    def edges(self) -> list[dict[str, Any]]:
        return self.gi["edges"]

    def by_id(self) -> dict[str, dict[str, Any]]:
        return {n["id"]: n for n in self.nodes}

    def pipeline(self) -> list[dict[str, Any]]:
        return [
            n
            for n in self.nodes
            if n.get("type") == "Insight" and not n["id"].startswith("insight:authored-")
        ]

    def segment_index(self, text: str) -> Optional[int]:
        want = _norm(text.rstrip("…"))
        return next(
            (
                i
                for i, s in enumerate(self.segments)
                if want and want in _norm(str(s.get("text") or ""))
            ),
            None,
        )

    def english_at(self, index: Optional[int]) -> Optional[str]:
        if not self.turns or index is None or index >= self.trusted:
            return None
        return paired_english(index, str(self.segments[index].get("text") or ""), self.turns)

    def speaker_of(self, segment: dict[str, Any]) -> Optional[str]:
        """The person id of a segment's speaker.

        First the direct evidence: a Person node named as the segment's speaker label. Only for a
        label with no such node, the person the corpus's SPOKEN_BY edges give a quote of that
        label's lines — edges, because person canonicalisation rewrote them, while a quote's own
        ``speaker_id`` can still hold the raw placeholder (``person:speaker-01``). The edges come
        second because the committed corpus also credited people who never speak (2026-10-11).
        """
        label = str(segment.get("speaker_label") or "")
        persons = {
            str(n["properties"].get("name") or ""): n["id"]
            for n in self.nodes
            if n.get("type") == "Person"
        }
        if label in persons:
            return persons[label]
        spoken_by = {e["from"]: e["to"] for e in self.edges if e.get("type") == "SPOKEN_BY"}
        ids = set(persons.values())
        for n in self.nodes:
            pid = spoken_by.get(n["id"])
            if n.get("type") != "Quote" or pid not in ids:
                continue
            idx = self.segment_index(str(n["properties"].get("text") or ""))
            if idx is not None and str(self.segments[idx].get("speaker_label") or "") == label:
                return str(pid)
        return None

    def remove(self, ids: set[str]) -> None:
        self.gi["nodes"] = [n for n in self.nodes if n["id"] not in ids]
        self.gi["edges"] = [
            e for e in self.edges if e.get("from") not in ids and e.get("to") not in ids
        ]

    # -- the steps ------------------------------------------------------------------------------
    def claim_at(self, index: Optional[int], text: str) -> bool:
        """Whether a line is a claim: said by a person, and not a question, pleasantry, segue,
        prompt or sponsor read (``is_not_a_claim``, through the paired translation when trusted)."""
        if index is not None:
            label = str(self.segments[index].get("speaker_label") or "").strip().lower()
            if label in _NOT_A_PERSON:
                return False
        return not is_not_a_claim(text, self.english_at(index))

    def drop_non_claims(self) -> int:
        """Step 1. Returns how many were removed (the fill restores the count)."""
        drop = {
            n["id"]
            for n in self.pipeline()
            if not self.claim_at(
                self.segment_index(str(n["properties"].get("text") or "")),
                str(n["properties"].get("text") or ""),
            )
        }
        self.remove(drop)
        self.stats["insights_removed"] = len(drop)
        return len(drop)

    def fill_with_claims(self, wanted: int) -> None:
        """Step 2: the next lines that are claims, as the fixed builder picks them — each speaker
        not yet quoted by the remaining insights first (``speaker_balanced``)."""
        episode = next((n["id"] for n in self.nodes if n.get("type") == "Episode"), None)
        topics = [
            (n["id"], str(n["properties"].get("label") or ""))
            for n in self.nodes
            if n.get("type") == "Topic"
        ]
        kept = self.pipeline()
        taken = {_norm(str(n["properties"].get("text") or "").rstrip("…")) for n in kept}
        covered = frozenset(
            str(self.segments[i].get("speaker_label") or "")
            for i in (self.segment_index(str(n["properties"].get("text") or "")) for n in kept)
            if i is not None
        )
        candidates: list[tuple[int, str, Optional[str]]] = []  # (segment index, text, english)
        for idx, seg in enumerate(self.segments):
            txt = str(seg.get("text") or "").strip()
            text = txt if len(txt) <= 200 else txt[:197].rstrip() + "…"
            english = self.english_at(idx)
            # Compared as STORED (long insights are truncated with "…"): the full line looked new
            # next to its own truncated insight, and minted a duplicate id.
            if len(txt) < 40 or _norm(text.rstrip("…")) in taken or not self.claim_at(idx, txt):
                continue
            candidates.append((idx, text, english))
        chosen = speaker_balanced(
            [
                (text, str(self.segments[idx].get("speaker_label") or ""))
                for idx, text, _ in candidates
            ],
            wanted,
            covered,
            topical=[bool(about_topic_ids(en or text, topics)) for _idx, text, en in candidates],
        )
        base = len(kept)
        for k, c in enumerate(chosen):
            _idx, text, english = candidates[c]
            pos = base + k
            iid = f"insight:{short_hash(text)}"
            self.nodes.append(
                {
                    "id": iid,
                    "type": "Insight",
                    "properties": {
                        "text": text,
                        "episode_id": self.gi.get("episode_id"),
                        "grounded": True,
                        "insight_type": _INSIGHT_TYPES[pos % len(_INSIGHT_TYPES)],
                        "position_hint": round(0.2 + 0.15 * (pos % 5), 3),
                        **_route_and_tag(text),
                    },
                }
            )
            if episode:
                self.edges.append({"type": "HAS_INSIGHT", "from": episode, "to": iid})
            for tid in about_topic_ids(english or text, topics):
                self.edges.append({"type": "ABOUT", "from": iid, "to": tid})
            self.stats["insights_added"] += 1

    def anchor_authored(self) -> None:
        """Step 3, before grounding: a pipeline insight saying the same line reuses the quote."""
        by_id = self.by_id()
        authored = {n["id"] for n in self.nodes if n["id"].startswith("insight:authored-")}
        for e in self.edges:
            if e.get("type") != "SUPPORTED_BY" or e.get("from") not in authored:
                continue
            q = by_id.get(e["to"])
            if q is None or q["properties"].get("timestamp_start_ms"):
                continue
            pid = q["properties"].get("speaker_id")
            name = str((by_id.get(pid) or {}).get("properties", {}).get("name") or "") or None
            anchor = authored_quote_anchor(
                str(q["properties"].get("text") or ""), self.segments, self.transcript, name
            )
            if anchor:
                q["properties"].update(anchor)
                self.stats["authored"] += 1

    def quote_for(self, seg: dict[str, Any], insight_id: str) -> str:
        """An existing quote node saying this segment, or a new one made from it."""
        qtext = str(seg.get("text") or "").strip()
        for n in self.nodes:
            if n.get("type") == "Quote" and _norm(str(n["properties"].get("text") or "")) == _norm(
                qtext
            ):
                return n["id"]
        digest = hashlib.sha256(f"{self.gi.get('episode_id')}:{insight_id}:{qtext}".encode())
        qid = f"quote:{digest.hexdigest()[:16]}"
        at = self.transcript.find(qtext)
        props: dict[str, Any] = {
            "text": qtext,
            "episode_id": self.gi.get("episode_id"),
            "char_start": max(at, 0),
            "char_end": max(at, 0) + len(qtext),
            "timestamp_start_ms": int(round(float(seg.get("start") or 0.0) * 1000)),
            "timestamp_end_ms": int(round(float(seg.get("end") or 0.0) * 1000)),
            "transcript_ref": next(
                (
                    q["properties"]["transcript_ref"]
                    for q in self.nodes
                    if q.get("type") == "Quote" and q["properties"].get("transcript_ref")
                ),
                None,
            ),
        }
        pid = self.speaker_of(seg)
        if pid:
            props["speaker_id"] = pid
            self.edges.append({"type": "SPOKEN_BY", "from": qid, "to": pid})
        self.nodes.append({"id": qid, "type": "Quote", "properties": props})
        return qid

    def ground_pipeline(self) -> None:
        """Step 4: each pipeline insight SUPPORTED_BY the segment that says it."""
        grounded: dict[str, str] = {}
        for n in self.pipeline():
            idx = self.segment_index(str(n["properties"].get("text") or ""))
            if idx is None:
                self.stats["unsaid"] += 1
                continue
            grounded[n["id"]] = self.quote_for(self.segments[idx], n["id"])
            self.stats["regrounded"] += 1
        pipeline_ids = {n["id"] for n in self.pipeline()}
        self.gi["edges"] = [
            e
            for e in self.edges
            if not (e.get("type") == "SUPPORTED_BY" and e.get("from") in pipeline_ids)
        ] + [{"type": "SUPPORTED_BY", "from": iid, "to": qid} for iid, qid in grounded.items()]

    def drop_orphan_quotes(self) -> None:
        """Step 5, a stable edge order, and node ids unique (three p06 GIs listed a topic twice)."""
        seen: set[str] = set()
        self.gi["nodes"] = [n for n in self.nodes if not (n["id"] in seen or seen.add(n["id"]))]
        used = {e["to"] for e in self.edges if e.get("type") == "SUPPORTED_BY"}
        orphan = {n["id"] for n in self.nodes if n.get("type") == "Quote" and n["id"] not in used}
        self.remove(orphan)
        self.stats["quotes_removed"] = len(orphan)
        unique = {(str(e.get("type")), str(e.get("from")), str(e.get("to"))): e for e in self.edges}
        self.gi["edges"] = [unique[k] for k in sorted(unique)]

    def vary_grounding(self, low_grounding: bool) -> None:
        """The builder's rule, re-applied to the episode's insights as they now are: every pipeline
        insight grounded, and in a ``low_grounding_dialogue`` episode every second one not — so the
        grounding_rate enricher still has variation to see (authored claims keep their own)."""
        for i, n in enumerate(self.pipeline()):
            n["properties"]["grounded"] = not (low_grounding and i % 2 == 1)

    def realign(self, low_grounding: bool = False) -> dict[str, int]:
        removed = self.drop_non_claims()
        self.fill_with_claims(removed)
        self.vary_grounding(low_grounding)
        self.anchor_authored()
        self.ground_pipeline()
        self.drop_orphan_quotes()
        return self.stats


def realign(
    gi: dict[str, Any],
    segments: list[dict[str, Any]],
    transcript: str,
    turns: Optional[dict[str, list[str]]] = None,
    low_grounding: bool = False,
) -> dict[str, int]:
    """Re-ground one episode's GI in place (the module docstring's five steps)."""
    return _Episode(gi, segments, transcript, turns).realign(low_grounding)


def _episode_inputs(
    gi_path: Path,
) -> tuple[list[dict[str, Any]], str, Optional[dict[str, list[str]]]]:
    """The segments and transcript an episode's insights are written in, and its translation."""
    tdir = gi_path.parent.parent / "transcripts"
    stem = gi_path.name[: -len(".gi.json")]
    translation = tdir / f"{stem}.translation.json"
    if not translation.is_file():
        segments = json.loads((tdir / f"{stem}.segments.json").read_text("utf-8"))
        return segments, (tdir / f"{stem}.txt").read_text(encoding="utf-8"), None
    tr = json.loads(translation.read_text("utf-8"))
    lang = str(tr.get("source_language") or "")
    turns: dict[str, list[str]] = {}
    for u in tr.get("units") or []:
        turns.setdefault(str(u.get("turn_id")), []).extend(
            str(x.get("en_text") or "") for x in u.get("sentences") or []
        )
    segments = json.loads((tdir / f"{stem}.{lang}.segments.json").read_text("utf-8"))
    return segments, (tdir / f"{stem}.{lang}.txt").read_text(encoding="utf-8"), turns


def rederive_episode_enrichments(gi_path: Path, gi: dict[str, Any]) -> list[Path]:
    """The per-episode envelopes built from the GI, rebuilt with the builder's own functions so
    they agree with the re-grounded insights (``insight_density``, ``insight_sentiment``)."""
    meta_dir = gi_path.parent
    stem = gi_path.name[: -len(".gi.json")]
    meta = json.loads((meta_dir / f"{stem}.metadata.json").read_text("utf-8"))
    raw = (meta.get("episode") or {}).get("duration_seconds")
    duration = int(raw) if isinstance(raw, (int, float)) else 1800
    episode_id = str(gi.get("episode_id") or "")
    out = meta_dir / "enrichments"
    written = []
    for name, envelope in (
        ("insight_density", _insight_density_envelope(gi, episode_id, duration)),
        ("insight_sentiment", _insight_sentiment_envelope(gi, episode_id)),
    ):
        path = out / f"{stem}.{name}.json"
        text = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
        if not path.is_file() or path.read_text("utf-8") != text:
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--corpus", type=Path, default=Path("tests/fixtures/app-validation-corpus/v3"))
    p.add_argument("--apply", action="store_true", help="write the changes (default: report only)")
    args = p.parse_args()
    version = Path("tests/fixtures/FIXTURES_VERSION").read_text(encoding="utf-8").strip()
    gt_dir = Path("tests/fixtures/ground-truth") / version / "ground_truth"
    total: dict[str, int] = {}
    changed = rederived = 0
    for gi_path in sorted((args.corpus / "feeds").rglob("*.gi.json")):
        gi = json.loads(gi_path.read_text(encoding="utf-8"))
        before = json.dumps(gi, sort_keys=True)
        segments, transcript, turns = _episode_inputs(gi_path)
        low = has_failure_mode(gi_path.name[: -len(".gi.json")], gt_dir, "low_grounding_dialogue")
        for k, v in realign(gi, segments, transcript, turns, low).items():
            total[k] = total.get(k, 0) + v
        if json.dumps(gi, sort_keys=True) != before:
            changed += 1
            if args.apply:
                gi_path.write_text(
                    json.dumps(gi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        if args.apply:
            rederived += len(rederive_episode_enrichments(gi_path, gi))
    verb = "changed" if args.apply else "would change"
    print(f"{verb} {changed} GI files: {json.dumps(total)}")
    if args.apply:
        print(f"rewrote {rederived} per-episode enrichment envelopes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
