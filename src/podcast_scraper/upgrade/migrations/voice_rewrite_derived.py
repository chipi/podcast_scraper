"""What a voice-by-voice rewrite must carry besides the voices: ad maps and the metadata counts.

Moved here from 0032 (where it was written and measured: 99 ad maps left beside their ads by a
rename, 2026-10-16 snapshot) so every migration built on m0025's rewrite carries them, starting with
0031 (its derived-surface census found 3 ad maps off their transcript, 2026-10-18). 0032 keeps its
own copy until its change record opens it; the two must stay identical until then.

``.adfree.admap.json`` holds offsets into the raw transcript (``source_length``, the pre- and
post-roll cut points, the excised ranges and their texts). A renamed or unnamed voice changes the
speaker prefixes in that text, so each offset moves with the text around it: through the segment
spans of a rendered transcript, through a character diff of one rewritten in place. An ad map that
does not describe the text on disk, or an offset inside a rewritten label, is left alone and counted
(0030 rebuilds those). The metadata's GI and KG counts are recomputed from the rewritten artifacts.
"""

from __future__ import annotations

import bisect
import difflib
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .m0023_transcript_speaker_prefixes_resynced import _align, _load, _rendered, _rows, Refused
from .m0029_naming_arc_voices_repaired import _Episode as _VoiceEpisode


class DerivedSurfacesEpisode(_VoiceEpisode):
    """0029's voice-by-voice episode that, after ``plan``, carries the ad map of every transcript
    it re-renders and resyncs the metadata counts."""

    def plan(self) -> bool:
        planned = super().plan()
        if planned:
            for path, text in list(self.text_files.items()):
                if path.name.endswith(".txt") and not path.name.endswith(".adfree.txt"):
                    self._carry_admap(path, text)
            self._recount()
        return planned

    def _recount(self) -> None:
        """The metadata's GI and KG counts, from the artifacts as rewritten (0030 does the same).

        The voice rewrite removes and adds KG people; left alone, ``knowledge_graph.node_count``
        and ``edge_count`` kept the old figures (28/30 against a graph of 27/28 on The Curiosity
        Shop's "Resisting Influence", 2026-10-16 snapshot)."""
        meta = self.json_files.get(self.meta) or _load(self.meta)
        if not isinstance(meta, dict):
            return
        before = json.dumps(meta, sort_keys=True)
        stem = str(self.meta)[: -len(".metadata.json")]
        for suffix, block in ((".gi.json", "grounded_insights"), (".kg.json", "knowledge_graph")):
            path = Path(stem + suffix)
            doc = self.json_files.get(path) or _load(path)
            counts = meta.get(block)
            if not isinstance(doc, dict) or not isinstance(counts, dict):
                continue
            nodes = doc.get("nodes") or []
            if block == "grounded_insights" and "insight_count" in counts:
                counts["insight_count"] = sum(1 for n in nodes if n.get("type") == "Insight")
            if block == "knowledge_graph" and doc.get("nodes") is not None:
                counts["node_count"] = len(nodes)
                counts["edge_count"] = len(doc.get("edges") or [])
        if json.dumps(meta, sort_keys=True) != before:
            self.json_files[self.meta] = meta
            self._bump("metadata_counts_synced")

    def _carry_admap(self, path: Path, new_text: str) -> None:
        admap_path = path.with_name(path.name[: -len(".txt")] + ".adfree.admap.json")
        admap = _load(admap_path)
        if not isinstance(admap, dict):
            return
        old_text = path.read_text(encoding="utf-8")
        if admap.get("source_length") != len(old_text) or old_text == new_text:
            self._bump("admaps_left_for_0030" if old_text != new_text else "admaps_unchanged")
            return
        seg_path = path.with_name(path.name[: -len(".txt")] + ".segments.json")
        new_rows = _rows(self.json_files.get(seg_path, _load(seg_path)))
        try:
            old_spans, _labels = _align(old_text, _rendered(_rows(_load(seg_path))))
            new_spans, _labels = _align(new_text, _rendered(new_rows))
            if len(old_spans) != len(new_spans):
                raise Refused("the renders have different segments")
        except Refused:
            opcodes = difflib.SequenceMatcher(
                None, old_text, new_text, autojunk=False
            ).get_opcodes()

            def moved(pos: Any) -> Optional[int]:
                return _move_by_diff(int(pos), opcodes, len(old_text), len(new_text))

        else:

            def moved(pos: Any) -> Optional[int]:
                return _move_offset(int(pos), old_spans, new_spans, len(old_text), len(new_text))

        out = dict(admap)
        out["source_length"] = len(new_text)
        points = [
            (key, moved(admap[key]))
            for key in ("preroll_cut_end", "postroll_cut_start")
            if isinstance(admap.get(key), int)
        ]
        moved_ranges = [(moved(a), moved(b)) for a, b in admap.get("excised_ranges") or []]
        ranges = [[a, b] for a, b in moved_ranges if a is not None and b is not None]
        if any(p is None for _k, p in points) or len(ranges) != len(moved_ranges):
            self._bump("admaps_left_for_0030")
            return
        out.update(points)
        out["excised_ranges"] = ranges
        out["excised_texts"] = [new_text[a:b] for a, b in ranges]
        if isinstance(admap.get("chars_removed"), int):
            old_cut = sum(int(b) - int(a) for a, b in admap.get("excised_ranges") or [])
            new_cut = sum(b - a for a, b in ranges)
            out["chars_removed"] = admap["chars_removed"] + new_cut - old_cut
        if out != admap:
            self.text_files[admap_path] = json.dumps(out, indent=2)
            self._bump("admaps_carried")


def admap_matches(transcript: Path) -> Optional[bool]:
    """Does the transcript's ad map describe the text on disk? ``None`` when it has none."""
    admap = _load(transcript.with_name(transcript.name[: -len(".txt")] + ".adfree.admap.json"))
    if not isinstance(admap, dict) or not transcript.is_file():
        return None
    text = transcript.read_text(encoding="utf-8")
    ranges = admap.get("excised_ranges") or []
    texts = admap.get("excised_texts") or []
    return admap.get("source_length") == len(text) and all(
        text[a:b] == t for (a, b), t in zip(ranges, texts)
    )


def _move_by_diff(pos: int, opcodes: List[Any], old_len: int, new_len: int) -> Optional[int]:
    """*pos* through a character diff: shifted in an unchanged stretch, kept at the start of a
    changed one, ``None`` inside it."""
    if pos == old_len:
        return new_len
    for tag, i1, i2, j1, _j2 in opcodes:
        if i1 <= pos < i2:
            return j1 + (pos - i1) if tag == "equal" else (j1 if pos == i1 else None)
    return None


def _move_offset(
    pos: int, old: List[Tuple[int, int]], new: List[Tuple[int, int]], old_len: int, new_len: int
) -> Optional[int]:
    """*pos* in the old render -> the same place in the new one, or ``None`` inside a label.

    In a segment's text it keeps its distance from the text's start. Between two texts (``" "`` or
    ``"\\nLabel: "``) the gap's first two characters (the line break, the label's first letter)
    and its end keep their place; anywhere else is inside a speaker label that was rewritten.
    """
    i = bisect.bisect_right([s for s, _e in old], pos) - 1
    if i >= 0 and pos <= old[i][1]:
        return new[i][0] + (pos - old[i][0])
    gap_start, new_gap_start = (old[i][1], new[i][1]) if i >= 0 else (0, 0)
    gap_end = old[i + 1][0] if i + 1 < len(old) else old_len
    new_gap_end = new[i + 1][0] if i + 1 < len(new) else new_len
    if pos == gap_end:
        return new_gap_end
    if pos - gap_start <= 1:
        return min(new_gap_start + (pos - gap_start), new_gap_end)
    return None
