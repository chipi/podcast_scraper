"""The step-8 reviewer's verdicts on a list migration's entries, as an input the migration reads.

A list migration (0029, 0031, 0032) applies names someone decided by reading. The corpus change
method's step 8 has a fresh reviewer read the entries again; a verdict is only worth something if
the migration acts on it, the same way for every list migration, with nothing per entry in code
(operator, 2026-10-18: "no manual hacks as part of the gates").

The file: ``upgrade/verdicts/<migration id>.tsv``, shipped with the package. ``#``-comment header
lines carry ``key: value`` (``migration``, ``reviewed_commit``, ``list_sha256``, ``reviewer``,
``date``), then one tab-separated row per entry under the column header ``COLUMNS``. ``verdict`` is
one of ``VERDICTS``:

- ``CORRECT``: the listed name is the voice's;
- ``CORRECT_MISSPELLED``: the right person, spelled differently from the episode's own text (a class
  the operator keeps for later; the entry is applied as listed);
- ``WRONG_PERSON``: the listed name is someone else's; the migration must not leave it on the voice.

Every problem refuses the whole file with ``VerdictError`` and says which: an unknown verdict, a
missing header or column, a duplicate key, another migration's file, a row whose name is not the
list's for that voice (stale), a row for no list entry. A missing file is an error for a migration
that declares one; a file with no rows is valid and decides nothing.

BASIS COLUMNS (0034's revision, operator 2026-10-26: keep the name unless the text shows someone
else). A file may add ``basis``, ``segment_idx`` and ``line_sha256`` after ``evidence``
(``COLUMNS_WITH_BASIS``); a migration that reads with ``require_basis`` refuses one without them.
``basis`` names why, from the closed set its verdict allows (``BASES``); ``segment_idx`` is the
deciding segment and ``line_sha256`` the sha256 of its text, so a public file proves its evidence
without carrying transcript text. Absence of evidence is never a removal: a voice nothing decides
is ``CORRECT`` with basis ``undecidable``, the one basis that has no deciding line.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

VERDICTS = ("CORRECT", "CORRECT_MISSPELLED", "WRONG_PERSON")
COLUMNS = ("episode_id", "meta", "voice", "name", "verdict", "evidence")
COLUMNS_WITH_BASIS = COLUMNS + ("basis", "segment_idx", "line_sha256")
UNDECIDABLE = "undecidable"
_KEEP_BASES = ("self_intro", "addressed", "host_read", "continuity", "gold", "clip_own_person")
BASES = {
    "CORRECT": _KEEP_BASES + (UNDECIDABLE,),
    "CORRECT_MISSPELLED": _KEEP_BASES + (UNDECIDABLE,),
    "WRONG_PERSON": (
        "another_person",
        "ad_or_promo_not_own",
        "clip_other_person",
        "not_a_person",
        "merged_voices",
    ),
}
HEADER_KEYS = ("migration", "reviewed_commit", "list_sha256", "reviewer", "date")
VERDICTS_DIR = Path(__file__).resolve().parent / "verdicts"

Key = Tuple[str, str]


class VerdictError(ValueError):
    """The verdicts file cannot be used as it is; the message says why."""


@dataclass(frozen=True)
class Verdict:
    """One reviewed entry."""

    name: str
    verdict: str
    evidence: str
    meta: str
    basis: str = ""
    segment_idx: Optional[int] = None
    line_sha256: str = ""


@dataclass
class Verdicts:
    """A validated verdicts file."""

    migration_id: str
    header: Dict[str, str]
    rows: Dict[Key, Verdict] = field(default_factory=dict)
    sha256: str = ""

    def get(self, episode_id: str, voice: str) -> Optional[Verdict]:
        """The reviewer's verdict on this episode's voice, or ``None`` when it has none."""
        return self.rows.get((episode_id, voice))


def list_sha256(entries: Mapping[Key, object]) -> str:
    """A stable fingerprint of a migration's list, so a verdicts file names the list it read."""
    canon = json.dumps(
        sorted([list(k), list(v) if isinstance(v, tuple) else v] for k, v in entries.items()),
        ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def verdicts_path(migration_id: str, directory: Optional[Path] = None) -> Path:
    """The verdicts file of a migration: ``<id>.tsv`` in ``directory``, by default the package's."""
    return (directory or VERDICTS_DIR) / f"{migration_id}.tsv"


def parse(
    text: str,
    migration_id: str,
    names: Mapping[Key, str],
    listed: Optional[Mapping[Key, object]] = None,
    require_basis: bool = False,
    bases: Optional[Mapping[str, Sequence[str]]] = None,
) -> Verdicts:
    """Validate ``text`` against the migration's list (``{(episode_id, voice): listed name}``).

    ``listed``, when given, is the migration's WHOLE frozen list (every field of every entry:
    roles, both names, the evidence). The header's ``list_sha256`` must then be that list's, so
    an entry edited after the review (a role flipped, an old name changed) fails the load, not
    only a judged name. Without it the hash covers ``names``, as before.

    ``require_basis`` refuses a file without the basis columns (``COLUMNS_WITH_BASIS``).
    ``bases`` replaces ``BASES`` for this load only (a migration whose verdicts mean something
    else, e.g. 0035's "a rename not added for lack of evidence").
    """
    header: Dict[str, str] = {}
    body = []
    for line in text.splitlines():
        if line.startswith("#"):
            hkey, _, value = line[1:].partition(":")
            header[hkey.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    missing = [k for k in HEADER_KEYS if not header.get(k)]
    if missing:
        raise VerdictError(f"header lacks {', '.join(missing)}")
    if header["migration"] != migration_id:
        raise VerdictError(f"file is for {header['migration']!r}, not {migration_id!r}")
    if header["list_sha256"] != list_sha256(names if listed is None else listed):
        raise VerdictError(
            "list_sha256 is not the current list's: the list changed since it was reviewed"
        )
    if not body:
        raise VerdictError("no column header row")
    reader = csv.DictReader(io.StringIO("\n".join(body)), delimiter="\t")
    cols = tuple(reader.fieldnames or ())
    if cols not in (COLUMNS, COLUMNS_WITH_BASIS) or (require_basis and cols != COLUMNS_WITH_BASIS):
        expected = COLUMNS_WITH_BASIS if require_basis else COLUMNS
        raise VerdictError(f"columns are {cols}, expected {expected}")
    with_basis = cols == COLUMNS_WITH_BASIS
    out = Verdicts(migration_id=migration_id, header=header)
    for n, row in enumerate(reader, start=2):
        key = (row["episode_id"].strip(), row["voice"].strip())
        verdict = row["verdict"].strip()
        if verdict not in VERDICTS:
            raise VerdictError(
                f"row {n}: unknown verdict {verdict!r} (allowed: {', '.join(VERDICTS)})"
            )
        if key in out.rows:
            raise VerdictError(f"row {n}: duplicate key {key}")
        if key not in names:
            raise VerdictError(f"row {n}: {key} is no entry of {migration_id}'s list")
        if row["name"].strip() != names[key]:
            raise VerdictError(
                f"row {n}: stale: {key} reads {row['name'].strip()!r}, the list says {names[key]!r}"
            )
        basis, idx, line = "", None, ""
        if with_basis:
            basis, idx, line = _basis(n, verdict, row, BASES if bases is None else bases)
        out.rows[key] = Verdict(
            name=row["name"].strip(),
            verdict=verdict,
            evidence=(row["evidence"] or "").strip(),
            meta=(row["meta"] or "").strip(),
            basis=basis,
            segment_idx=idx,
            line_sha256=line,
        )
    out.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def _basis(
    n: int, verdict: str, row: Mapping[str, Optional[str]], bases: Mapping[str, Sequence[str]]
) -> Tuple[str, Optional[int], str]:
    """A row's ``(basis, segment_idx, line_sha256)``, or ``VerdictError`` naming what is wrong."""
    basis = (row.get("basis") or "").strip()
    raw_idx = (row.get("segment_idx") or "").strip()
    line = (row.get("line_sha256") or "").strip()
    if not basis:
        raise VerdictError(f"row {n}: no basis")
    allowed = tuple(bases.get(verdict, ()))
    if basis not in allowed:
        raise VerdictError(
            f"row {n}: basis {basis!r} is not one {verdict} allows ({', '.join(allowed)})"
        )
    if basis == UNDECIDABLE:
        # Nothing decided it, so no deciding line; a line given anyway must still be well formed.
        if not raw_idx and not line:
            return basis, None, ""
    if not raw_idx or not line:
        raise VerdictError(f"row {n}: basis {basis!r} needs a segment_idx and a line_sha256")
    if not raw_idx.isdigit():
        raise VerdictError(f"row {n}: segment_idx {raw_idx!r} is not a segment index")
    if len(line) != 64 or any(c not in "0123456789abcdef" for c in line):
        raise VerdictError(f"row {n}: line_sha256 {line!r} is not a sha256")
    return basis, int(raw_idx), line


def line_sha256(text: str) -> str:
    """The ``line_sha256`` of a deciding line: its text, stripped, as UTF-8."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load(
    migration_id: str,
    names: Mapping[Key, str],
    directory: Optional[Path] = None,
    listed: Optional[Mapping[Key, object]] = None,
    require_basis: bool = False,
    bases: Optional[Mapping[str, Sequence[str]]] = None,
) -> Verdicts:
    """The migration's verdicts file, validated; ``VerdictError`` when missing or unusable."""
    path = verdicts_path(migration_id, directory)
    if not path.is_file():
        raise VerdictError(
            f"no verdicts file {path.name}: this migration reads one (method step 8)"
        )
    return parse(
        path.read_text(encoding="utf-8"), migration_id, names, listed, require_basis, bases
    )
