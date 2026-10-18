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
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

VERDICTS = ("CORRECT", "CORRECT_MISSPELLED", "WRONG_PERSON")
COLUMNS = ("episode_id", "meta", "voice", "name", "verdict", "evidence")
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


@dataclass
class Verdicts:
    """A validated verdicts file."""

    migration_id: str
    header: Dict[str, str]
    rows: Dict[Key, Verdict] = field(default_factory=dict)
    sha256: str = ""

    def get(self, episode_id: str, voice: str) -> Optional[Verdict]:
        return self.rows.get((episode_id, voice))


def list_sha256(entries: Mapping[Key, object]) -> str:
    """A stable fingerprint of a migration's list, so a verdicts file names the list it read."""
    canon = json.dumps(
        sorted([list(k), list(v) if isinstance(v, tuple) else v] for k, v in entries.items()),
        ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def verdicts_path(migration_id: str, directory: Optional[Path] = None) -> Path:
    return (directory or VERDICTS_DIR) / f"{migration_id}.tsv"


def parse(text: str, migration_id: str, names: Mapping[Key, str]) -> Verdicts:
    """Validate ``text`` against the migration's list (``{(episode_id, voice): listed name}``)."""
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
    if header["list_sha256"] != list_sha256(names):
        raise VerdictError(
            "list_sha256 is not the current list's: the list changed since it was reviewed"
        )
    if not body:
        raise VerdictError("no column header row")
    reader = csv.DictReader(io.StringIO("\n".join(body)), delimiter="\t")
    cols = tuple(reader.fieldnames or ())
    if cols != COLUMNS:
        raise VerdictError(f"columns are {cols}, expected {COLUMNS}")
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
        out.rows[key] = Verdict(
            name=row["name"].strip(),
            verdict=verdict,
            evidence=(row["evidence"] or "").strip(),
            meta=(row["meta"] or "").strip(),
        )
    out.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def load(migration_id: str, names: Mapping[Key, str], directory: Optional[Path] = None) -> Verdicts:
    """The migration's verdicts file, validated; ``VerdictError`` when missing or unusable."""
    path = verdicts_path(migration_id, directory)
    if not path.is_file():
        raise VerdictError(
            f"no verdicts file {path.name}: this migration reads one (method step 8)"
        )
    return parse(path.read_text(encoding="utf-8"), migration_id, names)
