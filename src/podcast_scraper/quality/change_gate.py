"""The corpus change gate: which changes go through the method in full, and whether they did.

``docs/guides/CORPUS_CHANGE_METHOD.md`` is the method. On 2026-10-18 a migration (0031, the repair
of 0029) was built straight from the task, skipping the method's reading list, prediction and
acceptance steps, although the method had been written for exactly that kind of change the day
before. Memory did not make it happen, so this module makes it mechanical. Three places call it:

- a Claude Code hook (``scripts/hooks/corpus_change_gate_hook.py``) refuses an edit to content code
  until the change's record has its pre-build steps (``check_build_ready``);
- the git pre-commit hook runs the same check on staged files;
- a unit test (CI) refuses a registered migration without a complete record (``check_landing``).

WHICH CHANGES. The operator's rule: the full method is for changes that can alter what a listener
or reader is shown about what was said and who said it. A migration's class is decided by what it
WRITES (``classify_written``) and by whether any gating scoreboard number moves, never by a label
alone: ``scripts/measure/corpus_change_gate.py`` measures both on a clone and the landing check
refuses a record that says ``light`` when the measurement says ``full``.

A RECORD is ``<records_dir>/<id>.md``: a front matter block (``id``, ``class``, ``status``,
``issue``, ``touches``) and one ``## N. <step>`` section per method step. A section still holding
the template's ``_TODO_`` is not done. A step that was skipped is written ``MISSED:`` with what was
skipped, and counts only once the record carries ``Operator accepted:`` with a date — the operator
decides what a miss costs, not the builder. Stdlib only: the Claude hook runs on the system Python.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CONFIG_PATH = Path("config") / "corpus_change_gate.json"
TODO = "_TODO_"
#: Steps whose sections must be done before any content code is written (method §4).
PRE_BUILD_STEPS = ("1", "2", "3", "4", "5")
#: Steps whose sections must be done before a migration lands on main.
LANDING_STEPS = PRE_BUILD_STEPS + ("6", "7", "8")
_DATED = re.compile(r"^Dated:\s*\d{4}-\d{2}-\d{2}", re.MULTILINE)
_DATED_AT = re.compile(
    r"^Dated:\s*(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?)", re.MULTILINE
)
_INTEGER = re.compile(r"\b\d+\b")
_ACCEPTED = re.compile(r"^Operator accepted:\s*\d{4}-\d{2}-\d{2}", re.MULTILINE)
_SECTION = re.compile(r"^## (\d+)\.[^\n]*\n", re.MULTILINE)
_VERDICT = re.compile(r"^Verdict:\s*(GO WITH CONDITIONS|GO|NO-GO)\b", re.MULTILINE)


def load_config(repo_root: Path) -> Dict[str, Any]:
    """The gate's configuration from ``config/corpus_change_gate.json``."""
    data: Dict[str, Any] = json.loads((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    return data


def is_content_code(rel_path: str, config: Dict[str, Any]) -> bool:
    """True when ``rel_path`` (repo-relative, ``/``-separated) is guarded content code."""
    return any(fnmatch.fnmatch(rel_path, pat) for pat in config["content_code"])


@dataclass
class Record:
    """One change record, parsed."""

    path: Path
    meta: Dict[str, Any]
    sections: Dict[str, str] = field(default_factory=dict)
    text: str = ""

    @property
    def id(self) -> str:
        return str(self.meta.get("id") or self.path.stem)

    @property
    def change_class(self) -> str:
        return str(self.meta.get("class") or "")

    @property
    def is_open(self) -> bool:
        return str(self.meta.get("status") or "open") != "landed"

    def touches(self, rel_path: str) -> bool:
        """True when the repo-relative path matches one of the record's ``touches`` patterns."""
        return any(fnmatch.fnmatch(rel_path, pat) for pat in self.meta.get("touches") or [])


def parse_record(path: Path) -> Optional[Record]:
    """A record from its file, or None when it has no front matter."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    meta: Dict[str, Any] = {}
    key: Optional[str] = None
    for line in text[4:end].splitlines():
        if line.lstrip().startswith("#") or not line.strip():
            continue
        if line.startswith("  - ") and key:
            meta.setdefault(key, []).append(line[4:].split(" #")[0].strip())
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        value = v.split(" #")[0].strip()
        meta[key] = value if value else []
    body = text[end + 5 :]
    sections: Dict[str, str] = {}
    marks = list(_SECTION.finditer(body))
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        sections[m.group(1)] = body[m.end() : stop].strip()
    return Record(path=path, meta=meta, sections=sections, text=text)


def records(repo_root: Path, config: Dict[str, Any]) -> List[Record]:
    """Every parseable record under the records directory (the template excluded)."""
    out = []
    for p in sorted((repo_root / config["records_dir"]).glob("*.md")):
        if p.name in ("TEMPLATE.md", "README.md"):
            continue
        rec = parse_record(p)
        if rec is not None:
            out.append(rec)
    return out


def _section_problem(rec: Record, step: str) -> Optional[str]:
    body = rec.sections.get(step)
    if body is None:
        return f"step {step}: section missing"
    if not body or TODO in body:
        return f"step {step}: not done (still {TODO})"
    if body.startswith("MISSED:") and not _ACCEPTED.search(rec.text):
        return f"step {step}: MISSED, and the record has no 'Operator accepted: <date>'"
    if step == "4" and not body.startswith("MISSED:"):
        if not _DATED.search(body):
            return "step 4: the prediction has no 'Dated: YYYY-MM-DD' line"
        if not _INTEGER.search(_DATED.sub("", body)):
            return "step 4: the prediction states no integers"
    return None


def _measured_at(rec: Record) -> Optional[datetime]:
    """When the change was last measured (its gate report), or None before the first measurement."""
    gate = rec.path.with_suffix(".gate.json")
    if not gate.is_file():
        return None
    raw = json.loads(gate.read_text(encoding="utf-8")).get("measured_at")
    if raw:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    return datetime.utcfromtimestamp(gate.stat().st_mtime)


def _revision_problems(rec: Record) -> List[str]:
    """A change already measured is REVISED only after a new prediction AND new acceptance
    criteria: the 0031 repair passed steps 1-5 through accepted misses, so without this a second
    round of code needed neither (2026-10-18). The newest ``Dated:`` in steps 4 and 5 must each be
    later than the gate report."""
    measured = _measured_at(rec)
    if measured is None:
        return []
    out = []
    for step, what in (("4", "prediction"), ("5", "acceptance (by the reviewer)")):
        stamps = [
            datetime.fromisoformat(m.replace(" ", "T"))
            for m in _DATED_AT.findall(rec.sections.get(step, ""))
        ]
        if not stamps or max(stamps) <= measured:
            out.append(
                f"step {step}: the change was measured at "
                f"{measured.isoformat(timespec='minutes')}; "
                f"a revision needs a new {what} dated after it ('Dated: YYYY-MM-DDTHH:MM', UTC)"
            )
    return out


def check_build_ready(rec: Record) -> List[str]:
    """Why content code may NOT yet be written under this record (empty: it may)."""
    if rec.change_class == "light":
        why = rec.sections.get("0", "")
        return [] if why and TODO not in why else ["light: '## 0. Why light' is not filled"]
    if rec.change_class != "full":
        return [f"class must be 'full' or 'light', is {rec.change_class!r}"]
    problems = [p for s in PRE_BUILD_STEPS if (p := _section_problem(rec, s))]
    return problems + _revision_problems(rec)


def check_edit(repo_root: Path, rel_paths: Iterable[str]) -> List[str]:
    """Refusal reasons for writing ``rel_paths`` now; empty when every write is allowed."""
    config = load_config(repo_root)
    recs = [r for r in records(repo_root, config) if r.is_open]
    problems: List[str] = []
    for rel in sorted(set(rel_paths)):
        if not is_content_code(rel, config):
            continue
        mine = [r for r in recs if r.touches(rel)]
        if not mine:
            problems.append(
                f"{rel} is content code and no open change record lists it under 'touches'. "
                f"Start one from {config['records_dir']}/TEMPLATE.md "
                "(docs/guides/CORPUS_CHANGE_METHOD.md §4) before writing code."
            )
            continue
        ready = [r for r in mine if not check_build_ready(r)]
        # A change already MEASURED guards its paths against edits made under another record:
        # each measured record listing the path must be ready for a revision itself (2026-10-18:
        # 0031's file was editable under the hook's record, skipping 0031's own revision rule).
        measured = [r for r in mine if _measured_at(r) is not None]
        blocking = [r for r in measured if check_build_ready(r)]
        if not ready or blocking:
            for r in blocking or mine:
                for why in check_build_ready(r):
                    problems.append(f"{rel}: record {r.path.name}: {why}")
    return problems


def check_landing(repo_root: Path, registered_ids: Iterable[str]) -> List[str]:
    """Refusal reasons for the registered migrations under the regime; empty when all may land."""
    config = load_config(repo_root)
    by_id = {r.id: r for r in records(repo_root, config)}
    problems: List[str] = []
    for mid in sorted(registered_ids):
        if mid[:4] < config["regime_from_migration"]:
            continue
        rec = by_id.get(mid)
        if rec is None:
            problems.append(f"{mid}: no change record {config['records_dir']}/{mid}.md")
            continue
        gate = rec.path.with_suffix(".gate.json")
        if not gate.is_file():
            problems.append(
                f"{mid}: no gate report {gate.name} (scripts/measure/corpus_change_gate.py)"
            )
            continue
        measured = json.loads(gate.read_text(encoding="utf-8")).get("computed_class")
        if rec.change_class == "light" and measured != "light":
            problems.append(f"{mid}: record says light, the gate measured {measured!r}")
        if rec.change_class == "full" or measured == "full":
            problems += [f"{mid}: {p}" for s in LANDING_STEPS if (p := _section_problem(rec, s))]
            verdicts = _VERDICT.findall(rec.sections.get("8", ""))
            if not verdicts or verdicts[-1] == "NO-GO":
                problems.append(
                    f"{mid}: step 8: the last 'Verdict:' line is "
                    f"{verdicts[-1] if verdicts else 'missing'}, not GO / GO WITH CONDITIONS"
                )
            else:
                problems += _outcome_problems(repo_root, mid, rec, gate, verdicts[-1])
        elif not rec.sections.get("0") or TODO in rec.sections.get("0", ""):
            problems.append(f"{mid}: light: '## 0. Why light' is not filled")
    return problems


OUTCOMES_DIR = Path("src") / "podcast_scraper" / "upgrade" / "outcomes"


def _outcome_problems(
    repo_root: Path, mid: str, rec: Record, gate: Path, verdict: str
) -> List[str]:
    """The shipped outcome (``upgrade/outcomes/<id>.json``, read by the migration outcome hook) is
    the measured one: its verdict is the record's last, the record names its sha256, and its
    scoreboards and plan equal the gate report's. Stated limit: an outcome, gate report and record
    edited together pass this; only the step-8 re-derivation guards that."""
    path = repo_root / OUTCOMES_DIR / f"{mid}.json"
    if not path.is_file():
        return [
            f"{mid}: no shipped outcome {OUTCOMES_DIR / path.name} "
            "(scripts/measure/build_outcome.py)"
        ]
    raw = path.read_bytes()
    out = json.loads(raw)
    rep = json.loads(gate.read_text(encoding="utf-8"))
    problems = []
    if out.get("verdict") != verdict:
        problems.append(f"{mid}: the outcome's verdict {out.get('verdict')!r} is not the record's")
    sha = hashlib.sha256(raw).hexdigest()
    if sha not in rec.sections.get("8", ""):
        problems.append(f"{mid}: step 8 does not name the outcome's sha256 {sha}")
    for key in ("scoreboard_before", "scoreboard_after", "plan_sha256", "episode_deltas"):
        if out.get(key) != rep.get(key):
            problems.append(f"{mid}: the outcome's {key} is not the gate report's")
    if out.get("code_dirty") or rep.get("code", {}).get("dirty"):
        problems.append(f"{mid}: the outcome was measured on uncommitted code")
    return problems


def _key_paths(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted key paths whose values differ between two JSON values (lists compared whole)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            out += _key_paths(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else str(k))
        return out
    return [] if a == b else [prefix]


def classify_written(
    changed: Iterable[Tuple[str, Optional[bytes], Optional[bytes]]], config: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """``(class, reasons)`` for a set of written files ``(rel path, before, after)``.

    Full when any written file is a content file, or a keyed file changed a content key.
    """
    reasons: List[str] = []
    for rel, before, after in changed:
        name = rel.split("/")[-1]
        if any(
            fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in config["content_files"]
        ):
            reasons.append(f"content file written: {rel}")
            continue
        for pat, keys in config["content_keys"].items():
            if not fnmatch.fnmatch(name, pat):
                continue
            try:
                a = json.loads(before) if before else {}
                b = json.loads(after) if after else {}
            except ValueError:
                reasons.append(f"unreadable keyed file written: {rel}")
                continue
            hit = [
                k for k in _key_paths(a, b) if any(k == c or k.startswith(c + ".") for c in keys)
            ]
            if hit:
                reasons.append(f"content key changed in {rel}: {', '.join(hit[:3])}")
    return ("full" if reasons else "light"), reasons


def gating_moves(
    before: Dict[str, int], after: Dict[str, int], config: Dict[str, Any]
) -> Dict[str, int]:
    """Gating scoreboard numbers that moved, ``{metric: delta}``."""
    return {
        m: int(after.get(m, 0)) - int(before.get(m, 0))
        for m in config["gating_metrics"]
        if int(after.get(m, 0)) != int(before.get(m, 0))
    }
