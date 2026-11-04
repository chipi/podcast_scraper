"""A path an artifact stores, read relative to the episode's run directory.

Some metadata and GI artifacts store the transcript path as written at ingest
(``/app/output/feeds/<feed>/<run>/transcripts/x.txt``). Joined to the run as-is, an absolute path
ignores the corpus being read and names the file under the ingest host's path, which is absent in a
copy or a restore (121 of 3,130 member episodes on snapshot-prod-20261103). The part after the run
directory's name is the same file in this corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_relative(value: Any, run: Path) -> str:
    """``value`` relative to ``run``: unchanged when it is relative; the part after the LAST
    ``run.name`` component when it is absolute and passes through the run; ``""`` when it is
    absolute and does not, or ends at the run directory itself (it names no file in this run).

    The same rule as ``upgrade.migrations.m0025_one_person_one_entry._run_relative``, which keeps
    its own copy (guarded content code); a test pins the two to agree.
    """
    rel = str(value or "")
    path = Path(rel)
    if not path.is_absolute():
        return rel
    parts = path.parts
    if run.name not in parts:
        return ""
    at = len(parts) - 1 - parts[::-1].index(run.name)
    return str(Path(*parts[at + 1 :])) if at + 1 < len(parts) else ""
