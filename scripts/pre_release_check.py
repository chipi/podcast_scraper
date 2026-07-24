#!/usr/bin/env python3
"""Fast pre-flight checks before ``make ci`` for a release (ADR-031).

Validates:
  - ``pyproject.toml`` ``[project].version`` matches ``__version__`` in ``__init__.py``
  - ``docs/releases/RELEASE_vX.Y.Z.md`` exists for that version
  - ``docs/COMPATIBILITY.md`` contains a row for that version (#797)
  - Release notes file and ``docs/releases/index.md`` mention the version

Does **not** run tests or MkDocs; run ``make pre-release`` (Makefile) for the full gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_pyproject_version(root: Path) -> str:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef, import-untyped]

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ver = data.get("project", {}).get("version")
    if not ver or not isinstance(ver, str):
        print("pre_release_check: pyproject.toml missing [project].version", file=sys.stderr)
        sys.exit(1)
    return ver.strip()


def _read_init_version(root: Path) -> str:
    init_path = root / "src" / "podcast_scraper" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        print(f"pre_release_check: could not parse __version__ in {init_path}", file=sys.stderr)
        sys.exit(1)
    return m.group(1).strip()


def _check_release_notes(root: Path, version: str) -> None:
    rel = root / "docs" / "releases" / f"RELEASE_v{version}.md"
    if not rel.is_file():
        print(
            f"pre_release_check: missing {rel.relative_to(root)} — run make release-docs-prep",
            file=sys.stderr,
        )
        sys.exit(1)
    body = rel.read_text(encoding="utf-8")
    if version not in body:
        print(
            f"pre_release_check: {rel.name} should contain version string {version!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_releases_index(root: Path, version: str) -> None:
    idx = root / "docs" / "releases" / "index.md"
    if not idx.is_file():
        print("pre_release_check: missing docs/releases/index.md", file=sys.stderr)
        sys.exit(1)
    text = idx.read_text(encoding="utf-8")
    token = f"RELEASE_v{version}.md"
    if token not in text and version not in text:
        print(
            "pre_release_check: docs/releases/index.md should link to or mention "
            f"{token} or {version!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_compatibility_matrix(root: Path, version: str) -> None:
    compat = root / "docs" / "COMPATIBILITY.md"
    if not compat.is_file():
        print(
            "pre_release_check: missing docs/COMPATIBILITY.md — add a row for this release (#797)",
            file=sys.stderr,
        )
        sys.exit(1)
    body = compat.read_text(encoding="utf-8")
    if version not in body:
        print(
            f"pre_release_check: docs/COMPATIBILITY.md must mention version {version!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def check_perf_report(root: Path, version: str) -> bool:
    """Forced-but-NON-GATING: warn loudly when no perf-trace report mentions
    *version* (perf-traces framework, docs/guides/perf-traces/reports/).

    Official per-release performance results are a required release step but not
    a CI gate — you generate + iterate the numbers until happy. This check makes
    skipping it impossible to miss (a prominent stderr banner) without ever
    failing the build. Returns True when a matching report is found.

    See docs/guides/perf-traces/index.md ("Per release").
    """
    reports_dir = root / "docs" / "guides" / "perf-traces" / "reports"
    found = False
    if reports_dir.is_dir():
        # Reports follow the framework convention ``<date>-<version>.md``; match on
        # the FILENAME (a report may mention other versions in prose, so a content
        # substring is too loose — it would spuriously satisfy a neighbouring bump).
        for md in reports_dir.glob(f"*{version}.md"):
            if md.name == "index.md":
                continue
            found = True
            print(f"pre_release_check: perf report present for {version} ({md.name})")
            break
    if not found:
        print(
            "\n"
            "  ============================================================\n"
            f"  ⚠  NO PERF REPORT for {version}.\n"
            "     Official per-release performance results are REQUIRED\n"
            "     (not a CI gate — generate + iterate until happy):\n"
            "       1. capture: scripts/dev/capture-{search,graph,surface}-*.sh\n"
            "       2. author:  docs/guides/perf-traces/reports/"
            "<date>-<version>.md\n"
            "       3. index:   add a row to reports/index.md\n"
            "     See docs/guides/perf-traces/index.md. NOT skipping the build.\n"
            "  ============================================================\n",
            file=sys.stderr,
        )
    return found


def run_checks(root: Path | None = None) -> str:
    """Run all checks. Returns the resolved version string.

    Raises SystemExit on failure.
    """
    base = root if root is not None else _repo_root()
    pv = _read_pyproject_version(base)
    iv = _read_init_version(base)
    if pv != iv:
        print(
            f"pre_release_check: version mismatch pyproject.toml={pv!r} "
            f"__init__.py={iv!r} — run: make bump VERSION={pv}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # Pre-release builds (``2.7.0.dev0``, ``...b1``, ``...rc1``) are not FINAL releases,
    # so the release-notes / index / compatibility-matrix gates don't apply — they gate
    # a real release. The pyproject↔__init__ match above still runs (drift guard).
    from packaging.version import InvalidVersion, Version

    # Forced-but-non-gating perf report reminder — runs for BOTH pre-release and
    # final versions (we want the nudge on every dev bump), never fails the build.
    check_perf_report(base, pv)
    try:
        is_pre = Version(pv).is_prerelease
    except InvalidVersion:
        is_pre = False
    if is_pre:
        print(
            f"pre_release_check: {pv} is a PRE-RELEASE — skipping release-notes/index/"
            f"compatibility checks (those gate FINAL releases). Bump to a final X.Y.Z "
            f"before cutting a release."
        )
        return pv
    _check_release_notes(base, pv)
    _check_releases_index(base, pv)
    _check_compatibility_matrix(base, pv)
    return pv


def main() -> None:
    pv = run_checks()
    print(f"pre_release_check: OK (version {pv}, release notes present)")


if __name__ == "__main__":
    main()
