#!/usr/bin/env python3
"""Idempotently patch .venv/bin/activate to set PIP_FIND_LINKS when wheels/spacy/*.whl exists."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVATE = ROOT / ".venv" / "bin" / "activate"

DEACTIVATE_HOOK = """    if [ -n "${_PS_SPACY_WHEELS_ACTIVE:-}" ]; then
        unset _PS_SPACY_WHEELS_ACTIVE
        if [ -n "${_OLD_PIP_FIND_LINKS+x}" ]; then
            export PIP_FIND_LINKS="$_OLD_PIP_FIND_LINKS"
            unset _OLD_PIP_FIND_LINKS
        else
            unset PIP_FIND_LINKS
        fi
    fi
"""

ACTIVATE_TAIL = """
# podcast_scraper: use wheels/spacy when present (see wheels/README.md)
_ps_spacy_wheels="${VIRTUAL_ENV}/../wheels/spacy"
if [ -d "${_ps_spacy_wheels}" ] && ls "${_ps_spacy_wheels}"/*.whl >/dev/null 2>&1; then
    if [ -n "${PIP_FIND_LINKS+x}" ]; then
        export _OLD_PIP_FIND_LINKS="${PIP_FIND_LINKS}"
    fi
    export PIP_FIND_LINKS="$(cd "${_ps_spacy_wheels}" && pwd)"
    export _PS_SPACY_WHEELS_ACTIVE=1
fi
unset _ps_spacy_wheels
"""


def main() -> int:
    if not ACTIVATE.is_file():
        print(f"error: {ACTIVATE} not found — create .venv first", file=sys.stderr)
        return 1
    text = ACTIVATE.read_text()
    if "_PS_SPACY_WHEELS_ACTIVE" in text:
        print(f"Already patched: {ACTIVATE}")
        return 0

    needle = "deactivate () {\n"
    pos = text.find(needle)
    if pos == -1:
        print("error: could not find deactivate() in activate script", file=sys.stderr)
        return 1
    pos += len(needle)
    text = text[:pos] + DEACTIVATE_HOOK + text[pos:]
    text = text.rstrip() + ACTIVATE_TAIL + "\n"
    ACTIVATE.write_text(text)
    print(f"Patched: {ACTIVATE} — run: source .venv/bin/activate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
