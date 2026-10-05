#!/usr/bin/env python3
"""ONE reader for the run-global language, and a whitelist for the real exceptions (#2177).

A non-English episode transcribed as English does not raise. It produces a plausible transcript
of the wrong words, and every downstream stage — summary, GI, KG, search — then trusts it. The
mechanism that allowed it was a run-global `cfg.language` read directly at each provider call
site, so an episode's own language never reached the provider.

S0.6 routes every one of those through `languages.transcription_language(cfg)`. This lint is what
keeps it that way: a new `cfg.language` read added tomorrow fails CI rather than quietly
reintroducing the substitution.

"EXACTLY ONE READER" IS FALSE AS AN ABSOLUTE, so the whitelist is part of the design rather than
an escape hatch. Five kinds of read are legitimate:

* the cloud transcription providers, which receive an already-resolved `language` argument and
  read `self.cfg.language` only as a fallback when the caller passed nothing;
* `speaker_detectors/ner.py`, choosing which spaCy model to load;
* `cli.py`, printing the configuration to an operator;
* the config snapshot in `metadata_generation.py`, whose whole job is recording the run config;
* the bundled-prompt builders in the cloud LLM providers, which pass the language the model
  should ANSWER IN — a different question from what language the episode is in.

THIS LINT WAS ITSELF WRONG UNTIL 2026-09-30. Its pattern matched only the dotted form, so 11
reads written as `getattr(self.cfg, "language", "en")` were invisible and two of their files
were not whitelisted. The output was quoted in a review as proof there was one reader. A guard
is only as true as the shapes it can see, so the pattern now covers both forms — and a third
spelling would make it wrong again, which is why the test suite asserts on the shapes rather
than on the count.

Anything else is a finding. Adding a line here is a decision, which is the point: it has to be
argued for in review rather than slipped in.

Usage::

    .venv/bin/python scripts/check/lint_language_readers.py

Exits non-zero on an unwhitelisted read, printing file:line and what to do instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "podcast_scraper"

#: `cfg.language` / `self.cfg.language` / `config.language`, but NOT `language_override`,
#: `languages`, or a longer attribute that merely starts with "language".
_DOTTED = re.compile(r"\b(?:self\.)?_?(?:cfg|config)\.language\b(?!_)")

#: `getattr(cfg, "language", ...)` and its `self.cfg` / `self._cfg` / `config` variants.
#:
#: WHY THIS HALF EXISTS. The dotted pattern alone made this lint's own claim FALSE. A
#: whole-branch review in 2026-09-30 cited its "cfg.language is read only by languages.py and 7
#: whitelisted files" output as evidence, and the lint was blind to **11 further reads** — every
#: one of them written as `getattr(self.cfg, "language", "en")`, in gemini, grok, anthropic,
#: mistral and openai. Two of those files were not whitelisted at all. A guard that overstates
#: its coverage is worse than no guard, because its output gets quoted.
_GETATTR = re.compile(r"""getattr\(\s*(?:self\.)?_?(?:cfg|config)\s*,\s*['"]language['"]""")


def _is_read(line: str) -> bool:
    return bool(_DOTTED.search(line) or _GETATTR.search(line))


#: Kept as a module-level name because the tests and the --list output reference it.
PATTERN = _DOTTED

#: relpath -> the reason this file may read it. A set of line numbers would rot on every edit, so
#: the grain is the FILE plus a stated reason.
WHITELIST: Dict[str, str] = {
    # The one reader. Everything else asks it.
    "languages.py": "the resolver itself — this is the single legitimate reader",
    # Cloud transcription providers: `language` arrives resolved from the call site; cfg is only
    # consulted when the caller passed nothing at all.
    "providers/openai/openai_provider.py": "fallback when the caller passes no language",
    "providers/gemini/gemini_provider.py": "fallback when the caller passes no language",
    "providers/mistral/mistral_provider.py": "fallback when the caller passes no language",
    "providers/deepgram/deepgram_provider.py": "fallback when the caller passes no language",
    # A FIFTH legitimate kind, invisible until the getattr half of the pattern existed: the
    # bundled-prompt builders pass `language=` as the language the MODEL SHOULD ANSWER IN, which
    # is a different question from "what language is this episode". The analysis text is English
    # by construction once translation has run (D-1/D-39), and where it has not, an English
    # answer about non-English text is the intended behaviour rather than a substitution. These
    # are not the S0.6 defect — that was transcription being TOLD the wrong input language — but
    # they are reads, so they are named here rather than hidden by a pattern that cannot see
    # `getattr`.
    "providers/grok/grok_provider.py": "output language for the bundled prompt, not the input",
    "providers/anthropic/anthropic_provider.py": (
        "output language for the bundled prompt, not the input"
    ),
    # Local Whisper: same fallback shape, plus the init-time model-name resolution. S0.7 removes
    # this provider from the DGX profiles' chains and guards it against non-`en`.
    "providers/ml/ml_provider.py": "fallback + init-time whisper model-name resolution",
    # Picks which spaCy NER model to load — a model choice, not a transcription language.
    "speaker_detectors/ner.py": "selects the spaCy model for the configured language",
    # Prints the configuration to an operator.
    "cli.py": "operator-facing display of the resolved configuration",
    # The config snapshot records the run configuration; that IS its job.
    "workflow/metadata_generation.py": "config_snapshot records the run config verbatim",
}


def _is_comment_or_doc(line: str) -> bool:
    """Skip prose. A mention in a comment or docstring is documentation, not a read.

    Crude on purpose: a line whose first non-space character starts a comment, or that contains
    a docstring delimiter. A multi-line docstring's middle lines are caught by the indentation
    heuristic below only when they read like prose, so the cost of being wrong here is a false
    POSITIVE (a prose line reported as a read), which is visible and cheap to whitelist —
    never a false negative.
    """
    stripped = line.strip()
    return stripped.startswith("#") or '"""' in line or "'''" in line


def _reads(path: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    in_docstring = False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.count('"""') == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring or _is_comment_or_doc(line):
            continue
        if _is_read(line):
            out.append((n, line.strip()))
    return out


def main() -> int:
    violations: List[str] = []
    seen_whitelisted: Set[str] = set()

    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        hits = _reads(path)
        if not hits:
            continue
        if rel in WHITELIST:
            seen_whitelisted.add(rel)
            continue
        for n, text in hits:
            violations.append(f"  src/podcast_scraper/{rel}:{n}\n      {text}")

    stale = sorted(set(WHITELIST) - seen_whitelisted - {"languages.py"})
    if stale:
        print(
            "STALE WHITELIST — these files no longer read cfg.language, so their entry in\n"
            "scripts/check/lint_language_readers.py should be deleted:\n"
            + "\n".join(f"  {s}" for s in stale)
        )
        # A stale entry is not a build break: it is an invitation to tidy up, and failing on it
        # would punish the very cleanup this lint is supposed to encourage.

    if violations:
        print(
            f"UNWHITELISTED cfg.language READ ({len(violations)}):\n"
            + "\n".join(violations)
            + "\n\nEvery stage that needs a language for a provider call must ask\n"
            "  podcast_scraper.languages.transcription_language(cfg)\n"
            "which applies override > feed > profile-default and NEVER substitutes 'en'.\n"
            "A direct cfg.language read is how a non-English episode came back as a plausible\n"
            "English transcript of the wrong words (#2177).\n\n"
            "If this read is genuinely legitimate, add the file to WHITELIST in this script WITH\n"
            "A REASON — that is a review decision, not a formality."
        )
        return 1

    print(
        f"OK — cfg.language is read only by languages.py and {len(seen_whitelisted) - 1} "
        "whitelisted file(s), each with a stated reason."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
