"""Profile-governance audit: which fields diverge across profiles WITHOUT the registry knowing?

Three incidents in two days, one class: a Config field hand-authored in some profile YAMLs (or
leaking its Config default into the rest) diverged silently and changed behaviour nobody chose —
``gi_value_gate_provider`` (a direct-Anthropic call mid-pipeline), ``llm_pipeline_mode``
(mega_bundled truncating the first real DGX episode, #1878), ``deepseek_summary_model`` (the
Config default 'deepseek-chat' leaking into an RFC-106 emergency tier). Each was found only when
it misbehaved in prod.

This tool makes the class visible and RATCHETED:

* every top-level key hand-authored in any profile YAML that is also a Config field is classified
  as GOVERNED (in REGISTRY_GOVERNED_FIELDS — the drift test already owns it), UNGOVERNED-UNIFORM
  (resolves identically across every profile — includes profiles where the key is absent, so a
  default-leak that happens to match is still uniform), or UNGOVERNED-DIVERGENT — the danger
  class: profiles behave differently and no registry preset declares why.
* ``--report`` prints the full classification with per-profile values for every divergent field —
  the "what's different, what's the same" view.
* ``--check`` (CI mode, wired into a test) fails if any UNGOVERNED-DIVERGENT field is missing
  from the accepted-divergence baseline, or if a baseline entry went stale (no longer divergent /
  now governed). Hand-tuning stays legal; UNDECLARED hand-tuning does not. To accept a new
  divergence, add it to ``config/profile-governance-accepted.yaml`` with a reason —
  which is exactly the review moment the last three incidents never had.

Values are compared on the RESOLVED Config (validators, nested sugar, name-derived fields), not
raw YAML — raw comparison cannot see a default leak, and the default leak is the sneakiest shape.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO / "config" / "profiles"
BASELINE_PATH = REPO / "config" / "profile-governance-accepted.yaml"

for _k in (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPGRAM_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "GROK_API_KEY",
    "MISTRAL_API_KEY",
    "LITELLM_API_KEY",
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
):
    os.environ.setdefault(_k, "dummy-for-validation")


def _profiles() -> list[str]:
    return sorted(
        Path(p).stem
        for p in glob.glob(str(PROFILES_DIR / "*.yaml"))
        if not p.endswith(".example.yaml") and not Path(p).name.startswith(".")
    )


def collect():
    from podcast_scraper.config import Config
    from podcast_scraper.providers.ml.model_registry import REGISTRY_GOVERNED_FIELDS

    governed = set(REGISTRY_GOVERNED_FIELDS)
    config_fields = set(Config.model_fields)

    # The audit universe: keys a HUMAN wrote in at least one profile. Fields nobody hand-authors
    # anywhere are uniform by construction and would only add noise.
    authored: dict[str, set[str]] = {}
    for name in _profiles():
        raw = yaml.safe_load((PROFILES_DIR / f"{name}.yaml").read_text()) or {}
        for key in raw:
            if key in config_fields and key != "profile":
                authored.setdefault(key, set()).add(name)

    resolved: dict[str, object] = {}
    skipped: list[str] = []
    for name in _profiles():
        try:
            resolved[name] = Config.model_validate(
                {"rss_url": "https://governance-audit.example/f.xml", "profile": name}
            )
        except Exception as exc:  # noqa: BLE001 — a profile that cannot build is its own finding
            skipped.append(f"{name} ({type(exc).__name__})")

    rows = []
    for field, authors in sorted(authored.items()):
        values = {}
        for name, cfg in resolved.items():
            v = getattr(cfg, field, None)
            values.setdefault(repr(v), []).append(name)
        if field in governed:
            klass = "GOVERNED"
        elif len(values) <= 1:
            klass = "UNGOVERNED-UNIFORM"
        else:
            klass = "UNGOVERNED-DIVERGENT"
        rows.append((field, klass, values, sorted(authors)))
    return rows, skipped


def load_baseline() -> dict[str, str]:
    if not BASELINE_PATH.exists():
        return {}
    doc = yaml.safe_load(BASELINE_PATH.read_text()) or {}
    return dict(doc.get("accepted", {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="full classification report")
    ap.add_argument(
        "--check", action="store_true", help="ratchet mode: fail on undeclared divergence"
    )
    args = ap.parse_args()

    rows, skipped = collect()
    divergent = [r for r in rows if r[1] == "UNGOVERNED-DIVERGENT"]

    if args.report or not args.check:
        counts = {}
        for _, klass, _, _ in rows:
            counts[klass] = counts.get(klass, 0) + 1
        print(f"profiles: {len(_profiles())}  authored Config fields: {len(rows)}  {counts}")
        if skipped:
            print(f"unresolvable profiles (excluded from value comparison): {skipped}")
        for field, klass, values, authors in rows:
            if klass != "UNGOVERNED-DIVERGENT" and not args.report:
                continue
            print(f"\n{field}  [{klass}]  authored-in: {len(authors)} profile(s)")
            if klass == "UNGOVERNED-DIVERGENT":
                for val, names in sorted(values.items(), key=lambda kv: -len(kv[1])):
                    shown = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
                    print(f"    {val}  <- {len(names)}: {shown}")

    if args.check:
        baseline = load_baseline()
        new = [f for f, *_ in divergent if f not in baseline]
        divergent_names = {f for f, *_ in divergent}
        stale = [f for f in baseline if f not in divergent_names]
        if new:
            print(
                "\nGOVERNANCE RATCHET FAIL — ungoverned fields diverge across profiles and no one "
                "has signed the divergence:",
                file=sys.stderr,
            )
            for f in new:
                print(f"  {f}", file=sys.stderr)
            print(
                "Either govern the field (add to ProfilePreset + REGISTRY_GOVERNED_FIELDS + "
                "materialize) or declare it with a reason in "
                f"{BASELINE_PATH.relative_to(REPO)}.",
                file=sys.stderr,
            )
            return 1
        if stale:
            print(
                f"\nGOVERNANCE RATCHET FAIL — stale baseline entries (no longer divergent or now "
                f"governed), prune them so the file stays honest: {stale}",
                file=sys.stderr,
            )
            return 1
        print(
            f"\ngovernance ratchet OK: {len(divergent)} accepted divergence(s), "
            f"0 undeclared, 0 stale"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
