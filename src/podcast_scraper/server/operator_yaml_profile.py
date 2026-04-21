"""Split / merge top-level ``profile:`` for viewer operator YAML.

Mirrors SPA ``operatorYamlProfile.ts``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def format_profile_scalar(name: str) -> str:
    """Quote a scalar for a one-line ``profile:`` value when needed (safe YAML subset)."""
    s = name.strip()
    if not s:
        return '""'
    if re.match(r"^[A-Za-z0-9_.-]+$", s):
        return s
    return json.dumps(s)


def split_operator_yaml_profile(content: str) -> tuple[str, str]:
    """Return ``(profile_name, body_without_profile_line)``."""
    raw = content.replace("\r\n", "\n")
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^\s*profile:\s*(.+?)\s*$", line)
        if not m:
            continue
        v = m.group(1).strip()
        h = v.find("#")
        if h >= 0:
            v = v[:h].strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        rest = "\n".join(lines[:i] + lines[i + 1 :])
        body = rest.lstrip("\n").rstrip("\n")
        return v, body
    return "", raw.rstrip("\n")


def _body_has_operator_substance(body: str) -> bool:
    """True when ``body`` has any non-blank, non-comment line."""
    for line in body.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return True
    return False


def merge_operator_yaml_profile(profile: str, body: str) -> str:
    """Emit YAML with optional first-line ``profile:`` then ``body``."""
    b = body.lstrip("\n").rstrip("\n")
    if not profile.strip():
        return f"{b}\n" if b else ""
    p = format_profile_scalar(profile.strip())
    if not b:
        return f"profile: {p}\n"
    return f"profile: {p}\n{b}\n"


def expand_profile_only_with_packaged_example(
    content: str,
    *,
    example_path: Path | None,
) -> str:
    """If ``content`` has no keys beyond ``profile:``/comments, append example body."""
    if example_path is None or not example_path.is_file():
        return content
    profile, body = split_operator_yaml_profile(content)
    if _body_has_operator_substance(body):
        return content
    example_text = example_path.read_text(encoding="utf-8", errors="replace")
    _, ex_body = split_operator_yaml_profile(example_text)
    if not _body_has_operator_substance(ex_body):
        return content
    ex_body = ex_body.strip() + "\n"
    if profile.strip():
        return f"profile: {format_profile_scalar(profile)}\n{ex_body}"
    return ex_body
