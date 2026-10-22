"""The catch-all log source drops exactly what the app drop-ins ship (``unscoped.alloy``).

Each app's Alloy drop-in keeps a precise list of containers; ``unscoped.alloy`` ships every other
container under ``app=unscoped`` by dropping the union of those lists. When an app adds a container
to its drop-in and not here, that container ships twice; when this list claims one nobody ships, it
ships nowhere. Both fail here. ``orrery.alloy`` lives in the orrery repo, so its part is pinned
below and kept by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

OBS = Path(__file__).resolve().parents[4] / "infra" / "observability"
ORRERY = {"orrery-web", "orrery-mcp", "orrery-lab-api", "orrery-pipeline-runner-.*"}
# websites.alloy lives in chipi/humancopilot.eu; its keep-filter is pinned here and kept by hand.
WEBSITES = {"websites-nginx"}
# evals.alloy lives in chipi/eval-harness; same arrangement.
EVALS = {"evals-nginx", "evals-inspector"}
# docker ps on prod, 2026-10-17
PROD_CONTAINERS = {
    "alloy": False,
    "audio-refs": False,
    "cadvisor": False,
    "compose-api-1": True,
    "compose-viewer-1": True,
    "litellm": True,
    "litellm-postgres": True,
    "litellm-spend-push": True,
    "operator-api-1": True,
    "operator-viewer-1": True,
    "orrery-lab-api": True,
    "orrery-mcp": True,
    "orrery-web": True,
    "player-api-1": True,
    "player-board-summarizer-1": True,
    "player-digest-scheduler-1": True,
    "player-learning-app-1": True,
    "player-mcp-1": True,
    "player-obs-1": True,
    "podcast-reprocess-39013530187": True,
    "compose-pipeline-llm-run-0a1b": True,
    "podcast-sweep-20261017": True,
    "orrery-pipeline-runner-7": True,
    "websites-nginx": True,
    "evals-nginx": True,
    "evals-inspector": True,
}


def _block(text: str, header: str) -> str:
    start = text.index(header)
    depth = 0
    for i in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start : i + 1]
    raise AssertionError(f"unclosed {header}")


def _name_regex(relabel_block: str, action: str) -> str:
    for rule in re.findall(r"rule \{(.*?)\n\s*\}", relabel_block, re.S):
        if re.search(rf'action\s*=\s*"{action}"', rule):
            raw = re.search(r'regex\s*=\s*"((?:[^"\\]|\\.)*)"', rule)
            assert raw and "__meta_docker_container_name" in rule
            return str(json.loads(f'"{raw.group(1)}"'))
    raise AssertionError(f"no {action} rule")


def _alternatives(name_regex: str) -> set[str]:
    assert name_regex.startswith("/(") and name_regex.endswith(")")
    inner, parts, depth, cur = name_regex[2:-1], [], 0, ""
    for ch in inner:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return {*parts, cur}


def _app_keep_regexes() -> dict[str, str]:
    out = {}
    for path in sorted(OBS.glob("*.alloy")):
        if path.name == "unscoped.alloy":
            continue
        text = path.read_text()
        for src in re.findall(r'loki\.source\.docker "\w+" \{.*?\n\}', text, re.S):
            relabel = re.search(r"targets\s*=\s*discovery\.relabel\.(\w+)\.output", src)
            assert relabel, f"{path.name}: a docker log source without a discovery.relabel"
            block = _block(text, f'discovery.relabel "{relabel.group(1)}"')
            out[f"{path.name}:{relabel.group(1)}"] = _name_regex(block, "keep")
    return out


def _unscoped_drop() -> str:
    return _name_regex(
        _block((OBS / "unscoped.alloy").read_text(), 'discovery.relabel "unscoped"'), "drop"
    )


def test_every_app_drop_in_is_read() -> None:
    assert set(_app_keep_regexes()) == {
        "litellm.alloy:litellm",
        "operator.alloy:operator_public",
        "operator.alloy:podcast",
        "player.alloy:player",
    }


def test_the_drop_list_is_the_union_of_the_app_keep_lists() -> None:
    claimed = (
        set().union(*map(_alternatives, _app_keep_regexes().values())) | ORRERY | WEBSITES | EVALS
    )
    assert _alternatives(_unscoped_drop()) == claimed


@pytest.mark.parametrize("name,claimed", sorted(PROD_CONTAINERS.items()))
def test_a_container_ships_once(name: str, claimed: bool) -> None:
    keeps = [
        *_app_keep_regexes().values(),
        "/(" + "|".join(sorted(ORRERY | WEBSITES | EVALS)) + ")",
    ]
    by_apps = sum(bool(re.fullmatch(r, f"/{name}")) for r in keeps)
    by_unscoped = not re.fullmatch(_unscoped_drop(), f"/{name}")
    assert by_apps == (1 if claimed else 0)
    assert by_apps + by_unscoped == 1
