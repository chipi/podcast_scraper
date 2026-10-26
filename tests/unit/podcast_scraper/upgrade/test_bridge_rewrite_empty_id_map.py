"""An empty id map must leave a bridge intact (found 2026-10-26: 0025 left 15 served bridges `{}`).

`rewrite_bridges_m0007.apply_to_payload` is called by m0017's `_Episode._rewrite_bridge`, which
clears the bridge and updates it from the result; m0020 reuses that same `_Episode`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migrations import (
    m0017_speaker_names_canonicalised as m17,
    m0020_titled_person_ids_remerged as m20,
)
from podcast_scraper.upgrade.rewrite_bridges_m0007 import apply_to_payload

pytestmark = [pytest.mark.unit]


def _bridge() -> dict:
    return {
        "episode_id": "e1",
        "identities": [
            {
                "id": "person:ann-lee",
                "display_name": "Ann Lee",
                "aliases": ["A. Lee"],
                "sources": {"kg": True, "gi": False},
            }
        ],
    }


def test_an_empty_mapping_returns_a_deep_copy_not_the_input() -> None:
    payload = _bridge()
    snapshot = json.dumps(payload, sort_keys=True)
    out, changes = apply_to_payload(payload, {})
    assert changes == 0 and out == payload and out is not payload
    out["identities"][0]["aliases"].append("X")
    out["identities"][0]["sources"]["gi"] = True
    out["identities"].append({"id": "person:new"})
    assert json.dumps(payload, sort_keys=True) == snapshot


@pytest.mark.parametrize("names", [{}, {"Old Name": "New Name"}])
def test_m0017_bridge_rewrite_with_an_empty_id_map_keeps_the_identities(
    tmp_path: Path, names: dict
) -> None:
    run = tmp_path / "feeds" / "f" / "run_0"
    (run / "metadata").mkdir(parents=True)
    meta = run / "metadata" / "ep.metadata.json"
    meta.write_text(json.dumps({"episode": {"episode_id": "e1"}, "content": {}}))
    bridge = run / "metadata" / "ep.bridge.json"
    bridge.write_text(json.dumps(_bridge()))
    ep = m17._Episode(meta)
    assert ep.load()
    ep.rewrite(names, {})
    out = ep.files[bridge]
    assert out != {} and out.get("identities")
    assert [i["id"] for i in out["identities"]] == ["person:ann-lee"]
    assert out["identities"][0]["sources"] == {"kg": True, "gi": False}
    assert out["episode_id"] == "e1"


def test_m0020_reaches_the_bridge_rewrite_only_through_m0017s_episode() -> None:
    assert m20._Episode is m17._Episode
    assert m20.plan_maps is m17.plan_maps
