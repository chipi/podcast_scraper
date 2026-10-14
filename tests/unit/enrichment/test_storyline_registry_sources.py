"""RFC-128 (#2349): where a run's storyline registry comes from, and what a run reports."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers import topic_theme_clusters as ttc
from podcast_scraper.enrichment.enrichers._loaders import load_kg, partition_topic_nodes
from podcast_scraper.enrichment.paths import discover_episode_bundles

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
_CTX = type("Ctx", (), {"run_id": "t", "enricher_id": "topic_theme_clusters"})()


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    shutil.copytree(CORPUS / "feeds", tmp_path / "feeds")
    return tmp_path


def _run(root: Path) -> dict:
    data: dict = ttc._compute(None, root, discover_episode_bundles(root), {}, _CTX)
    art = root / "enrichments" / "topic_theme_clusters.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({"data": data}), encoding="utf-8")
    return data


def _ids(data: dict) -> set[str]:
    return {c["graph_compound_parent_id"] for c in data["clusters"]}


def test_a_normal_run_continues_the_artifacts_registry_and_reports_healthy(root: Path) -> None:
    first = _run(root)
    assert first["registry_health"]["source"] == "proposals", "a corpus's first run"
    second = _run(root)
    health = second["registry_health"]
    assert health["source"] == "artifact" and health["ok"]
    assert health["shared_topics"] == 0 and health["vanished"] == []
    assert _ids(second) == _ids(first)


def test_an_unreadable_artifact_continues_from_the_registry_copy_and_reports_it(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _run(root)
    (root / "enrichments" / "topic_theme_clusters.json").write_text('{"data": {"clus', "utf-8")
    with caplog.at_level("ERROR"):
        second = _run(root)
    assert second["registry_health"]["source"] == "copy" and not second["registry_health"]["ok"]
    assert _ids(second) == _ids(first), "the copy keeps every storyline"
    assert "storylines: registry health=bad source=copy" in caplog.text


def test_an_artifact_lost_with_its_copy_is_reported_as_rebuilt(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _run(root)
    (root / "enrichments" / "topic_theme_clusters.json").write_text("not json", "utf-8")
    (root / "enrichments" / ttc.REGISTRY_COPY).unlink()
    with caplog.at_level("ERROR"):
        data = _run(root)
    assert data["registry_health"]["source"] == "rebuilt" and not data["registry_health"]["ok"]
    assert "health=bad source=rebuilt" in caplog.text


def test_lift_is_co_occurrence_lift_over_the_published_members(root: Path) -> None:
    """``lift_to_cluster`` keeps the meaning it had before the registry: the topic's summed lift
    (``cnt * N / (df_a * df_b)``, pairs seen in 2+ episodes) to the other members published."""
    data = _run(root)
    bundles = discover_episode_bundles(root)
    df: dict[str, int] = defaultdict(int)
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for b in bundles:
        topics, _f = partition_topic_nodes(load_kg(b))
        ids = sorted({str(t["id"]) for t in topics if t.get("id")})
        for t in ids:
            df[t] += 1
        for a, c in combinations(ids, 2):
            pairs[(a, c)] += 1
    n = len(bundles)

    def lift(a: str, c: str) -> float:
        cnt = pairs.get((min(a, c), max(a, c)), 0)
        return cnt * n / (df[a] * df[c]) if cnt >= 2 else 0.0

    checked = 0
    for c in data["clusters"]:
        ids = [m["topic_id"] for m in c["members"]]
        for m in c["members"]:
            want = sum(lift(m["topic_id"], o) for o in ids if o != m["topic_id"])
            assert m["lift_to_cluster"] == pytest.approx(round(want, 4), abs=1e-3)
            assert "strength" in m
            checked += 1
    assert checked


def test_the_alert_rule_matches_the_line_a_bad_run_logs(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import re

    import yaml

    rules = yaml.safe_load(
        (ROOT / "config" / "grafana" / "alerts" / "common" / "enrichment.yaml").read_text()
    )
    (rule,) = [
        r
        for g in rules["groups"]
        for r in g["rules"]
        if r["uid"] == "enrich-common-storyline-registry-bad"
    ]
    expr = next(d["model"]["expr"] for d in rule["data"] if d["refId"] == "bad")
    phrase = re.search(r'"([^"]+)"', expr)
    assert phrase
    _run(root)
    (root / "enrichments" / "topic_theme_clusters.json").write_text("x", "utf-8")
    with caplog.at_level("ERROR"):
        _run(root)
    assert phrase.group(1) in caplog.text


def test_every_run_records_its_counts_and_changes_for_the_dashboard(root: Path) -> None:
    from podcast_scraper.search import clustering_history

    first = _run(root)
    _run(root)
    rows = clustering_history.read(root, "storyline")
    assert len(rows) == 2
    assert rows[0]["storylines"] == len(first["clusters"]) and rows[0]["ok"] is True
    assert rows[0]["episodes_total"] == len(discover_episode_bundles(root))
    assert 0 < rows[0]["episodes_covered"] <= rows[0]["episodes_total"]
    assert [c["id"] for c in rows[0]["changes"]["created"]] == sorted(_ids(first))
    assert rows[1]["changes"]["created"] == [], "the second run changed nothing"


def _panel_pattern(panel_id: int) -> re.Pattern[str]:
    """The Grafana panel's LogsQL ``extract`` pattern as a regex (``<name>`` -> a value)."""
    board = json.loads(
        (ROOT / "config" / "grafana" / "dashboards" / "vps" / "podcast-app.json").read_text()
    )
    (panel,) = [p for p in board["panels"] if p["id"] == panel_id]
    pattern = re.search(r'extract "([^"]+)"', panel["targets"][0]["expr"])
    assert pattern
    return re.compile(
        re.sub(
            r"<\w+>", r"(\\S+)", re.escape(pattern.group(1)).replace(r"\<", "<").replace(r"\>", ">")
        )
    )


def test_the_grafana_storyline_panel_reads_the_line_a_run_logs(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        data = _run(root)
    m = _panel_pattern(8).search(caplog.text)
    assert m, "the panel's extract pattern matches nothing a run logs"
    assert int(m.group(1)) == len(data["clusters"])


def test_the_grafana_theme_panel_reads_the_line_a_build_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from podcast_scraper.search.topic_clusters import record_theme_run

    payload = {
        "clusters": [{"graph_compound_parent_id": "tc:a", "canonical_label": "a"}],
        "topic_count": 7,
        "singletons": 1,
        "id_history": {"successors": {}, "gone": ["tc:b"]},
    }
    with caplog.at_level("INFO"):
        record_theme_run(tmp_path, {"clusters": []}, payload)
    m = _panel_pattern(9).search(caplog.text)
    assert m and m.groups() == ("1", "7", "0", "1")
