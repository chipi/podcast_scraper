"""Guardrail: spaCy wheel requirement file stays aligned with pyproject.toml."""

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _non_comment_lines(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_spacy_wheel_requirements_match_pyproject() -> None:
    root = _repo_root()
    req_path = root / "scripts" / "spacy_model_wheels_requirements.txt"
    pyproject_text = (root / "pyproject.toml").read_text()
    lines = _non_comment_lines(req_path)
    assert len(lines) == 2, "expected exactly en-core-web-sm and en-core-web-trf lines"
    for line in lines:
        assert line.startswith("en-core-web-"), line
        url = line.split("@", 1)[1].strip()
        assert url in pyproject_text, f"wheel URL not in pyproject.toml: {url[:72]}..."
