"""Child-process entry for the entity id map build: ``python -m`` this, read JSON from stdout.

See :func:`podcast_scraper.kg.entity_clusters._build_entity_id_map_isolated` for why the build
leaves the api process. A plain ``python -m`` child rather than ``multiprocessing``: a spawned
worker re-imports the PARENT's ``__main__``, which in the api is ``opentelemetry-instrument
python -m podcast_scraper.cli serve`` and from a test is pytest -- the child crashed and every
build silently fell back to the in-process path this exists to avoid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """``<corpus_root> <same_show_required 1|0>`` -> the map as one JSON object on stdout."""
    from podcast_scraper.kg.entity_clusters import build_entity_id_map

    root, same_show = argv
    id_map = build_entity_id_map(Path(root), same_show_required=same_show == "1")
    sys.stdout.write(json.dumps(id_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
