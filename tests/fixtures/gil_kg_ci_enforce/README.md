# CI fixtures: GIL + KG quality metrics

Committed **`metadata/*.gi.json`** and **`*.kg.json`** used by:

- GitHub Actions (`python-app.yml`): `gil_quality_metrics.py` and `kg_quality_metrics.py` with `--enforce`
- Local: `make quality-metrics-ci`

Regenerate `ci_sample.gi.json` if the GIL schema or stub shape changes:

```bash
export PYTHONPATH=src:$PYTHONPATH
python -c "
from pathlib import Path
from podcast_scraper.gi import build_artifact, write_artifact
p = Path('tests/fixtures/gil_kg_ci_enforce/metadata')
p.mkdir(parents=True, exist_ok=True)
art = build_artifact(
    'ci-fixture',
    'Hello world transcript sample for CI quality metrics fixture.',
    prompt_version='v1',
)
write_artifact(p / 'ci_sample.gi.json', art, validate=True)
"
```

Update `ci_sample.kg.json` from `tests/fixtures/kg/minimal.kg.json` if the KG v1 schema changes.
