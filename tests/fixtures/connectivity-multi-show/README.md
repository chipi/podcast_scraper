# Multi-show connectivity fixture (#1058 chunk 4)

3 shows × 2 episodes = 6 episodes, hand-crafted to exercise every
connectivity surface the relational query layer expects under #1058:

- **Cross-show Person** — `Dr. Alice Hayes` hosts show-a and show-c,
  guests on show-b; `Bob Chen` hosts show-b, guests on show-c
- **Intra-episode co-speakers** — every episode has host + guest
- **Per-show Topic + ABOUT** — bullet-derived Topics per show, with
  ABOUT edges from each Insight
- **Typed `MENTIONS_PERSON` + `MENTIONS_ORG`** — at least one of each
  across the fixture
- **Cross-show concept-Topic + `RELATED_TO`** — `concept:topic-ai-safety`
  links show-a's "AI safety" + show-b's "AI alignment" + show-c's
  "alignment problem"; `concept:topic-clean-energy` links show-a's
  "clean energy" + show-c's "renewable energy transition"
- **Entity neighborhood** — `Person → MENTIONS_PERSON ← Insight →
  ABOUT → Topic` traversal lands

## Layout

```
feeds/
├── show-{a,b,c}/
│   ├── metadata/
│   │   ├── show-X_epN.metadata.json
│   │   ├── show-X_epN.bridge.json
│   │   ├── show-X_epN.gi.json
│   │   └── show-X_epN.kg.json
│   └── transcripts/
│       └── show-X_epN.txt
```

## Regenerating

The fixture is generated deterministically by `build_fixture.py` — the
script encodes the connectivity contract and emits sorted, indented
JSON so re-runs are byte-equal:

```bash
.venv/bin/python tests/fixtures/connectivity-multi-show/build_fixture.py
```

The script is the source of truth — edit it (not the JSON) and re-run.

## Validation

`tests/integration/connectivity/test_multi_show_fixture.py` asserts the
contract holds end-to-end. If that test fails, either re-generate the
fixture (script-edit-induced staleness) or update the contract test
(if the connectivity contract itself moved).
