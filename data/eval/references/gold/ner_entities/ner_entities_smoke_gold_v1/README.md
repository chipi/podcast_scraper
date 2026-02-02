# NER Entities Smoke Gold v1 (Materialized)

This folder contains **gold NER annotations** for the smoke dataset.

## Dataset

- **dataset_id:** curated_5feeds_smoke_v1
- **episodes covered:** 5 (p01_e01, p02_e01, p03_e01, p04_e01, p05_e01)

## Labels in scope

- PERSON, ORG, GPE, PRODUCT, EVENT

## Annotation policy (v1)

- **PERSON:** host + guest names (all occurrences)
- **ORG:** show title (all occurrences)
- Other labels are reserved for future expansion.

## Preprocessing / materialization contract

Gold offsets and `text_fingerprint` are valid **only** for the materialized `.txt` inputs generated with:

- **preprocessing_profile:** cleaning_v3

If preprocessing or materialization changes, regenerate gold.

## Files

- One file per episode: `<episode_id>.json`
- `index.json` declares coverage and label scope
