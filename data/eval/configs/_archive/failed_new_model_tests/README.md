# Failed New Model Tests

This archive contains experiment configurations that were tested but did not meet quality thresholds.

## Archived Configs

1. **`baseline_map_distilbart12-6__reduce_led-base__fatter-map.yaml`**
   - **Issue**: Boilerplate leak + failed episode
   - **Model**: DistilBART-12-6 (map) → LED-base (reduce)
   - **Notes**: Fatter map configuration led to boilerplate content leaking into summaries and at least one failed episode

2. **`baseline_map_distilbart12-6__reduce_led-large.yaml`**
   - **Issue**: Min tokens collapse risk
   - **Model**: DistilBART-12-6 (map) → LED-large (reduce)
   - **Notes**: Risk of summary collapse below minimum token thresholds

3. **`baseline_map_bart-base__reduce_led-large.yaml`**
   - **Issue**: Slow + over-compressed
   - **Model**: BART-base (map) → LED-large (reduce)
   - **Notes**: Performance too slow for production use, and summaries were over-compressed

4. **`baseline_map_distilbart12-6__reduce_led-base.yaml`**
   - **Model**: DistilBART-12-6 (map) → LED-base (reduce)
   - **Notes**: Archived to make way for v2 version

5. **`baseline_map_flan-t5-base__reduce_led-base.yaml`**
   - **Model**: FLAN-T5-base (map) → LED-base (reduce)
   - **Notes**: Archived to make way for v2 version

6. **`baseline_map_pegasus-cnn__reduce_led-base.yaml`**
   - **Model**: Pegasus-CNN (map) → LED-base (reduce)
   - **Notes**: Archived to make way for v2 version (later promoted to baseline_ml_prod_candidate_v1)

7. **`baseline_map_distilbart12-6__reduce_led-base_v2.yaml`**
   - **Model**: DistilBART-12-6 (map) → LED-base (reduce)
   - **Notes**: v2 version archived after Pegasus → LED-base was selected as production candidate

8. **`baseline_map_flan-t5-base__reduce_led-base_v2.yaml`**
   - **Model**: FLAN-T5-base (map) → LED-base (reduce)
   - **Notes**: v2 version archived after Pegasus → LED-base was selected as production candidate

## Archive Date

Archived: 2026-02-01 (initial batch)
Updated: 2026-02-01 (added v2 configs after production candidate selection)

## Note

Configs 4-6 were archived after creating `_v2` versions. Configs 7-8 (v2 versions) were archived after `baseline_map_pegasus-cnn__reduce_led-base_v2` was promoted to `baseline_ml_prod_candidate_v1`.

## Related Runs

Corresponding experiment runs are archived in:
`data/eval/runs/_archived/failed_new_model_tests/`
