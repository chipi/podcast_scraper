# Drift check configs (2.6)

Configs used to verify current code reproduces stored baselines (no drift). Run these before testing new models to confirm no regression from code changes.

**Version:** 2.6

|Config|Baseline|Use|
|------|--------|---|
|`baseline_ml_prod_authority_v1_drift_check_2.6.yaml`|baseline_ml_prod_authority_v1|ML summarization prod (Pegasus-CNN + LED-base)|
|`baseline_ml_dev_authority_smoke_v1_drift_check_2.6.yaml`|baseline_ml_dev_authority_smoke_v1|ML summarization dev (BART-small + long-fast)|
|`baseline_ner_prod_authority_v1_drift_check_2.6.yaml`|baseline_ner_prod_authority_v1|NER prod (en_core_web_trf)|
|`baseline_ner_dev_authority_v1_drift_check_2.6.yaml`|baseline_ner_dev_authority_v1|NER dev (en_core_web_sm)|

**Example:**

```bash
make experiment-run \
  CONFIG=data/eval/configs/_archive/drift_check_2.6/baseline_ml_prod_authority_v1_drift_check_2.6.yaml \
  BASELINE=baseline_ml_prod_authority_v1
```

Results go to `data/eval/runs/baseline_ml_prod_authority_v1_drift_check_2.6/` (run_id includes `_2.6`).
