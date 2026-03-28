# Experiment Metrics Report

**Run ID:** `baseline_ml_prod_authority_smoke_v1`
**Dataset ID:** `curated_5feeds_smoke_v1`
**Episode Count:** 5

## Intrinsic Metrics

### Quality Gates

- **Boilerplate Leak Rate:** 0.0%
- **Speaker Label Leak Rate:** 0.0%
- **Speaker Name Leak Rate (WARN):** 0.0%
- **Truncation Rate:** 0.0%
- **Failed Episodes:** None

### Length Metrics

- **Average Tokens:** 54.2000
- **Min Tokens:** 35.0000
- **Max Tokens:** 82.0000

### Performance Metrics

- **Average Latency:** 20786ms

## vs Reference Metrics

### vs silver_gpt4o_benchmark_v1

**Error:** Episode ID mismatch for reference 'silver_gpt4o_benchmark_v1': missing={'p04_e02', 'p03_e02', 'p01_e02', 'p02_e02', 'p05_e02'}, extra=set()
