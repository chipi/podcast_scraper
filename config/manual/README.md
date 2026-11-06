# Manual operator artifacts

## Grafana — pipeline execution (Loki)

For **GitHub #746**, import the dashboard JSON into Grafana (Dashboards → Import)
and map your **Loki** datasource.

- **Canonical file:** the pipeline-execution dashboard JSON in the infrastructure repository's Grafana dashboards

Enable **`jsonl_metrics_enabled`** and **`jsonl_metrics_echo_stdout`** on long batch runs so pipeline containers emit one JSON object per stdout line; query with `| json | event_type="run_finished"`. HTTP Prometheus scrape of the API is unchanged.
