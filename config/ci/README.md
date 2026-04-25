# CI / CD configuration files

YAML and other small configs **only used by automation** (GitHub Actions, compose stack-test overlays, etc.).

- **`stack-test-seed/`** — Seed files copied into the `corpus_data` volume by `make stack-test-seed` before the Playwright full UI flow runs:
  - `feeds.spec.yaml` — single seeded RSS URL; the spec adds a second feed via the Configuration UI and asserts both are picked up.
  - `viewer_operator.yaml` — minimal corpus-level operator overrides (`max_episodes`, `workers`, `transcribe_missing`, …). `stack-test-seed` appends `pipeline_install_extras: ml` (default) or `pipeline_install_extras: llm` (`STACK_TEST_OPERATOR_VARIANT=cloud-thin`) to pick the compose service the API job factory spawns (`pipeline` vs `pipeline-llm`).

Application defaults for humans stay under `config/manual/`, `config/examples/`, and `config/profiles/`.
