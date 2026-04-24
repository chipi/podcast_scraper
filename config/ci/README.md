# CI / CD configuration files

YAML and other small configs **only used by automation** (GitHub Actions, compose stack-test overlays, etc.).

- **`stack-test-config.yaml`** — RFC-078 full-stack stack-test: fixture RSS + airgapped-style limits (see `compose/docker-compose.stack-test.yml`, `make stack-test-*`, `make stack-test-assert-logs`, `make stack-test-export`, `make stack-test-assert-artifacts`).

Application defaults for humans stay under `config/manual/`, `config/examples/`, and `config/profiles/`.
