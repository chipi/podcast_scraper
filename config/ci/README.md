# CI / CD configuration files

YAML and other small configs **only used by automation** (GitHub Actions, compose smoke overlays, etc.).

- **`smoke-config.yaml`** — RFC-078 full-stack smoke: fixture RSS + airgapped-style limits (see `compose/docker-compose.smoke.yml`, `make smoke-*`, `make smoke-assert-logs`, `make smoke-export-corpus`, `make smoke-assert-artifacts`).

Application defaults for humans stay under `config/manual/`, `config/examples/`, and `config/profiles/`.
