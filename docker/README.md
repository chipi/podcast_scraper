# Container images

Build contexts are normally the **repository root** (`..` from this directory in compose, or `.` when invoking `docker build` from the repo root) so `COPY pyproject.toml`, `COPY src/`, etc. work unchanged.

| Directory | Image | Built by |
| --------- | ----- | -------- |
| **`api/`** | FastAPI `serve` (the `/api/*` server, no SPA static files) | `make stack-test-build` |
| **`viewer/`** | Nginx + the prebuilt Vue SPA | `make stack-test-build` |
| **`pipeline/`** | One-shot pipeline runner — `INSTALL_EXTRAS=ml` (default, ships local Whisper / spaCy / transformers / FAISS) or `INSTALL_EXTRAS=llm` (cloud APIs only) | `make stack-test-build` (ml) and `make stack-test-build-cloud` (llm) |
| **`mock-feeds/`** | Tiny Nginx sidecar serving bundled RSS / audio / transcript fixtures so the platform can be exercised without internet access | `make stack-test-build` |

Compose definitions live in [`../compose/`](../compose/). For the operator-facing run-it guide, see [Docker Compose guide](../docs/guides/DOCKER_COMPOSE_GUIDE.md). For pipeline image-tier (`ml` vs `llm`) trade-offs, see [Docker variants guide](../docs/guides/DOCKER_VARIANTS_GUIDE.md).
