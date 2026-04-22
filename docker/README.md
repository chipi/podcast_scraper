# Container images

Build contexts are normally the **repository root** (`.`) so `COPY pyproject.toml`, `COPY src/`, etc. work unchanged.

| Directory | Image | Notes |
| --------- | ----- | ----- |
| **`pipeline/`** | ML / LLM / core pipeline runner (`podcast_scraper` CLI, optional preload) | `docker build -f docker/pipeline/Dockerfile .` |
| **`api/`** | FastAPI `serve` (viewer API, no bundled SPA) | Stack + local API tests |
| **`viewer/`** | Nginx + built Vue SPA | RFC-079 stack front door |

Legacy references to a root `Dockerfile` should be updated to **`docker/pipeline/Dockerfile`**.

Compose definitions live in **[`../compose/`](../compose/)** (sibling of `docker/`), not the repository root.
