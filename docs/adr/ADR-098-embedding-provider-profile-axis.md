# ADR-098: Embedding provider as a profile axis, supersede RFC-089 §D4

- **Status**: Accepted
- **Date**: 2026-06-06
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) (§D4 superseded)
- **Related ADRs**: [ADR-096](ADR-096-dgx-spark-prod-primary-with-fallback.md) (unaffected — embedding routing is profile-local, prod fallback contract unchanged)
- **Related issues**: [#897](https://github.com/chipi/podcast_scraper/issues/897) (implementation)

## Context

[RFC-089 §D4](../rfc/RFC-089-dgx-spark-tailnet-integration.md) introduced a custom FastAPI service on the DGX Spark — the "embedding shim" — to wrap `sentence-transformers/all-MiniLM-L6-v2` and expose `POST /embed` on port `:8001`. The shim existed because Ollama doesn't serve sentence-transformers checkpoints out of the box, and at the time RFC-089 was written, the corpus index was built against MiniLM and had to keep working.

During DGX bring-up (#810), the operator observed that two HTTP services on DGX (`:11434` Ollama + `:8001` shim) felt wrong. On audit, the framing was right: the shim is a stack-compatibility adapter, not a design choice. It has no functional reason to exist beyond "MiniLM was the model we'd already committed to."

Re-examining the embedding model choice with 2026 data:

| Model                                    | Params | Dim  | Max context | MTEB avg | Released |
| ---------------------------------------- | ------ | ---- | ----------- | -------- | -------- |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M    | 384  | **256**     | ~56.5    | 2021     |
| `nomic-embed-text`                       | 137M   | 768  | 8192        | ~62.4    | 2024     |
| `mxbai-embed-large`                      | 335M   | 1024 | 512         | ~64.7    | 2024     |

Two material concerns with MiniLM in our context:

1. **Truncation.** Podcast transcript chunks frequently exceed 256 tokens. Half the chunk's content is silently invisible to the index. We can't tell which half. Re-chunking smaller is a workaround that fragments retrieval.
2. **Retrieval quality lag.** A 5–8 MTEB-point gap is felt — it's the difference between "wanted chunk in top-1" and "wanted chunk in top-3" for typical queries.

The shim could in principle keep MiniLM going on DGX. It just shouldn't.

## Decision

**Embedding provider becomes a profile-level axis.** Two values today:

- `sentence_transformers` (default) — in-process load via `embedding_loader.get_embedding_model`. Unchanged from current behavior. Used by `local.yaml`, `airgapped*.yaml`, and cloud profiles.
- `ollama` — POST to the Ollama HTTP API at `vector_embedding_endpoint`'s base URL. Used by DGX profiles (`local_dgx_balanced.yaml`).

The DGX-served embedding model becomes `nomic-embed-text` — Ollama-native, Apache 2.0, 8192-token context, 768-dim, mature in the Ollama registry.

**The shim is deleted.** `infra/dgx/embedding-shim/` is removed; the `infra/dgx/converge/` pyinfra deploy is trimmed to drop the systemd unit + venv install + `:8001` health probe. RFC-089 §D4 is marked superseded.

**Vector indexes record their provider** in `index_meta.json` alongside `embedding_model`. The staleness checker (`server/index_staleness.py`) has a new `REASON_EMBEDDING_PROVIDER_MISMATCH` reason; switching profiles between `sentence_transformers` and `ollama` invalidates existing indexes and triggers a rebuild on next index run. This is unavoidable — the vectors are not comparable across providers even when the model id is nominally the same.

## Consequences

### Positive

- One service on DGX, not two. Cleaner mental model.
- Retrieval quality improves end-to-end on DGX profiles (no further work).
- Chunks up to 8192 tokens are embedded in full — eliminates a silent truncation bug.
- Embedding endpoint and chat endpoint share the same Ollama lifecycle (one `systemctl status` to operate).
- Pipeline keeps `sentence_transformers` as the default — local / cloud / CI behavior is unchanged, no migration imposed on non-DGX users.

### Negative

- One-time FAISS rebuild when switching to a DGX profile. Operator-visible. Staleness machinery handles it automatically on next index run; documented in [DGX_RUNBOOK §P1](../guides/DGX_RUNBOOK.md).
- 768-dim vectors (vs 384) → roughly 2× FAISS index size. For a 1000-episode corpus with ~100 chunks each, ~300 MB total. Trivial in absolute terms.
- The GI subsystem (`gi/about_edges.py`) still uses MiniLM via `ABOUT_EDGE_EMBEDDING_MODEL`. Left out of scope for this ADR — graph edges have their own retraining cycle and the migration is additive when there's a need.

### Out of scope

- Switching prod (`cloud_with_dgx_whisper_primary`) to DGX embeddings. ADR-096's primary-with-fallback contract was built around Whisper, not embeddings; widening it requires its own decision around latency and DGX availability. Prod continues to compute embeddings in-pipeline on the cloud side.
- Replacing the in-process `sentence_transformers` path for cloud / local profiles. They still work the same way.
- Other Ollama embedding models. `mxbai-embed-large` would also work; pick can be reopened later.

## Alternatives considered

1. **Keep the shim.** Status-quo of RFC-089 §D4. Rejected: see Context — shim is an adapter without a design rationale, plus MiniLM has known limits.
2. **Run embeddings on laptop CPU forever.** Defer the question. Rejected: works against the whole point of having DGX, and laptop-CPU indexing of a growing corpus does compound.
3. **Build the shim against `nomic-embed-text` instead of MiniLM.** Would resolve the quality and context-length issues without involving Ollama. Rejected: doesn't eliminate the "two services" friction, and `nomic-embed-text` is a first-class Ollama model already — re-implementing its HTTP boundary is duplication.

## References

- #897 — implementation issue
- #810 — DGX bring-up (where the friction surfaced)
- `src/podcast_scraper/providers/ml/embedding_ollama.py` — new Ollama embedding client
- `src/podcast_scraper/providers/ml/embedding_loader.py` — router by `provider`
- `src/podcast_scraper/server/index_staleness.py` — `REASON_EMBEDDING_PROVIDER_MISMATCH`
- Ollama `/api/embed` docs: <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings>
- `nomic-embed-text` model card: <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
