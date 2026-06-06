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

The initial plan was to replace the shim with Ollama-served `nomic-embed-text`. That hypothesis came from generic MTEB averages (~5–8 point lead over MiniLM) and the observation that podcast transcript chunks regularly exceed MiniLM's 256-token context limit. The plan **changed** based on empirical data described below.

## Decision

**Embedding provider becomes a profile-level axis** — `vector_embedding_provider: Literal["sentence_transformers", "ollama"]`. The architecture supports both backends end-to-end (config schema, loader router, index-metadata persistence, staleness detection that rebuilds FAISS on provider mismatch).

**`sentence_transformers` remains the default for every profile, including DGX.** DGX-served Ollama embeddings are available via the new axis but not enabled in any shipped profile. The Ollama embedding client exists; the door is open; no profile crosses it today.

**The shim is deleted.** `infra/dgx/embedding-shim/` is removed; RFC-089 §D4 is marked superseded. DGX runs Ollama only for LLM stages (summary, GI, KG, autoresearch) and Whisper for transcription (per ADR-096); embeddings stay in-process on whatever host runs the pipeline.

**Vector indexes record their provider** in `index_meta.json`. The staleness checker has a new `REASON_EMBEDDING_PROVIDER_MISMATCH` reason; switching `vector_embedding_provider` invalidates existing indexes and triggers a rebuild on next index run. This is correct: vectors are not comparable across providers even when the model id is nominally the same.

## Why the recommendation changed — A/B eval results

The operator ran the A/B harness at `src/podcast_scraper/evaluation/embedding_provider_eval.py` against their corpus (`.test_outputs/manual/my-manual-run4`, 100 unique episodes, 1200 insight→transcript ground-truth pairs from `SUPPORTED_BY` edges) in three modes:

| Mode                   | Description                                  | Recall@10 Δ (nomic vs MiniLM) |
| ---------------------- | -------------------------------------------- | ----------------------------- |
| Quote                  | insight → supporting Quote (short text)      | **+2.5 pt** (nomic wins)      |
| Transcript-whole       | insight → 8k chars as ONE vector             | **+9.6 pt** (nomic wins)      |
| **Transcript-chunked** | insight → 300-token chunks (production path) | **−1.9 pt** (**MiniLM** wins) |

The transcript-whole mode favoured nomic by design: MiniLM is asked to embed 8,000 chars in a single 256-token-truncated vector, which is **not what it ever does in production**. The live indexer pre-chunks transcripts at `vector_chunk_size_tokens=300` and embeds each chunk separately. Under the production-realistic mode (transcript-chunked, max chunk score per transcript), **MiniLM is consistently better** than nomic by 2–5 points on Recall@{1,5,10,20}, MRR, and nDCG@10. Plus MiniLM runs in-process at ~7 ms p50; nomic over HTTP at ~33 ms p50 (4.5× slower).

Reports committed under `eval/embedding_provider_comparison/` for verification.

The three modes together show a useful methodological lesson worth preserving with the code: a fair retrieval A/B must hold chunking constant. We almost shipped the wrong default; the chunked-both eval caught it.

## Consequences

### Positive

- DGX runs **only Ollama** for the pipeline's purposes — clean mental model, one service per concern (LLM = Ollama on DGX, embeddings = sentence_transformers on host).
- No new operational dependency for embedding (no Ollama-up requirement, no tailnet round-trip on the critical path).
- Embedding latency stays at ~7 ms p50 in-process instead of ~33 ms p50 over HTTP.
- No 2× FAISS storage cost from 768-dim vectors.
- The provider abstraction is in place if the answer changes — a future better-than-MiniLM model that wins under fair chunking can be enabled by flipping a profile field.

### Negative

- The shim deletion + provider plumbing was substantial work that doesn't (yet) flip any default. The architectural option has value, but it's option value, not delivered cost reduction.
- Operators eventually re-evaluating embedding providers must remember to test under production-realistic chunking — a tempting but wrong test design exists (transcript-whole).
- The GI subsystem (`gi/about_edges.py`) still uses MiniLM via `ABOUT_EDGE_EMBEDDING_MODEL`. Out of scope; can migrate as a follow-up if the answer changes.

### Out of scope

- Switching prod (`cloud_with_dgx_whisper_primary`) to DGX embeddings. Same conclusion as for local DGX — empirically not justified, and ADR-096's fallback contract was built around Whisper, not embeddings.
- Other Ollama embedding models (`mxbai-embed-large` etc.). The harness can score them when needed; revisit when there's a reason to think the answer would flip.
- Replacing the GI subsystem's embedding model.

## Alternatives considered

1. **Keep the shim.** Status-quo of RFC-089 §D4. Rejected: it was a stack-compatibility adapter without an architectural rationale. The shim is gone regardless of which provider wins.
2. **Switch DGX profiles to Ollama + `nomic-embed-text` as default.** Initial plan based on MTEB and the truncation hypothesis. Rejected after the empirical A/B: nomic loses to MiniLM under production-realistic chunking on this corpus, and the HTTP+model latency is real.
3. **Run embeddings on laptop CPU forever, no abstraction.** Rejected: the abstraction is cheap and gives operators a clean knob to retest periodically as new embedding models ship.

## References

- #897 — implementation issue
- #810 — DGX bring-up (where the friction surfaced)
- `src/podcast_scraper/providers/ml/embedding_ollama.py` — Ollama embedding client (available, unused by default)
- `src/podcast_scraper/providers/ml/embedding_loader.py` — router by `provider`
- `src/podcast_scraper/server/index_staleness.py` — `REASON_EMBEDDING_PROVIDER_MISMATCH`
- `src/podcast_scraper/evaluation/embedding_provider_eval.py` — A/B harness
- `eval/embedding_provider_comparison/` — committed report data
- Ollama `/api/embed` docs: <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings>
- `nomic-embed-text` model card: <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
