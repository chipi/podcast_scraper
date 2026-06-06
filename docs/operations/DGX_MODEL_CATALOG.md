# DGX model catalog

Authoritative list of Ollama models pinned on the DGX Spark, with their
manifest digests. Per
[RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) decision #7 —
embedding model addition per
[ADR-098](../adr/ADR-098-embedding-provider-profile-axis.md) (supersedes
RFC-089 §D4).

## How to read this file

- **Tag** is what the pipeline asks Ollama for (`vector_embedding_model`,
  `ollama_summary_model`, etc.).
- **Digest** is Ollama's manifest SHA — pin this to detect upstream model
  drift.
- **Pulled** is the UTC date the digest was captured on the operator's DGX.

## How to refresh a row

After `ollama pull <tag>` on DGX:

```bash
ssh <dgx-host> "ollama show <tag> --modelfile | grep -E '^FROM' "
```

Or read the manifest directly:

```bash
ssh <dgx-host> "cat ~/.ollama/models/manifests/registry.ollama.ai/library/<tag-without-version>/<version>"
```

Update the row + date below; commit.

## Catalog

### LLM (chat / completion)

| Tag                   | Purpose                       | Digest          | Pulled     |
| --------------------- | ----------------------------- | --------------- | ---------- |
| `llama3.3:70b`        | General autoresearch / GI/KG  | _pending_       | _pending_  |
| `qwen2.5:72b`         | Comparison runs               | _pending_       | _pending_  |
| `gemma2:27b`          | Comparison runs               | _pending_       | _pending_  |
| `gpt-oss:120b`        | Operator-pulled (not RFC-089) | _pending_       | _pending_  |

### Embeddings — not used on DGX by default (ADR-098)

DGX does not serve embeddings in shipped profiles. The pipeline uses
`sentence-transformers/all-MiniLM-L6-v2` in-process on the host. The
A/B in `eval/embedding_provider_comparison/transcript-chunked/` showed
MiniLM beats nomic under production-realistic chunking on this corpus.

| Tag                | Purpose                        | Dim | Context | Digest (weights layer)         | Pulled     |
| ------------------ | ------------------------------ | --- | ------- | ------------------------------ | ---------- |
| `nomic-embed-text` | Optional — for A/B eval reruns | 768 | 8192    | `sha256:970aa74c0a90` (274 MB) | 2026-06-06 |

Full weights digest: `sha256:970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6`
Captured from laptop Ollama 0.19.0; verify after `ollama pull` on DGX with
`cat ~/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest`
and check the layer of type `application/vnd.ollama.image.model`.

Pull only if you want to re-run the embedding-provider A/B against a
newer model release. Not required for the pipeline.

### Speech (Whisper)

Whisper Large v3 deployment shape is TBD — see RFC-089 §D3 item 4. Either
Ollama vision/audio extension or a separate FastAPI shim; document the
choice in this file once the operator commits.

## Drift policy

Any new pull or version bump on a pinned row → update this file in the same
PR that bumps the profile or code referencing the tag. CI doesn't enforce
this yet; reviewer responsibility.
