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

Digest = the weights-layer SHA from each manifest (`application/vnd.ollama.image.model`).
Refresh with `sudo jq -r '.layers[] | select(.mediaType == "application/vnd.ollama.image.model") | .digest' /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/<tag>`.

| Tag                    | Purpose                       | Size    | Digest (short)         | Pulled     |
| ---------------------- | ----------------------------- | ------- | ---------------------- | ---------- |
| `llama3.3:70b`         | General autoresearch / GI/KG  | 42.5 GB | `sha256:4824460d29f2`  | 2026-06-06 |
| `qwen2.5:72b-instruct` | Comparison runs               | 47.4 GB | `sha256:6e7fdda508e9`  | 2026-06-06 |
| `gemma2:27b`           | Comparison runs               | 15.6 GB | `sha256:d7e4b00a7d7a`  | 2026-06-06 |
| `gpt-oss:120b`         | Operator-pulled (not RFC-089) | 65.4 GB | `sha256:6be6d66a3f54`  | 2026-06-05 |

Full digests:

- `llama3.3:70b` → `sha256:4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d`
- `qwen2.5:72b-instruct` → `sha256:6e7fdda508e91cb0f63de5c15ff79ac63a1584ccafd751c07ca12b7f442101b8`
- `gemma2:27b` → `sha256:d7e4b00a7d7a8d03d4eed9b0f3f61a427e9f0fc5dea6aeb414e41dee23dc8ecc`
- `gpt-oss:120b` → `sha256:6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a`

RFC-089 originally listed `gemma2:27b-instruct` and `qwen2.5:72b-instruct`. The
`-instruct` suffix on `gemma2` is not a real Ollama tag — `gemma2:27b` IS the
instruct-tuned default; on `qwen2.5` the suffix is correct. Catalog reflects
what was actually pulled.

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

**Deferred to [#814](https://github.com/chipi/podcast_scraper/issues/814)** —
Whisper Large v3 is not in the Ollama library; the mechanism choice (Ollama
vision/audio extension vs separate FastAPI shim vs alternative) is the
prod-Whisper-via-DGX phase's decision. Pinning waits until that decision lands.

## Drift policy

Any new pull or version bump on a pinned row → update this file in the same
PR that bumps the profile or code referencing the tag. CI doesn't enforce
this yet; reviewer responsibility.
