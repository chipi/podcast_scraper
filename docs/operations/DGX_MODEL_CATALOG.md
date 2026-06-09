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

| Tag                    | Purpose                                                                       | Pulled     |
| ---------------------- | ----------------------------------------------------------------------------- | ---------- |
| `llama3.3:70b`         | General autoresearch / GI/KG                                                  | 2026-06-06 |
| `qwen2.5:72b-instruct` | Comparison runs                                                               | 2026-06-06 |
| `gemma2:27b`           | Comparison runs                                                               | 2026-06-06 |
| `gpt-oss:120b`         | Operator-pulled (not RFC-089)                                                 | 2026-06-05 |
| `qwen3.5:35b`          | **Prod LLM champion** (kept by #924 — see eval report linked below)           | 2026-06-08 |
| `qwen3.5:27b`          | #928 championship candidate (highest RougeL; latency-disqualified for prod)   | 2026-06-08 |
| `qwen3.5:9b`           | Comparison cell (#912 bundled-JSON flakiness known)                           | 2026-06-08 |
| `qwen3.6:latest`       | **Champion contender** via #924 — pending #932/#933 validation before swap    | 2026-06-08 |
| `qwen3-coder:30b`      | Operator personal track (excluded from autoresearch by design)                | 2026-06-08 |
| `gpt-oss:20b`          | #924 sweep — closest non-Qwen3 to baseline                                    | 2026-06-08 |
| `deepseek-r1:7b`       | #924 sweep — R1 distill (reasoning-tuned, not summary-shaped)                 | 2026-06-08 |
| `deepseek-r1:14b`      | #924 sweep — best of R1 distill family, still well below baseline             | 2026-06-08 |
| `deepseek-r1:32b`      | #924 sweep — slower AND worse than 14b                                        | 2026-06-08 |
| `deepseek-r1:70b`      | #924 sweep — **operationally disqualified** (~8 min/ep; killed mid-rerun)     | 2026-06-08 |

Full digests:

- `llama3.3:70b` → `sha256:4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d`
- `qwen2.5:72b-instruct` → `sha256:6e7fdda508e91cb0f63de5c15ff79ac63a1584ccafd751c07ca12b7f442101b8`
- `gemma2:27b` → `sha256:d7e4b00a7d7a8d03d4eed9b0f3f61a427e9f0fc5dea6aeb414e41dee23dc8ecc`
- `gpt-oss:120b` → `sha256:6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a`
- `qwen3.5:35b` → `sha256:900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a`
- `qwen3.5:27b` → `sha256:d4b8b4f4c350f5d322dc8235175eeae02d32c6f3fd70bdb9ea481e3abb7d7fc4`
- `qwen3.5:9b` → `sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c`
- `qwen3.6:latest` → `sha256:f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d`
- `gpt-oss:20b` → `sha256:e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb`
- `deepseek-r1:7b` → `sha256:96c415656d377afbff962f6cdb2394ab092ccbcbaab4b82525bc4ca800fe8a49`
- `deepseek-r1:14b` → `sha256:6e9f90f02bb3b39b59e81916e8cfce9deb45aeaeb9a54a5be4414486b907dc1e`
- `deepseek-r1:32b` → `sha256:6150cb382311b69f09cc0f9a1b69fc029cbd742b66bb8ec531aa5ecf5c613e93`
- `deepseek-r1:70b` → `sha256:4cd576d9aa16961244012223abf01445567b061f1814b57dfef699e4cf8df339`

RFC-089 originally listed `gemma2:27b-instruct` and `qwen2.5:72b-instruct`. The
`-instruct` suffix on `gemma2` is not a real Ollama tag — `gemma2:27b` IS the
instruct-tuned default; on `qwen2.5` the suffix is correct. Catalog reflects
what was actually pulled.

The autoresearch matrix (qwen3.x family, gpt-oss:20b, deepseek-r1 distills,
qwen3-coder) was pulled across 2026-06-08 to support [#924's smoke v2
refresh sweep](../guides/eval-reports/EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md).
v2.1 sweep (#44/#45) will add gemma3, phi4, hermes3, mistral-small:24b, and
llama4 candidates — catalog will be updated when those are pulled.

### Embeddings — not used on DGX by default (ADR-098)

DGX does not serve embeddings in shipped profiles. The pipeline uses
`sentence-transformers/all-MiniLM-L6-v2` in-process on the host. The
A/B in `data/eval/embedding_provider_comparison/transcript-chunked/` showed
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
