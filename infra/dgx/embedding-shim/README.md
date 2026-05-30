# DGX embedding shim (RFC-089)

Small HTTP service on the DGX host (`:8001`) that exposes `POST /embed` for
`sentence-transformers/all-MiniLM-L6-v2`. Use from the laptop or prod/drill VPS
via tailnet when indexing or clustering corpora.

## Run (on DGX)

```bash
cd infra/dgx/embedding-shim
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Health

```bash
curl -sS "http://dgx-llm-1.<tailnet>:8001/health"
```

## Client (podcast_scraper)

```bash
python -m podcast_scraper.cli index --output-dir ./output \
  --embedding-endpoint "http://dgx-llm-1.<tailnet>:8001/embed"
```

GPU-built FAISS indexes are not byte-identical to CPU indexes; tests assert
functional top-K overlap, not file equality.
