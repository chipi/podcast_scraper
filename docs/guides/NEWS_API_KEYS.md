# News API keys

Some news sources come from an API that needs a key: the Guardian Content API today, others later.
This guide is how a key gets from the provider to a running source, the same way for every API.
The operator's decision of 2026-11-18: prepare this now, add the Guardian key later.

## How a source finds its key

A news source in `config/news.feeds.yaml` names the environment variable that holds its key with
`api_key_env`. The Guardian's transport falls back to `GUARDIAN_API_KEY` when the field is absent.
With no key set, the source lists nothing and the run logs one warning naming the source's URL
("the API key variable is not set"). Neither the key nor the variable's name is ever written to a
log; the variable's name is the source's `api_key_env` in `config/news.feeds.yaml`, or
`GUARDIAN_API_KEY` for the Guardian when the field is absent.

```yaml
  - url: https://content.guardianapis.com/search
    transport: guardian
    content_type: news_article
    news: {serveable: summary, mineable: true, max_items: 20, api_key_env: GUARDIAN_API_KEY}
```

## Adding a key

1. **Get the key** from the provider. The Guardian's is a free developer key from
   open-platform.theguardian.com.
2. **Local runs:** put `GUARDIAN_API_KEY=...` in the repository's gitignored `.env`. Never commit it.
3. **Production:** set the GitHub secret `PROD_GUARDIAN_API_KEY` in the repository's `prod`
   environment; the next `deploy-prod` (or `restage-prod-secrets`) delivers it. Never place a key
   file on the server by hand: each deploy rebuilds the secrets directory from GitHub secrets and
   would wipe it. On the way (ADR-115; chipi/infra #88): the secret is staged as the file
   `guardian_api_key` (empty while unset, so prod boots without it), mounted into the `api`,
   `pipeline` and `pipeline-llm` services (the api hands it to the pipeline jobs it starts), and the
   secrets shim exports it as `GUARDIAN_API_KEY`.
4. **Switch the source on** in `config/news.feeds.yaml` (the Guardian's block is there, commented
   out) and run the ingest. Its first run lists the newest articles.

## The next API

A new keyed transport reads `spec.api_key_env` like the Guardian's. Then each step above is the
same with the new variable's name: a `PROD_<NAME>_API_KEY` GitHub secret, staged by infra as the
file `<name>_api_key` (infra adds that slot; its recipe is chipi/infra
`docs/guides/DEPLOY_GOTCHAS.md`, "Adding a keyed news API"), and the source's
`api_key_env: <NAME>_API_KEY`.

## Not covered

- Rotating a key: update the GitHub secret and run `restage-prod-secrets` (or the next deploy).
- Rate limits and quotas are per provider; a source's `max_items` keeps each run small.
