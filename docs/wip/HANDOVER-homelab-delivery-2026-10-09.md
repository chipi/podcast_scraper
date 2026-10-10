# Handover — homelab delivery: the new-episodes push opens where the app says (2026-10-09)

For the homelab session. One change in `agentic-ai-homelab`, already on `origin/main`; it needs
deploying on the mini. Deploying is shared state: the operator's go first.

## What changed — `6483321` fix(delivery): new-episodes push opens where the app says (payload.open_url)

- `infra/delivery/delivery/templates/podcast/push/new-episodes.v1.json.j2`: `url` was always the
  FIRST episode's deep link, so a push announcing several ("+4 more") opened only one of them. It
  now reads `payload.open_url`, which the app decides: the episode when `count` is 1, Home's
  What's new (`/#whats-new`) when several. Falls back to the first episode's deep link when the
  field is absent (an envelope queued by an app server older than podcast_scraper `dad37ea5a`).
- `infra/delivery/schema/podcast/delivery-envelope.schema.json` and
  `.../fixtures/new-episodes.v1.golden.json`: re-vendored from podcast_scraper `dad37ea5a`. The
  only difference is the optional `open_url` property and the golden's `"open_url": "/#whats-new"`.
- `infra/delivery/tests/test_contract.py`: three render tests — several → `https://closelistening.app/#whats-new`,
  one → that episode, no `open_url` → the first episode.

Verified before the push: `pytest tests` in `infra/delivery` → `113 passed, 2 skipped`; on the old
template the "several" test fails.

## To deploy

The templates are baked into the image (`Dockerfile`: `COPY delivery/ ./delivery/`, image
`closelistening-delivery:local`), so a pull alone does nothing: rebuild the image from `main`
(`6483321` or later) and recreate the delivery worker containers, per your usual mini procedure
and the Mac mini Docker rule.

## Order with the prod deploy

Independent — either order works:

- Template first, app server still old: no `open_url` in envelopes → the fallback → today's
  behaviour (first episode).
- App server first, template still old: the extra field is ignored → today's behaviour.

## Check after deploy

The next new-episodes push announcing several episodes carries
`url = https://closelistening.app/#whats-new` (the renderer absolutises against the tenant's
`app_origin`); a push about one episode carries that episode's URL. The deployed-service e2e
(`infra/delivery/e2e/`, README "Deployed-service e2e") is the end-to-end path if you want to send
one deliberately.

What a TAP does with that url is the app's side: native apps handle push taps only from **1.0.3**
(it had no handler before); the web service worker already opened `url`.
