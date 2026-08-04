# ADR-144: Self-host the delivery queue; outsource only the email last-mile reputation

- **Status**: Accepted — 2026-08-04 (operator ratified the shape: self-host queue + service last-mile). Implementation tracked by epic #1413 / infra slice #1412; not yet executed.
- **Date**: 2026-08-04
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [RFC-110](../rfc/RFC-110-outbound-delivery-and-seam.md) (delivery design), [ADR-145](ADR-145-channel-agnostic-outbox-seam.md) (the seam this delivers over), [PRD-046](../prd/PRD-046-delivery-and-curation.md), ADR-114 (shared Caddy edge — the deployment pattern), PRD-035 Principle 4 (bridge-only audio)
- **Tracking**: epic #1413 / infra slice #1412

## Context

PRD-046 closes the delivery loop: a per-user digest + resurfacing nudges over **Web Push** and
**email**. That forces a build-vs-buy call for the delivery mechanism, and the two channels are not
symmetric:

- **Web Push** has no reputation problem. A server with a VAPID keypair signs payloads and hands them
  to the browser's push endpoints (FCM/Mozilla/Apple). Fully self-hostable, zero third party, zero
  ongoing cost.
- **Email** has a reputation problem that is the *actual* hard part. Running an MTA is trivial;
  landing in the inbox is not. Self-hosted senders fail on IP reputation: residential ISPs commonly
  block port 25 outright; cloud/VPS IPs are frequently pre-blocklisted and need weeks of warmup; and
  Gmail/Outlook silently spam-folder anything without correct SPF + DKIM + DMARC and a warmed IP.

The operator's steer (2026-08-04): *host the queue/manager locally (Listmonk), use a service for the
last mile that comes with reputation.* This ADR records that split and why it's the right boundary —
so future contributors don't "simplify" it back into either a pure-SaaS call or a fully self-hosted
MTA.

## Decision

1. **Web Push: fully self-hosted.** Generate + store a VAPID keypair; run a push worker in our own
   infra. No third-party push provider.
2. **Email queue/management: self-hosted Listmonk**, deployed behind the shared Caddy edge (ADR-114,
   the GlitchTip pattern — vhost + port per app on the homelab).
3. **Email last-mile: relay through Amazon SES** (SMTP smarthost) for sending reputation. SES is the
   cheapest path with real deliverability and minimal lock-in (standard SMTP; swappable).
4. **Do not direct-send** email from a homelab/VPS IP.
5. **Secrets** (SES credentials, VAPID private key) via the existing secret mechanism — never
   committed. SES is a **new runtime dependency** and is accepted as such under this ADR.

The dividing principle: **self-host the mechanism we can actually run well; outsource only the one
thing that is genuinely hard to self-host — sending reputation.**

## Consequences

**Positive**

- Deliverability (the hard part) is handled by a provider built for it; the digest lands in inboxes.
- We own the queue, templates, subscriber data, and Web Push end-to-end — most of the stack stays
  local and swappable; SES is a thin, replaceable edge.
- Web Push ships with **no** new dependency and **no** reputation risk, so the loop can be de-risked
  before email is wired.

**Negative / trade-off**

- SES is a new external dependency + a small recurring cost (~$0.10/1,000 emails) and an AWS account
  surface. Accepted: it's the narrowest possible outsourcing (last mile only) and SMTP-standard, so
  migrating relays later is low-effort.
- Two channels = two delivery code paths (push worker + Listmonk/SES relay) behind the one seam.

**Neutral**

- Listmonk is a self-hosted service to operate (backups, upgrades) like GlitchTip already is — same
  operational pattern, no new class of work.
- Email deliverability requires DNS work (SPF/DKIM/DMARC for `closelistening.app`) owned by the infra
  slice (#1412), independent of app work.

## Alternatives considered

- **Pure SaaS (Resend / Postmark) called directly from the app.** Zero infra, best time-to-first-send.
  Rejected: couples product code to a SaaS + its reputation, puts a mail dependency in the app's CI,
  gives away subscriber data + template control, and offers no clean delegation boundary. The operator
  explicitly preferred self-hosting the manageable parts.
- **Fully self-hosted MTA sending direct (Maddy / Postfix / Haraka), no relay.** Maximal control, zero
  per-email cost. Rejected for real users: IP-reputation/warmup/blocklist tax and ongoing
  deliverability firefighting. Acceptable only for dogfooding-to-self, not production sends.
- **Relay through the domain's existing mailbox provider (Google Workspace / Fastmail SMTP).** Free
  within limits and fine at today's tiny volume, but TOS discourages automated bulk and sending caps
  bite as users grow. Kept as a possible interim for early dogfooding; SES is the durable choice.
- **Third-party Web Push provider (e.g. OneSignal).** Rejected: Web Push is trivially self-hostable
  with VAPID; a provider adds a dependency + data-sharing for no benefit.
