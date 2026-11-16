"""Store review sign-in: ONE address that may sign in with a fixed code instead of an emailed link.

Google Play and the App Store review an app signed in, from their own devices, and the form asks for
credentials. Close Listening has none to give: sign-in is Google, Apple or a link sent to a mailbox,
and a reviewer controls none of those for an account of ours. So the operator configures a single
review address and a long random code; the reviewer types both in the app.

Deliberately narrow:

- **Off unless configured.** Both ``APP_REVIEW_SIGNIN_EMAIL`` and ``APP_REVIEW_SIGNIN_CODE_SHA256``
  must be set; otherwise the route answers 404 and nothing here is reachable.
- **One address.** The code signs in that address and no other; it cannot be pointed at a real
  person's account.
- **The code is never on the server.** Only its SHA-256 is configured; comparison is constant-time.
- **Everything else is the email link's path**: the allowlist, account creation, role, session.
  Switching it off is unsetting the two variables, or taking the address off the allowlist.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from podcast_scraper.server import app_magic_link


@dataclass(frozen=True)
class ReviewSignin:
    """The configured review address (normalised) and the hex SHA-256 of its code."""

    email: str
    code_sha256: str

    def matches(self, email: str, code: str) -> bool:
        """True when ``email`` is the review address and ``code`` hashes to the configured value."""
        digest = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()
        # Both compared, always, so a wrong address and a wrong code take the same time.
        same_code = hmac.compare_digest(digest, self.code_sha256)
        same_email = hmac.compare_digest(
            app_magic_link.normalise_email(email).encode("utf-8"), self.email.encode("utf-8")
        )
        return same_code and same_email


def review_signin_from_env() -> ReviewSignin | None:
    """The review sign-in from the environment, or None when it is not (fully) configured."""
    email = app_magic_link.normalise_email(os.environ.get("APP_REVIEW_SIGNIN_EMAIL", ""))
    digest = os.environ.get("APP_REVIEW_SIGNIN_CODE_SHA256", "").strip().lower()
    if "@" not in email or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return None
    return ReviewSignin(email=email, code_sha256=digest)
