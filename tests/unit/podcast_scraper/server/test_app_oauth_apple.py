"""Sign in with Apple — the provider (#2275). Keys are generated here; nothing calls Apple."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from podcast_scraper.server import app_oauth
from podcast_scraper.server.app_oauth import (
    APPLE_ISSUER,
    AppleProvider,
    OAuthError,
    providers_from_env,
)

pytestmark = [pytest.mark.unit]

SERVICES_ID = "app.closelistening.player.web"
EC_KEY = ec.generate_private_key(ec.SECP256R1())
EC_PEM = EC_KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
).decode()
APPLE_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _Jwks:
    """Stands in for PyJWKClient: Apple's published key is APPLE_RSA's public half."""

    def get_signing_key_from_jwt(self, _token: str) -> SimpleNamespace:
        return SimpleNamespace(key=APPLE_RSA.public_key())


def _provider() -> AppleProvider:
    return AppleProvider(
        team_id="3P3PX275ZM",
        key_id="KEY123",
        services_id=SERVICES_ID,
        private_key_pem=EC_PEM,
        jwks_client=_Jwks(),
    )


def _id_token(**over: object) -> str:
    claims = {
        "iss": APPLE_ISSUER,
        "aud": SERVICES_ID,
        "sub": "000123.abc",
        "email": "x@privaterelay.appleid.com",
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        **over,
    }
    return jwt.encode(claims, APPLE_RSA, algorithm="RS256")


def _fake_token_endpoint(monkeypatch: pytest.MonkeyPatch, id_token: str | None, seen: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["form"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(200, json={"id_token": id_token} if id_token else {})

    real = httpx.Client
    monkeypatch.setattr(
        app_oauth.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )


def test_authorization_url_asks_for_name_and_email_by_form_post() -> None:
    url = _provider().authorization_url(state="S", redirect_uri="https://x/cb")
    q = httpx.URL(url).params
    assert url.startswith("https://appleid.apple.com/auth/authorize")
    assert (q["client_id"], q["response_mode"], q["scope"], q["state"]) == (
        SERVICES_ID,
        "form_post",
        "name email",
        "S",
    )


def test_client_secret_is_an_es256_jwt_apple_will_accept_and_is_reused() -> None:
    p = _provider()
    token = p.client_secret(now=1_000_000)
    header = jwt.get_unverified_header(token)
    claims = jwt.decode(
        token,
        EC_KEY.public_key(),
        algorithms=["ES256"],
        audience=APPLE_ISSUER,
        options={"verify_exp": False},
    )
    assert (header["alg"], header["kid"]) == ("ES256", "KEY123")
    assert (claims["iss"], claims["sub"]) == ("3P3PX275ZM", SERVICES_ID)
    assert claims["exp"] - claims["iat"] <= 180 * 24 * 3600  # Apple's six-month ceiling
    assert p.client_secret(now=1_000_000 + 600) == token  # cached
    assert p.client_secret(now=1_000_000 + 24 * 3600) != token  # renewed near expiry


def test_exchange_takes_the_name_from_the_first_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    _fake_token_endpoint(monkeypatch, _id_token(), seen)
    user = json.dumps(
        {
            "name": {"firstName": "Ada", "lastName": "Lovelace"},
            "email": "x@privaterelay.appleid.com",
        }
    )
    ident = _provider().exchange_code(code="C", redirect_uri="https://x/cb", user_json=user)
    assert (ident.provider, ident.subject, ident.email, ident.name) == (
        "apple",
        "000123.abc",
        "x@privaterelay.appleid.com",
        "Ada Lovelace",
    )
    assert seen["form"]["client_id"] == SERVICES_ID and seen["form"]["code"] == "C"
    assert seen["form"]["client_secret"].count(".") == 2  # a JWT, not a static secret


def test_a_later_sign_in_without_the_user_field_falls_back_to_the_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_token_endpoint(monkeypatch, _id_token(), {})
    assert (
        _provider().exchange_code(code="C", redirect_uri="u").name == "x@privaterelay.appleid.com"
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"aud": "someone.else"},
        {"iss": "https://evil.example"},
        {"exp": int(time.time()) - 60},
    ],
)
def test_an_id_token_not_meant_for_us_is_rejected(
    monkeypatch: pytest.MonkeyPatch, bad: dict
) -> None:
    _fake_token_endpoint(monkeypatch, _id_token(**bad), {})
    with pytest.raises(OAuthError):
        _provider().exchange_code(code="C", redirect_uri="u")


def test_an_id_token_signed_by_anyone_but_apple_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forger = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(
        {
            "iss": APPLE_ISSUER,
            "aud": SERVICES_ID,
            "sub": "x",
            "email": "e",
            "exp": int(time.time()) + 600,
        },
        forger,
        algorithm="RS256",
    )
    _fake_token_endpoint(monkeypatch, forged, {})
    with pytest.raises(OAuthError):
        _provider().exchange_code(code="C", redirect_uri="u")


def test_missing_id_token_is_an_oauth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_token_endpoint(monkeypatch, None, {})
    with pytest.raises(OAuthError):
        _provider().exchange_code(code="C", redirect_uri="u")


def _apple_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_OAUTH_PROVIDER", "google")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_SECRET", "cs")
    monkeypatch.setenv("APP_OAUTH_APPLE_TEAM_ID", "3P3PX275ZM")
    monkeypatch.setenv("APP_OAUTH_APPLE_KEY_ID", "KEY123")
    monkeypatch.setenv("APP_OAUTH_APPLE_SERVICES_ID", SERVICES_ID)
    monkeypatch.setenv("APP_OAUTH_APPLE_PRIVATE_KEY", EC_PEM.replace("\n", "\\n"))


def test_providers_from_env_adds_apple_beside_google(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple_env(monkeypatch)
    monkeypatch.setenv("APP_OAUTH_PROVIDERS", "google,apple")
    providers = providers_from_env()
    assert list(providers) == ["google", "apple"]
    # The escaped one-line env value round-trips to a key that signs.
    assert providers["apple"].client_secret().count(".") == 2  # type: ignore[attr-defined]


def test_apple_is_never_inferred_from_credentials_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple_env(monkeypatch)
    monkeypatch.delenv("APP_OAUTH_PROVIDERS", raising=False)
    assert list(providers_from_env()) == ["google"]


def test_half_configured_apple_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple_env(monkeypatch)
    monkeypatch.setenv("APP_OAUTH_PROVIDERS", "apple")
    monkeypatch.delenv("APP_OAUTH_APPLE_KEY_ID")
    assert list(providers_from_env()) == ["google"]


def test_apple_never_rides_along_with_the_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    _apple_env(monkeypatch)
    monkeypatch.setenv("APP_OAUTH_PROVIDER", "mock")
    monkeypatch.setenv("APP_OAUTH_PROVIDERS", "apple")
    assert list(providers_from_env()) == ["mock"]
