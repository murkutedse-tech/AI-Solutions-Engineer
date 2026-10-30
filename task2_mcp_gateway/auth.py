"""Bearer-token authentication for the MCP gateway.

Tokens are HS256 JWTs carrying ``sub`` and ``role`` claims. Verification is
implemented with the standard library (hmac + constant-time compare) so the
gateway has no crypto dependency; swap ``JWTVerifier`` for an RS256/JWKS
verifier when fronting a real IdP — the gateway only depends on ``verify()``.

Mint a token for local testing:
    python -m task2_mcp_gateway.auth --role viewer --sub alice
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Literal

Role = Literal["admin", "viewer"]
ROLES: frozenset[str] = frozenset({"admin", "viewer"})
DEFAULT_DEV_SECRET = "dev-only-insecure-secret-change-me"


class AuthError(Exception):
    """Raised for any token problem. The message is safe to return to clients."""


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise AuthError("missing Authorization header")
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    return token.strip()


class JWTVerifier:
    def __init__(self, secret: str, *, issuer: str | None = None, audience: str | None = None, leeway_s: int = 30):
        self._key = secret.encode()
        self._issuer = issuer
        self._audience = audience
        self._leeway = leeway_s

    def mint(self, subject: str, role: Role, *, ttl_s: int = 3600) -> str:
        now = int(time.time())
        claims = {"sub": subject, "role": role, "iat": now, "exp": now + ttl_s}
        if self._issuer:
            claims["iss"] = self._issuer
        if self._audience:
            claims["aud"] = self._audience
        header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        payload = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
        signature = hmac.new(self._key, f"{header}.{payload}".encode(), hashlib.sha256).digest()
        return f"{header}.{payload}.{_b64url_encode(signature)}"

    def verify(self, token: str) -> Principal:
        try:
            header_b64, payload_b64, sig_b64 = token.split(".")
            header = json.loads(_b64url_decode(header_b64))
            signature = _b64url_decode(sig_b64)
        except ValueError:
            raise AuthError("malformed token") from None
        # Pin the algorithm: never let the token choose it ("alg": "none" / key-confusion attacks).
        if not isinstance(header, dict) or header.get("alg") != "HS256":
            raise AuthError("unsupported token algorithm")
        expected = hmac.new(self._key, f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise AuthError("invalid token signature")
        try:
            claims = json.loads(_b64url_decode(payload_b64))
        except ValueError:
            raise AuthError("malformed token") from None
        if not isinstance(claims, dict):
            raise AuthError("malformed token")

        now = time.time()
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)) or now > exp + self._leeway:
            raise AuthError("token expired or missing exp")
        nbf = claims.get("nbf")
        if isinstance(nbf, (int, float)) and now + self._leeway < nbf:
            raise AuthError("token not yet valid")
        if self._issuer and claims.get("iss") != self._issuer:
            raise AuthError("invalid token issuer")
        if self._audience:
            aud = claims.get("aud")
            if self._audience not in (aud if isinstance(aud, list) else [aud]):
                raise AuthError("invalid token audience")

        role, subject = claims.get("role"), claims.get("sub")
        if role not in ROLES:
            raise AuthError("token has no recognised role")
        if not isinstance(subject, str) or not subject:
            raise AuthError("token has no subject")
        return Principal(subject=subject, role=role)


def verifier_from_env() -> JWTVerifier:
    return JWTVerifier(
        os.environ.get("GATEWAY_JWT_SECRET", DEFAULT_DEV_SECRET),
        issuer=os.environ.get("GATEWAY_JWT_ISSUER") or None,
        audience=os.environ.get("GATEWAY_JWT_AUDIENCE") or None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mint a gateway bearer token (HS256).")
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--sub", default="local-user")
    parser.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()
    print(verifier_from_env().mint(args.sub, args.role, ttl_s=args.ttl))
