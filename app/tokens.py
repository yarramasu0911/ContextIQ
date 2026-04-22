"""JWT session tokens. Replaces passing `X-Password` on every API call.

Flow:
  POST /auth/login {username, password} -> {token}
  subsequent requests: Authorization: Bearer <token>
  token payload: {sub: username, role, exp}
"""

from __future__ import annotations

import os
import time
from typing import Optional

import jwt

SECRET_KEY = os.getenv(
    "CONTEXTIQ_JWT_SECRET",
    # Stable-ish dev default so tokens survive a dev restart. In production,
    # set CONTEXTIQ_JWT_SECRET to a real secret. Must be >= 32 bytes for
    # PyJWT to stop warning about insufficient HMAC key length.
    "dev-secret-rotate-me-in-prod-padding-xxxxxx",
)
ALGORITHM = "HS256"
DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24h


def create_token(
    username: str, role: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": int(time.time()) + ttl_seconds,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
