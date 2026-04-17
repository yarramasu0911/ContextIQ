"""Username / password / role store for ContextIQ.

Three roles — visibility enforced on retrieval and upload:

- admin: sees everything; uploads default to public (everyone can see)
- hr:    sees public + HR-team docs + own; uploads default to team
- user:  sees public + own; uploads default to private

On first load, seeds three demo accounts so the app is usable out of the box:
    admin / admin123  (role=admin)
    hr    / hr123     (role=hr)
    user  / user123   (role=user)
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

USERS_FILE = "data/users.json"

ROLES = ("admin", "hr", "user")

# role -> default visibility applied to uploads from that role
DEFAULT_VISIBILITY = {"admin": "public", "hr": "team", "user": "private"}

_SEED_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "hr": {"password": "hr123", "role": "hr"},
    "user": {"password": "user123", "role": "user"},
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _seed_if_empty(users: dict) -> dict:
    if users:
        return users
    seeded = {
        uname: {"password": hash_password(info["password"]), "role": info["role"]}
        for uname, info in _SEED_USERS.items()
    }
    save_users(seeded)
    return seeded


def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
        # backfill missing role on legacy entries
        changed = False
        for uname, info in users.items():
            if "role" not in info:
                info["role"] = "user"
                changed = True
        if changed:
            save_users(users)
        return _seed_if_empty(users)
    return _seed_if_empty({})


def save_users(users: dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str, role: str = "user") -> bool:
    if role not in ROLES:
        return False
    users = load_users()
    if username in users:
        return False
    users[username] = {"password": hash_password(password), "role": role}
    save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    return users[username]["password"] == hash_password(password)


def get_role(username: str) -> Optional[str]:
    users = load_users()
    info = users.get(username)
    return info.get("role") if info else None


def default_visibility_for(role: str) -> str:
    return DEFAULT_VISIBILITY.get(role, "private")


def visibilities_allowed_for_upload(role: str) -> list[str]:
    """Which visibility values an uploader with this role may pick."""
    if role == "admin":
        return ["public", "team", "private"]
    if role == "hr":
        return ["team", "private"]
    return ["private"]
