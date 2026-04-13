import hashlib
import json
import os

USERS_FILE = "data/users.json"


def hash_password(password: str) -> str:
    """Hash password with SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    """Load users from file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users: dict):
    """Save users to file."""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str) -> bool:
    """Register a new user. Returns True if successful."""
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": hash_password(password),
    }
    save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    """Check if username and password match."""
    users = load_users()
    if username not in users:
        return False
    return users[username]["password"] == hash_password(password)