"""Simple demo authentication for the Amazon QA chatbot."""

USERS = {
    "admin": {"password": "admin123", "name": "Admin User"},
    "demo": {"password": "demo123", "name": "Demo User"},
    "analyst": {"password": "analyst123", "name": "QA Analyst"},
}


def authenticate(username: str, password: str) -> dict | None:
    """Return user profile if credentials are valid, else None."""
    user = USERS.get(username.strip().lower())
    if user and user["password"] == password:
        return {"username": username.strip().lower(), "name": user["name"]}
    return None
