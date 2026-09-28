#!/usr/bin/env python3
"""Send a single test message via the Telegram bot to confirm the
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env are wired up correctly."""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def send_message(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main() -> None:
    load_env(ENV_PATH)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise SystemExit(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env"
        )

    result = send_message(
        token, chat_id, "Test message from Daily Triage Agent."
    )

    if result.get("ok"):
        print("Test message sent successfully.")
    else:
        raise SystemExit(f"Telegram API error: {result}")


if __name__ == "__main__":
    main()
