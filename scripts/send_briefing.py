#!/usr/bin/env python3
"""Format judgment_core's Must-Do / Checkpoint / Can-Do output as a
readable message and send it via Telegram. Pipeline stage 5 (Telegram
Delivery), still running on judgment_core's hardcoded task list — no
syllabus parsing wired in yet."""

import os
from pathlib import Path

from judgment_core import TASKS, TODAY, classify
from send_test_message import load_env, send_message

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

LABEL_HEADERS = {
    "Must-Do": "\U0001F534 MUST-DO",
    "Checkpoint": "\U0001F7E1 CHECKPOINT",
    "Can-Do": "\U0001F7E2 CAN-DO",
}


def format_briefing() -> str:
    results = [(task, *classify(task)) for task in TASKS]

    lines = [f"Daily Briefing — {TODAY.strftime('%A, %B %d, %Y')}"]

    for label in ("Must-Do", "Checkpoint", "Can-Do"):
        group = [r for r in results if r[1] == label]
        group.sort(key=lambda r: (r[0].due is None, r[0].due))
        lines.append("")
        lines.append(LABEL_HEADERS[label])
        if not group:
            lines.append("(none)")
            continue
        for task, _, reason in group:
            if task.due:
                date_str = f"due {task.due.isoformat()}"
            elif task.checkpoint_date:
                date_str = f"checkpoint {task.checkpoint_date.isoformat()}"
            else:
                date_str = "no fixed date"
            lines.append(f"• {task.name} ({date_str})")
            lines.append(f"  {reason}")

    return "\n".join(lines)


def main() -> None:
    load_env(ENV_PATH)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

    message = format_briefing()
    result = send_message(token, chat_id, message)

    if result.get("ok"):
        print("Briefing sent successfully.")
    else:
        raise SystemExit(f"Telegram API error: {result}")


if __name__ == "__main__":
    main()
