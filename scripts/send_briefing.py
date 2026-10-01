#!/usr/bin/env python3
"""Run the judgment core and send a short version of the daily briefing
via Telegram (Pipeline stage 5, Telegram Delivery).

The terminal gets the full briefing, reasoning included, exactly as
judgment_core.py prints it. Telegram gets only today's date and numbered
OVERDUE, MUST-DO and CAN-DO lists: task name, hours, and a date when the
task has one. No reasoning and no Upcoming section."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from judgment_core import (
    TODAY,
    Task,
    active_tasks,
    fmt_date,
    fmt_hours,
    judge,
    next_post,
    overdue_tasks,
    print_cost_total,
    print_judgment,
)
from send_test_message import load_env, send_message

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def task_date(task: Task | None) -> date | None:
    """The date shown on a Telegram line: the due date, or a posting
    task's post date. None means no date is shown."""
    if task is None:
        return None
    if task.posting:
        return next_post(task.posting)[1]
    return task.due


def overdue_name(task: Task) -> str:
    """First sentence of the task text, e.g. "Predictive — Final Project
    Proposal" — overdue tasks never go to the model, so there's no short
    model-written name for them. Lectures are cut to course plus label,
    e.g. "Prescriptive — Week 8 lecture" (the label is the title before
    its topic)."""
    if task.kind == "lecture" and task.title:
        course = task.description.split(" — ")[0]
        return f"{course} — {task.title.split(':')[0]}"
    return task.description.split(". ")[0].rstrip(".")


def item_line(n: int, item: dict, tasks: dict[int, Task]) -> str:
    parts = [fmt_hours(item["hours_today"])]
    d = task_date(tasks.get(item.get("id")))
    if d:
        parts.append(f"due {fmt_date(d)}")
    return f"{n}. {item['task']} ({', '.join(parts)})"


def format_briefing(result: dict) -> str:
    tasks = active_tasks()
    lines = [f"Daily Briefing — {TODAY.strftime('%A, %B %d, %Y')}"]

    overdue = overdue_tasks()
    if overdue:
        lines += ["", "OVERDUE"]
        lines += [
            f"{n}. {overdue_name(task)} (was due {fmt_date(task.due)})"
            for n, task in enumerate(overdue, 1)
        ]

    for header, key in (("MUST-DO", "must_do"), ("CAN-DO", "can_do")):
        items = result.get(key, [])
        lines += ["", header]
        if not items:
            lines.append("(none)")
        lines += [item_line(n, item, tasks) for n, item in enumerate(items, 1)]

    return "\n".join(lines)


def main() -> None:
    load_env(ENV_PATH)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing ANTHROPIC_API_KEY in .env")

    first_pass, result = judge()
    print()
    print_judgment(result, first_pass)
    print_cost_total()

    message = format_briefing(result)
    print("\n--- Telegram message ---")
    print(message)

    sent = send_message(token, chat_id, message)
    if sent.get("ok"):
        print("\nBriefing sent successfully.")
    else:
        raise SystemExit(f"Telegram API error: {sent}")


if __name__ == "__main__":
    main()
