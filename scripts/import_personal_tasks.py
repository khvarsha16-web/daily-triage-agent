#!/usr/bin/env python3
"""Import the weekly personal task file (Pipeline stage 2: Parsing, for
the to-do list). Run by hand after editing data/personal_tasks.txt.

One freeform task per line; blank lines and lines starting with "#" are
skipped. Lines unchanged since the last import keep their confirmed
result and are not re-reviewed. New or edited lines go to Claude once,
which returns a short title and a date ONLY if one is written in the
line (never estimated hours or importance). Code then checks each date:
a written date must appear in the line; a weekday must resolve by the
rule below and the result must fall on that weekday.

Weekday rule (relative to the import date): a bare weekday is the
nearest occurrence on or after the import date (today counts); "next
<weekday>" is that date plus 7 days.

Nothing is saved until you review the batch and accept it. The result
goes to data/parsed/personal.json (gitignored), which judgment_core.py
reads. Moo and recurring tasks like "talk to parents" stay in code.

Usage: python3 scripts/import_personal_tasks.py
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import anthropic

from extract_syllabus import date_in_quote
from parse_syllabus import DATA_DIR
from send_test_message import load_env

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
TASKS_FILE = DATA_DIR / "personal_tasks.txt"
OUT_PATH = DATA_DIR / "parsed" / "personal.json"

MODEL = "claude-sonnet-5"

WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]
# Full names and common abbreviations (e.g. "Thu", "Thurs").
WEEKDAY_PATTERN = (
    r"\b(next\s+)?(mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)"
    r"(day|sday|nesday|rsday|urday)?\b"
)
MONTH_DAY_PATTERN = (
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}/\d{1,2}\b"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_line": {"type": "string"},
                    "title": {"type": "string"},
                    "date": {
                        "anyOf": [
                            {"type": "string", "format": "date"},
                            {"type": "null"},
                        ]
                    },
                    "date_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["source_line", "title", "date", "date_text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You convert lines from a person's weekly to-do file into \
structured tasks. Each line is one task, written freeform.

Return exactly one item per input line, in the same order:
- source_line: the line copied exactly, character for character.
- title: a short name for the task (a few words), taken from the line.
- date: an ISO date (YYYY-MM-DD) ONLY if a date or day is actually written \
in the line. Otherwise null. Never invent or infer a date that isn't \
written (e.g. "soon", "this week" or no date at all → null).
- date_text: the exact words in the line that the date came from (e.g. \
"Oct 5", "Thursday", "next Fri"), copied verbatim; null when date is null.

Resolving dates, relative to the import date you are given:
- A written month and day (e.g. "Oct 5"): that date, in the import date's \
year unless that would put it more than 2 months in the past, then the \
next year.
- "today" / "tomorrow": the import date / the day after.
- A bare weekday (e.g. "Thursday"): the nearest occurrence on or after the \
import date — if the import date is that weekday, it is the import date.
- "next <weekday>": the bare-weekday date above plus 7 days.
- Anything else relative ("in 3 days", "end of month", "this weekend"): \
resolve it if you can, and code will flag it for the person to confirm.

Do not estimate hours, effort or importance, and do not add anything that \
isn't in the line."""


# --- File handling ------------------------------------------------------


def read_lines() -> list[str]:
    """Task lines from the file: stripped, blanks and # comments skipped,
    exact duplicates kept once."""
    if not TASKS_FILE.exists():
        raise SystemExit(f"No task file at {TASKS_FILE} — create it first.")
    seen, lines = set(), []
    for raw in TASKS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            print(f"Note: duplicate line kept once: {line!r}")
            continue
        seen.add(line)
        lines.append(line)
    return lines


def load_saved() -> dict[str, dict]:
    """Confirmed items from the last import, keyed by their source line."""
    if not OUT_PATH.exists():
        return {}
    items = json.loads(OUT_PATH.read_text())["items"]
    return {item["source_line"]: item for item in items}


# --- Model call ---------------------------------------------------------


def convert_lines(lines: list[str], import_date: date) -> list[dict]:
    client = anthropic.Anthropic()
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    user_message = (
        f"Import date: {import_date.strftime('%A, %B %d, %Y')} "
        f"({import_date.isoformat()})\n\n"
        f"Lines ({len(lines)}):\n<lines>\n{numbered}\n</lines>\n\n"
        "The numbers are not part of the lines; don't copy them."
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        raise SystemExit(
            "Anthropic API rejected the key in .env — check ANTHROPIC_API_KEY."
        )
    except anthropic.APIConnectionError:
        raise SystemExit("Network error reaching the Anthropic API.")

    if response.stop_reason == "refusal":
        raise SystemExit(f"Model declined to respond: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise SystemExit("Model output hit max_tokens; JSON may be incomplete.")

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["items"]


# --- Code checks ---------------------------------------------------------


def weekday_index(abbrev: str) -> int:
    return next(i for i, name in enumerate(WEEKDAYS) if name.startswith(abbrev))


def resolve_weekday(import_date: date, weekday: int, is_next: bool) -> date:
    """Nearest occurrence on or after the import date (today counts);
    "next" adds 7 days to that."""
    d = import_date + timedelta(days=(weekday - import_date.weekday()) % 7)
    return d + timedelta(days=7) if is_next else d


def check_item(item: dict, import_date: date) -> list[str]:
    """Reasons this item needs a look; empty if it passes every check."""
    line = item["source_line"]
    if item.get("date_set_by_user"):
        return []
    if item["date"] is None:
        looks_dated = (
            re.search(WEEKDAY_PATTERN, line, re.I)
            or re.search(MONTH_DAY_PATTERN, line, re.I)
            or re.search(r"\b(today|tomorrow)\b", line, re.I)
        )
        if looks_dated:
            return ["line looks like it has a date, but none was extracted"]
        return []

    try:
        d = date.fromisoformat(item["date"])
    except ValueError:
        return [f"date {item['date']!r} does not parse"]

    reasons = []
    text = item["date_text"] or ""
    if not text or text not in line:
        reasons.append(f"date_text {text!r} is not in the line")

    weekday = re.search(WEEKDAY_PATTERN, text, re.I)
    if date_in_quote(d, text) or re.search(rf"\b0?{d.month}/0?{d.day}\b", text):
        # A written date; if a weekday is written too (in the date words or
        # elsewhere in the line, e.g. "Oct 5 (Monday)"), they must agree.
        weekday = weekday or re.search(WEEKDAY_PATTERN, line, re.I)
        if weekday and d.weekday() != weekday_index(weekday.group(2).lower()):
            reasons.append(
                f"{d.strftime('%a %b %d')} is not a "
                f"{WEEKDAYS[weekday_index(weekday.group(2).lower())].title()}"
            )
    elif weekday:
        stated = weekday_index(weekday.group(2).lower())
        expected = resolve_weekday(import_date, stated, bool(weekday.group(1)))
        if d != expected:
            reasons.append(
                f"{text!r} should resolve to {expected.strftime('%a %b %d')} "
                f"from import date {import_date.strftime('%a %b %d')}, not "
                f"{d.strftime('%a %b %d')}"
            )
        # Independent of the rule above: the date must be that weekday.
        if d.weekday() != stated:
            reasons.append(
                f"{d.strftime('%a %b %d')} does not fall on a "
                f"{WEEKDAYS[stated].title()}"
            )
    elif re.fullmatch(r"(?i)today|tomorrow", text.strip()):
        expected = import_date + timedelta(days=text.strip().lower() == "tomorrow")
        if d != expected:
            reasons.append(f"{text!r} should be {expected.strftime('%a %b %d')}")
    else:
        reasons.append(f"can't verify {text!r} in code; confirm the date by hand")

    if d < import_date:
        reasons.append("date has already passed (it will show as overdue)")
    return reasons


def make_ids(items: list[dict], taken: set[str]) -> None:
    """Stable readable ids from titles, made unique in code."""
    for item in items:
        slug = re.sub(r"[^a-z0-9]+", "-", item["title"].lower()).strip("-")[:40]
        # "Personal appointment" → "personal-appointment", not
        # "personal-personal-appointment".
        slug = slug.removeprefix("personal-")
        base = f"personal-{slug or 'task'}"
        new_id, n = base, 2
        while new_id in taken:
            new_id, n = f"{base}-{n}", n + 1
        item["id"] = new_id
        taken.add(new_id)


# --- Review --------------------------------------------------------------


def fmt(d: str | None) -> str:
    return date.fromisoformat(d).strftime("%a %b %d") if d else "—"


def print_review(new: list[dict], kept: list[dict], removed: list[dict]) -> None:
    if new:
        print("\nNEW OR CHANGED LINES — review these:\n")
        for n, item in enumerate(new, 1):
            print(f"{n}. {item['source_line']}")
            print(f"     → {item['title']}  |  date: {fmt(item['date'])}")
            for reason in item["review_reasons"]:
                print(f"     ⚠ {reason}")
    if kept:
        print(f"\nUnchanged since last import, not re-reviewed: {len(kept)}")
        for item in kept:
            print(f"   {item['title']}  |  date: {fmt(item['date'])}")
    if removed:
        print(f"\nRemoved from the file since last import: {len(removed)}")
        for item in removed:
            print(f"   {item['title']}  |  date: {fmt(item['date'])}")


def edit_item(item: dict, import_date: date) -> None:
    title = input(f"  title [{item['title']}]: ").strip()
    if title:
        item["title"] = title
    raw = input(
        f"  date [{item['date'] or 'none'}] (YYYY-MM-DD, 'none', Enter keeps): "
    ).strip()
    if raw.lower() == "none":
        item["date"], item["date_text"] = None, None
        item["date_set_by_user"] = True
    elif raw:
        try:
            date.fromisoformat(raw)
        except ValueError:
            print("  Not a valid date; unchanged.")
            return
        item["date"] = raw
        item["date_set_by_user"] = True
    item["review_reasons"] = check_item(item, import_date)


def review(
    new: list[dict], kept: list[dict], removed: list[dict], import_date: date
) -> bool:
    """Batch review loop. True means accepted (save), False means quit."""
    while True:
        print_review(new, kept, removed)
        flagged = sum(1 for item in new if item["review_reasons"])
        print(
            "\n[a] accept all and save   "
            + ("[e N] edit item N   " if new else "")
            + "[q] quit without saving"
        )
        choice = input("> ").strip().lower()
        if choice == "q":
            return False
        if choice == "a":
            if flagged and input(
                f"{flagged} item(s) still flagged — accept anyway? [y/N] "
            ).strip().lower() != "y":
                continue
            return True
        edit = re.fullmatch(r"e\s*(\d+)", choice)
        if edit and 1 <= int(edit.group(1)) <= len(new):
            edit_item(new[int(edit.group(1)) - 1], import_date)
        else:
            print("Type a, e N, or q.")


def main() -> None:
    import_date = date.today()
    lines = read_lines()
    saved = load_saved()

    kept = [saved[line] for line in lines if line in saved]
    new_lines = [line for line in lines if line not in saved]
    removed = [item for line, item in saved.items() if line not in lines]

    if not new_lines and not removed:
        print("No changes since the last import; nothing to review.")
        return

    new: list[dict] = []
    if new_lines:
        load_env(ENV_PATH)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Missing ANTHROPIC_API_KEY in .env")
        new = convert_lines(new_lines, import_date)
        returned = [item["source_line"] for item in new]
        if returned != new_lines:
            raise SystemExit(
                "Model output doesn't match the input lines one-to-one "
                "(a line was changed, dropped, merged or reordered). Nothing "
                f"saved; try again.\nSent: {new_lines}\nGot: {returned}"
            )
        make_ids(new, {item["id"] for item in kept})
        for item in new:
            item["imported_on"] = import_date.isoformat()
            item["review_reasons"] = check_item(item, import_date)

    if not review(new, kept, removed, import_date):
        print("Quit — nothing saved.")
        return

    by_line = {item["source_line"]: item for item in kept + new}
    items = [by_line[line] for line in lines]  # file order
    for item in items:
        item.pop("review_reasons", None)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {"last_import": import_date.isoformat(), "items": items},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Saved {len(items)} task(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
