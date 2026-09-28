#!/usr/bin/env python3
"""Read a syllabus PDF from data/ and print what we can extract:
class dates, topics per session, and deadlines.

This is a first-pass, heuristic parser (Pipeline stage 2: Parsing).
It only prints to stdout for now — nothing is saved or sent anywhere.
Usage: python3 scripts/parse_syllabus.py <filename in data/>
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Loose date patterns: "Jan 15", "January 15, 2026", "1/15", "1/15/26",
# "Week 3", "Session 4"
DATE_PATTERN = re.compile(
    r"""(
        \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)
        [a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?
        | \b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b
        | \bWeek\s+\d+\b
        | \bSession\s+\d+\b
        | \bClass\s+\d+\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

DEADLINE_KEYWORDS = re.compile(
    r"\b(due|deadline|submit|assignment|exam|quiz|midterm|final|paper|"
    r"project)\b",
    re.IGNORECASE,
)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def find_sessions(lines: list[str]) -> list[tuple[str, str]]:
    sessions = []
    for line in lines:
        match = DATE_PATTERN.search(line)
        if match:
            date = match.group(1).strip()
            topic = line[: match.start()].strip() + " " + line[match.end():].strip()
            sessions.append((date, topic.strip(" -:|\t")))
    return sessions


def find_deadlines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if DEADLINE_KEYWORDS.search(line)]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/parse_syllabus.py <filename in data/>")

    pdf_path = DATA_DIR / sys.argv[1]
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    text = extract_text(pdf_path)
    lines = [line for line in text.splitlines() if line.strip()]

    print(f"--- Raw extracted text ({len(lines)} non-empty lines) ---\n")
    print(text)

    print("\n--- Detected class dates / sessions ---\n")
    sessions = find_sessions(lines)
    if not sessions:
        print("(none found)")
    for date, topic in sessions:
        print(f"{date}: {topic}")

    print("\n--- Detected deadline-related lines ---\n")
    deadlines = find_deadlines(lines)
    if not deadlines:
        print("(none found)")
    for line in deadlines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
