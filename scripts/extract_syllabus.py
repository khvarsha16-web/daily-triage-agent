#!/usr/bin/env python3
"""Model-based syllabus extractor (Pipeline stage 2: Parsing).

Sends a syllabus's schedule section to Claude with a strict JSON schema,
then checks every item in code: dates parse and fall inside the semester,
ids are unique, source_text is a verbatim quote from the extracted text,
and each item's date appears in its own quote. Items that fail a check
are kept and marked needs_review with the reasons, and so are items with
no date at all. Items the user has confirmed against the PDF
(confirmed_by_user) skip the date-in-quote and no-date checks.

Parse once: the result is saved to data/parsed/<name>.json (gitignored
with the rest of data/). Later runs read that file and never call the
model. To re-extract, delete the file. After editing the saved file by
hand, --recheck re-runs the checks on it (still no model call).

Rows matching a syllabus's skip_title_pattern are dropped in code after
extraction (kept under "skipped_by_config" in the saved file, never
loaded as items) — e.g. Agentic AI's Final Presentations, whose real
date isn't known yet.

Usage: python3 scripts/extract_syllabus.py <name> [--recheck]
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import anthropic

from parse_syllabus import DATA_DIR, extract_text
from send_test_message import load_env

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PARSED_DIR = DATA_DIR / "parsed"

MODEL = "claude-sonnet-5"

# From the Agentic AI syllabus: "Fall 2026, from August 24, 2026 to
# December 9, 2026".
SEMESTER_START = date(2026, 8, 24)
SEMESTER_END = date(2026, 12, 9)

SYLLABI = {
    "predictive": {
        "pdf": "Predictive syllabus.pdf",
        "course": "MKT/BUAN 6337 Predictive Analytics for Data Science",
        "id_prefix": "pred",
        # The schedule section runs from this heading to the next one.
        "start_marker": "Assignments & Academic Calendar",
        "end_marker": "PROJECT WORK",
    },
    "prescriptive": {
        "pdf": "prescriptive syllabus.pdf",
        "course": "BUAN 6398.SW1 Prescriptive Analytics",
        "id_prefix": "pres",
        # Ends at "Grading Scheme" (not "Notes:") so the due-time note on
        # the next page is included.
        "start_marker": "Topics and Tentative Academic and Assignments Calendar",
        "end_marker": "Grading Scheme",
        # Exam 3 is Dec 12, in finals week — later than the shared
        # SEMESTER_END.
        "semester_end": date(2026, 12, 12),
        # Week rows give only a "Week of" start date; the week's end is
        # computed in code (see fill_week_ends), not by the model.
        "week_of_rows": True,
    },
    "agentic": {
        "pdf": "agentic AI Syllabus.pdf",
        "course": "BUAN 6V99 Agentic AI & Process Automation",
        "id_prefix": "agai",
        "start_marker": "Tentative Course Schedule",
        "end_marker": "About Classes",
        # Live weekly classes are sessions (calendar context), not
        # self-paced lectures.
        "kinds": ["session", "deliverable", "break"],
        "session_rows": True,
        # The Final Presentation date (Nov 30 or Dec 7) is assigned
        # randomly and not known yet; it stays hand-typed in
        # judgment_core.py until it is.
        "skip_title_pattern": r"Final Presentations",
    },
}

# Kinds dated by a week row; only these get a computed week end.
WEEK_KINDS = ("lecture", "project_work", "break")

WEEK_OF_NOTE = """This syllabus's week rows give only a start date ("Week \
of"), not an end date. For an item dated by its week row (a lecture, \
project work, or break), set date_start to the week-of date and date_end to \
null — code computes the end of the week. Do not calculate it yourself. \
This overrides the week-row range rule for this syllabus. Deliverables and \
exams with their own date still use that date for both fields."""

SESSION_NOTE = """This syllabus is a live weekly class. Kinds for this \
syllabus are only session, deliverable and break:
- A numbered class row is kind "session" (a live class meeting), not \
lecture. Id by class number, e.g. "agai-session-6".
- A class row whose title is a presentation students give (e.g. \
"Mini-Project Presentations", "Proposal Presentations") is kind \
"deliverable" only — one item for the row, not a session as well. Id by its \
name, e.g. "agai-deliverable-mini-project".
- A row with class # "n/a" (no class) is kind "break".
This overrides the general kind list for this syllabus."""

KINDS = ["lecture", "project_work", "deliverable", "exam", "break"]

NULLABLE_DATE = {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]}


def kinds_for(config: dict) -> list[str]:
    return config.get("kinds", KINDS)


def schema_for(kinds: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "course": {"type": "string"},
                        "kind": {"type": "string", "enum": kinds},
                        "title": {"type": "string"},
                        "date_start": NULLABLE_DATE,
                        "date_end": NULLABLE_DATE,
                        "source_text": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "course",
                        "kind",
                        "title",
                        "date_start",
                        "date_end",
                        "source_text",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = """You extract a course schedule from text that was pulled \
out of a syllabus PDF. The schedule was a table, but text extraction \
flattened it: cells wrap across lines, adjacent rows can be stacked, and \
columns can run together on one line. Reconstruct what the table says — \
nothing more.

Return one item per lecture, project work block (a scheduled week of \
working on the project rather than a lecture), deliverable (assignment, \
submission, project milestone, required setup step), exam, or break. For \
each item:
- id: stable and readable, "<prefix>-<kind>-<short-slug>", e.g. \
"pred-lecture-1", "pred-exam-1", "pred-deliverable-final-project-proposal", \
"pred-break-thanksgiving". Use the prefix you are given. When lectures \
aren't individually numbered, use the week number instead, e.g. \
"pres-lecture-week-2".
- course: exactly the course string you are given.
- kind: lecture, project_work, deliverable, exam, or break.
- title: the item's name as written in the text (you may rejoin words split \
across lines).
- date_start / date_end: ISO dates (YYYY-MM-DD), all in 2026.
  - A single date: date_end equals date_start.
  - A range (a week row, a break, a multi-day exam window like "Oct 9 & \
10"): date_start is the first day, date_end the last.
  - A lecture with no date of its own takes its week row's date range.
  - If the date is missing, not a date (e.g. "ASAP"), or ambiguous — \
including when stacked rows make it unclear which week an item belongs to \
— use null for both. Do not guess.
- source_text: a VERBATIM quote copied from the text, containing the item \
and its date (if it has one). It may span several lines; copy the \
characters exactly, including the line breaks between them. Do not \
paraphrase, reorder, or fix typos.

Rules:
- Only include what is in the text. Do not add items, estimate hours or \
effort, rate importance or weight, or add notes.
- Do not include week rows that have no lecture, deliverable, exam, or \
break of their own."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def schedule_text(config: dict) -> str:
    text = extract_text(DATA_DIR / config["pdf"])
    start = text.index(config["start_marker"])
    end = text.index(config["end_marker"], start)
    return text[start:end]


def extract_items(config: dict, text: str) -> list[dict]:
    client = anthropic.Anthropic()
    user_message = (
        f"Course: {config['course']}\n"
        f"Id prefix: {config['id_prefix']}\n\n"
        + (f"{WEEK_OF_NOTE}\n\n" if config.get("week_of_rows") else "")
        + (f"{SESSION_NOTE}\n\n" if config.get("session_rows") else "")
        + f"Extracted schedule text:\n<schedule>\n{text}\n</schedule>"
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema_for(kinds_for(config)),
                }
            },
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

    text_out = next(b.text for b in response.content if b.type == "text")
    return json.loads(text_out)["items"]


def date_in_quote(d: date, quote: str, start: date | None = None) -> bool:
    """Whether a date is written in the quote, e.g. "Sep 16". An end date
    in the same month as the start may appear as a bare day ("Sep 7 – 13",
    "Oct 9 & 10")."""
    month = d.strftime("%b")
    month_pattern = "Sept?" if month == "Sep" else month
    if re.search(rf"\b{month_pattern}[a-z]*\.?\s+0?{d.day}\b", quote):
        return True
    if start is not None and start.month == d.month:
        return re.search(rf"\b0?{d.day}\b", quote) is not None
    return False


def skip_by_config(
    items: list[dict], config: dict
) -> tuple[list[dict], list[dict]]:
    """Drop items whose title matches the syllabus's skip_title_pattern.
    Done in code, not by the prompt, so the model's rule stays "only what
    is in the text" and the exclusion is visible in the saved file."""
    pattern = config.get("skip_title_pattern")
    if not pattern:
        return items, []
    kept = [i for i in items if not re.search(pattern, i["title"])]
    skipped = [i for i in items if re.search(pattern, i["title"])]
    return kept, skipped


def semester_bounds(config: dict) -> tuple[date, date]:
    return SEMESTER_START, config.get("semester_end", SEMESTER_END)


def fill_week_ends(items: list[dict], config: dict) -> list[dict]:
    """For syllabi whose week rows give only a start date, compute the
    week's end (start + 6 days) in code and mark it as computed."""
    if not config.get("week_of_rows"):
        return items
    for item in items:
        if (
            item["kind"] in WEEK_KINDS
            and item["date_start"] is not None
            and item["date_end"] is None
        ):
            try:
                start = date.fromisoformat(item["date_start"])
            except ValueError:
                continue  # left for check_items to flag
            item["date_end"] = (start + timedelta(days=6)).isoformat()
            item["date_end_computed"] = True
    return items


def check_items(
    items: list[dict],
    text: str,
    semester_start: date = SEMESTER_START,
    semester_end: date = SEMESTER_END,
) -> list[dict]:
    normalized_text = normalize(text)
    id_counts: dict[str, int] = {}
    for item in items:
        id_counts[item["id"]] = id_counts.get(item["id"], 0) + 1

    for item in items:
        reasons = []
        parsed = {}
        for field in ("date_start", "date_end"):
            value = item[field]
            if value is None:
                continue
            try:
                parsed[field] = date.fromisoformat(value)
            except ValueError:
                reasons.append(f"{field} {value!r} does not parse as a date")
                continue
            if not semester_start <= parsed[field] <= semester_end:
                reasons.append(
                    f"{field} {value} is outside the semester "
                    f"({semester_start} to {semester_end})"
                )

        confirmed = item.get("confirmed_by_user", False)
        if item["date_start"] is None and item["date_end"] is None:
            if not confirmed:
                reasons.append("no date found; confirm against the PDF")
        elif (item["date_start"] is None) != (item["date_end"] is None):
            reasons.append("only one of date_start / date_end is set")
        start, end = parsed.get("date_start"), parsed.get("date_end")
        if start and end and end < start:
            reasons.append("date_end is before date_start")

        if id_counts[item["id"]] > 1:
            reasons.append(f"duplicate id {item['id']!r}")

        quote = normalize(item["source_text"])
        if quote not in normalized_text:
            reasons.append("source_text is not a verbatim quote from the text")
        # A user-confirmed date came from the PDF, not from the quote.
        if not confirmed:
            if start and not date_in_quote(start, quote):
                reasons.append(f"date_start {start} is not written in source_text")
            # A computed week end is never written in the quote.
            if (
                end
                and end != start
                and not item.get("date_end_computed")
                and not date_in_quote(end, quote, start)
            ):
                reasons.append(f"date_end {end} is not written in source_text")

        item["needs_review"] = bool(reasons)
        item["review_reasons"] = (
            ["confirmed by user"] + reasons if confirmed else reasons
        )
    return items


def recheck(name: str) -> dict:
    """Re-run the checks on the saved file (e.g. after hand edits) and
    write the results back. Never calls the model."""
    out_path = PARSED_DIR / f"{name}.json"
    if not out_path.exists():
        raise SystemExit(f"No saved file to recheck: {out_path}")
    result = json.loads(out_path.read_text())
    config = SYLLABI[name]
    fill_week_ends(result["items"], config)
    check_items(result["items"], result["schedule_text"], *semester_bounds(config))
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Re-checked {out_path} (no model call)\n")
    return result


def load_or_extract(name: str) -> dict:
    config = SYLLABI[name]
    out_path = PARSED_DIR / f"{name}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    load_env(ENV_PATH)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing ANTHROPIC_API_KEY in .env")

    text = schedule_text(config)
    items, skipped = skip_by_config(extract_items(config, text), config)
    items = fill_week_ends(items, config)
    items = check_items(items, text, *semester_bounds(config))
    sem_start, sem_end = semester_bounds(config)
    result = {
        "course": config["course"],
        "source_pdf": config["pdf"],
        "extracted_with": MODEL,
        "extracted_on": date.today().isoformat(),
        "semester": {
            "start": sem_start.isoformat(),
            "end": sem_end.isoformat(),
        },
        "schedule_text": text,
        "items": items,
    }
    if skipped:
        result["skipped_by_config"] = skipped
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Extracted and saved to {out_path}\n")
    return result


def print_summary(result: dict, kinds: list[str]) -> None:
    items = result["items"]
    print(f"{result['course']} — {len(items)} items\n")

    # needs_review items first, then by date; undated items sort last.
    ordered = sorted(
        items,
        key=lambda i: (not i["needs_review"], i["date_start"] or "9999"),
    )
    rows = [("", "id", "title", "kind", "dates")]
    for item in ordered:
        if item["date_start"] is None:
            dates = "—"
        elif item["date_start"] == item["date_end"]:
            dates = item["date_start"]
        else:
            dates = f"{item['date_start']} → {item['date_end']}"
        flag = "⚠" if item["needs_review"] else ""
        rows.append((flag, item["id"], item["title"], item["kind"], dates))
    widths = [max(len(r[c]) for r in rows) for c in range(5)]
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())

    print("\nCounts by kind:")
    for kind in kinds:
        print(f"  {kind}: {sum(1 for i in items if i['kind'] == kind)}")

    skipped = result.get("skipped_by_config", [])
    if skipped:
        print(f"\nskipped by config: {len(skipped)}")
        for item in skipped:
            print(f"- {item['id']} ({item['title']}, {item['date_start']})")

    flagged = [i for i in items if i["needs_review"]]
    print(f"\nneeds_review: {len(flagged)}")
    for item in flagged:
        print(f"- {item['id']}")
        for reason in item["review_reasons"]:
            print(f"    {reason}")


def main() -> None:
    args = sys.argv[1:]
    do_recheck = "--recheck" in args
    names = [a for a in args if a != "--recheck"]
    if len(names) != 1 or names[0] not in SYLLABI:
        raise SystemExit(
            f"Usage: python3 scripts/extract_syllabus.py <{'|'.join(SYLLABI)}> "
            "[--recheck]"
        )
    result = recheck(names[0]) if do_recheck else load_or_extract(names[0])
    print_summary(result, kinds_for(SYLLABI[names[0]]))


if __name__ == "__main__":
    main()
