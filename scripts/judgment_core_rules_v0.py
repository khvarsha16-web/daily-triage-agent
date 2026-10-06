#!/usr/bin/env python3
"""ARCHIVED v0 judgment core (rule-based, eligibility-sorting).

Superseded by the model-based judgment_core.py on 2026-09-25 — this
version sorts every task that's *eligible* to be Must-Do into that
bucket, which recreates task overwhelm rather than solving it. Kept
around as a reference: it's what the Eval Cases entries in
notes/meeting-notes.md were tested against, and it's a useful rule-based
baseline to compare the model-based version's calls against.

Original docstring: Crude v0 judgment core: sorts a hardcoded task list
into Must-Do / Checkpoint / Can-Do with a one-sentence reason each.
Pipeline stage 3 (Judgment Core). Checkpoint is for tasks with a soft
check-in point (not a real deadline) that still deserve mild urgency
before that date. Prints to stdout only; nothing is saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

TODAY = date(2026, 9, 22)

URGENT_WITHIN_DAYS = 7          # due this close -> too late to defer
HIGH_EFFORT_HOURS = 15          # this much work needs an early start
HIGH_EFFORT_LEAD_DAYS = 20      # ...if the deadline is within this window
RECURRING_SETUP_LEAD_DAYS = 14  # new recurring commitments need setup time


@dataclass
class Task:
    name: str
    due: date | None = None
    fixed_time: bool = False       # immovable, specific-time commitment
    high_stakes: bool = False      # big chunk of a grade / high personal stakes
    effort_hours: float | None = None
    recurring_setup: bool = False  # kicks off a new recurring obligation
    hard_date: bool = False        # `due` is the actual day the action happens, not a soft deadline
    buffer_days: int = 0           # lead time needed before a hard_date so it isn't done right up against it
    checkpoint_date: date | None = None  # soft check-in point, not a real deadline
    checkpoint_note: str = ""      # what happens at the checkpoint
    context: str = ""              # extra info used in the fallback reason


TASKS = [
    Task(
        name="Prescriptive HW2 (group) — call, edit, submit",
        due=date(2026, 9, 27),
        context="graded homework, group meeting",
    ),
    Task(
        name="Prescriptive exam study (lectures 1-5 + review HW1&2)",
        due=date(2026, 10, 3),
        high_stakes=True,
        context="worth 24% of grade",
    ),
    Task(
        name="Predictive lectures 1-5 (self-paced)",
        due=date(2026, 10, 10),
        effort_hours=20,  # 4 hrs x 5 lectures
        context="exam is pen-and-paper",
    ),
    Task(
        name="Predictive final project proposal (group, post-feedback edits)",
        due=date(2026, 10, 14),
        context="lower weight, shared group effort",
    ),
    Task(
        name="Errand — decide/schedule",
        due=None,
        checkpoint_date=date(2026, 9, 23),
        checkpoint_note="check-in on a set date",
        context="important",
    ),
    Task(
        name="Fixed-time commitment",
        due=date(2026, 9, 23),
        fixed_time=True,
        context="fixed Wed 4pm",
    ),
    Task(
        name="ANT series reel 11",
        due=date(2026, 9, 24),
        context="recurring content deadline",
    ),
    Task(
        name="ANT series reel 12 (final episode)",
        due=date(2026, 9, 29),
        context="recurring content deadline",
    ),
    Task(
        name="ANT series full compile (ep 1-12)",
        due=date(2026, 9, 30),
        effort_hours=1,
        hard_date=True,
        buffer_days=2,
        context="Sep 30 is the actual posting day, not a soft deadline; one-time wrap-up, ~1hr edit once reels 11 & 12 are done",
    ),
    Task(
        name="Moo series — write/film/store 2+ reels in advance",
        due=date(2026, 10, 1),
        recurring_setup=True,
        context="kicks off recurring Tue/Thu posting after this",
    ),
]


def classify(task: Task) -> tuple[str, str]:
    days_left = (task.due - TODAY).days if task.due else None

    if task.fixed_time:
        return "Must-Do", f"Fixed, immovable commitment ({task.context})."

    if task.hard_date and days_left is not None:
        effective_days_left = days_left - task.buffer_days
        if effective_days_left <= URGENT_WITHIN_DAYS:
            return "Must-Do", (
                f"Due date is a hard action date (not a flexible deadline) and needs "
                f"{task.buffer_days} day(s) of buffer beforehand — only {effective_days_left} "
                f"effective day(s) of runway, not the raw {days_left}."
            )

    if days_left is not None and days_left <= URGENT_WITHIN_DAYS:
        return "Must-Do", f"Due in {days_left} day(s) — too close to defer."

    if task.high_stakes:
        return "Must-Do", f"High stakes ({task.context}), needs an early start even though the deadline isn't imminent."

    if (
        task.effort_hours is not None
        and task.effort_hours >= HIGH_EFFORT_HOURS
        and days_left is not None
        and days_left <= HIGH_EFFORT_LEAD_DAYS
    ):
        return "Must-Do", f"~{task.effort_hours:g} hours of work needed and only {days_left} days until the deadline — has to start now to fit."

    if (
        task.recurring_setup
        and days_left is not None
        and days_left <= RECURRING_SETUP_LEAD_DAYS
    ):
        return "Must-Do", f"Due in {days_left} days and kicks off a new recurring commitment ({task.context}) — slipping this delays every cycle after it."

    if task.checkpoint_date is not None:
        days_to_checkpoint = (task.checkpoint_date - TODAY).days
        return "Checkpoint", (
            f"Not a hard deadline, but {task.checkpoint_note} on {task.checkpoint_date.isoformat()} "
            f"({days_to_checkpoint} day(s) away) — worth having thought about this or made a decision "
            f"before then, even if it isn't finished."
        )

    if days_left is None:
        return "Can-Do", f"No fixed deadline ({task.context}) — nothing is forcing this yet, but worth deliberately scheduling soon."

    return "Can-Do", f"Due in {days_left} days with {task.context} — there's slack before this needs full attention."


def main() -> None:
    results = [(task, *classify(task)) for task in TASKS]

    for label in ("Must-Do", "Checkpoint", "Can-Do"):
        print(f"\n=== {label} ===")
        group = [r for r in results if r[1] == label]
        group.sort(key=lambda r: (r[0].due is None, r[0].due))
        for task, _, reason in group:
            if task.due:
                date_str = f"due {task.due.isoformat()}"
            elif task.checkpoint_date:
                date_str = f"checkpoint {task.checkpoint_date.isoformat()}"
            else:
                date_str = "no fixed date"
            print(f"- {task.name} ({date_str})")
            print(f"    {reason}")


if __name__ == "__main__":
    main()
