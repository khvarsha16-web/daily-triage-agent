#!/usr/bin/env python3
"""Model-based judgment core (Pipeline stage 3). Replaces the rule-based
v0 (archived as judgment_core_rules_v0.py). Instead of sorting tasks by
deadline *eligibility*, this sends the full raw task pool, today's date,
and a daily time budget to Claude and lets it reason through effort,
priority, and daily allocation itself — returning a short, curated
Must-Do list (not everything eligible) with per-task hour estimates
checked against the budget, a Can-Do list framed as a reward tier for
after Must-Do is done, and an Upcoming list of fixed future events that
need no action today. Date math (day gaps, buffer targets, recurring
dates) and overdue detection happen in code, not in the model; overdue
tasks are listed separately for the user to confirm, never sent to the
model to guess about.

Predictive, Prescriptive and Agentic AI coursework comes from the parsed
syllabi (data/parsed/<name>.json, made by extract_syllabus.py) merged with
the user's own facts in each course's overlay; all other tasks (and the
Agentic AI Final Presentation, whose date isn't known yet) are
hand-typed. The Monday Agentic AI class is a fixed commitment added to
Must-Do in code, not ranked by the model.
Prints to stdout only; nothing is saved or sent to Telegram.

Sample mode (--sample) runs the same prompt, rules and checks on the fake
course and fake personal tasks in sample/parsed/, with today pinned to
SAMPLE_TODAY and none of the hand-typed tasks or the Monday class.

Usage: python3 scripts/judgment_core.py [--sample]"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import anthropic

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

MODEL = "claude-sonnet-5"
# Sonnet 5 list prices, dollars per million tokens. Thinking tokens are
# billed as output and are already counted in usage.output_tokens.
INPUT_PRICE_PER_MTOK = 2.00
OUTPUT_PRICE_PER_MTOK = 10.00
DAILY_BUDGET_HOURS = 6
# Smallest useful study slice: coursework left in Can-Do while at least
# this much budget remains counts as displaced.
MIN_SLICE_HOURS = 0.5
# The displacement check only applies to coursework due within this many
# days. A guardrail from two data points (see notes/meeting-notes.md,
# 2026-09-28): the midterm at 8 days must be caught; tasks at 12-16 days
# shouldn't be forced into Must-Do.
DISPLACEMENT_WINDOW_DAYS = 10
# Only tasks whose relevant date is within this many days are sent to the
# model. A guardrail chosen from the report lead time and the crowded
# Sep 28 output (see notes/meeting-notes.md); may need revisiting.
HORIZON_DAYS = 14
# A lecture with this many days (or fewer) left in its week window, not
# completed, must be in Must-Do — same pattern as the hard-date buffer.
LECTURE_END_BUFFER_DAYS = 2

# Real current date. The task pool below is still hardcoded (updated
# by hand 2026-09-25) until syllabus parsing and a real to-do source are
# wired in, so it will drift out of date as tasks get done.
TODAY = date.today()

# Sample mode: fake data from sample/parsed/ instead of data/parsed/, and a
# pinned "today" so the fake dates never go stale. Read from the command
# line here because TODAY and TASKS are set at import time.
SAMPLE = "--sample" in sys.argv[1:]
SAMPLE_TODAY = date(2027, 3, 10)
if SAMPLE:
    TODAY = SAMPLE_TODAY

MON, TUE, THU, SAT = 0, 1, 3, 5  # date.weekday() values


@dataclass
class Stage:
    name: str
    min_hours: float
    max_hours: float
    # Shown with the template estimate (not with user-given hours left).
    note: str = ""


@dataclass
class EpisodeTemplate:
    """How every episode of a series gets produced. Stages run in order
    and can be worked in any chunk size; the whole episode must be
    finished finish_lead_days before its posting date (a (min, max)
    range) — the lead time is for reviewing prior episodes' analytics
    and revising before posting."""

    stages: tuple[Stage, ...]
    finish_lead_days: tuple[int, int]


@dataclass
class Series:
    """A recurring posting series: episode 1 posts on first_post, then
    episodes post on each of the given weekdays in order."""

    name: str
    first_post: date
    weekdays: tuple[int, ...]
    episodes_done: int
    template: EpisodeTemplate | None = None
    # Episodes already posted. The next post is never one of these, even
    # on its own posting day (e.g. posted earlier today).
    episodes_posted: int = 0
    # Hours left per stage on the first unfinished episode, e.g.
    # {"editing": 2}. Stages not listed are done. None means the episode
    # hasn't been started, so every stage's full template estimate applies.
    next_episode_hours_left: dict[str, float] | None = None


MOO_EPISODE = EpisodeTemplate(
    stages=(
        Stage("script", 0.5, 0.5),
        Stage(
            "video generation + editing",
            2,
            3,
            note="3h is the worst case and already includes AI tool "
            "failures, remakes and text-to-speech restitching — no extra "
            "slack needed",
        ),
    ),
    finish_lead_days=(7, 10),
)


MOO_SERIES = Series(
    name="Moo",
    first_post=date(2026, 10, 1),
    weekdays=(TUE, THU),
    episodes_done=5,
    template=MOO_EPISODE,
    next_episode_hours_left={"video generation + editing": 2},
    episodes_posted=1,  # episode 1 posted Thu Oct 1
)


@dataclass
class Task:
    description: str
    # Deterministic date info, turned into computed facts in code so the
    # model never does date math itself.
    due: date | None = None
    # The day the action actually happens (e.g. a posting day), when that
    # differs from `due`. buffer_days counts back from hard_date if set,
    # otherwise from due.
    hard_date: date | None = None
    buffer_days: int = 0
    series: Series | None = None
    completed: bool = False
    weekend: bool = False
    # Exam or big submission: series production shouldn't compete with it.
    major_academic: bool = False
    # Loosely recurring personal task: when it was last done, and roughly
    # how often it should happen.
    last_done: date | None = None
    every_days: int = 0
    # The user's own stated plan not to start before this date. Respected
    # by the displacement check and passed to the model as a fact.
    start_after: date | None = None
    # The user's own part is done and only check-ins remain. Skipped by
    # the displacement check; check-in facts still fire.
    checkins_only: bool = False
    # Syllabus date range (a lecture's week, an exam window, a break),
    # shown to the model as a computed fact.
    window_start: date | None = None
    window_end: date | None = None
    # Parsed syllabus id, e.g. "pred-exam-1"; None for hand-typed tasks.
    source_id: str | None = None
    # Short name for the "later items not shown" footer; parsed items use
    # their syllabus title.
    title: str | None = None
    # Parsed syllabus kind (lecture, deliverable, exam, ...); None for
    # hand-typed tasks.
    kind: str | None = None
    # Fixed hours of a not-splittable block; when scheduled, hours_today
    # must equal this exactly.
    block_hours: float | None = None
    # No real deadline behind it (e.g. an extra-credit homework): never
    # forced into Must-Do, no check-ins, and it simply drops off after its
    # date instead of showing as overdue.
    optional: bool = False
    # Parsed item the extractor flagged (e.g. no date found).
    needs_review: bool = False
    # Not shown at all before this date (not even inside the horizon) —
    # e.g. a task the user doesn't want to see until they start it.
    hidden_until: date | None = None
    # Generated posting task for a series: always stands for the series'
    # NEXT post on or after TODAY, so its text and facts are computed at
    # run time. Only the next post exists; once its date passes, the one
    # after takes its place, so a skipped post isn't tracked.
    posting: Series | None = None


# --- Parsed syllabus items -------------------------------------------------

PARSED_DIR = (
    Path(__file__).resolve().parent.parent
    / ("sample" if SAMPLE else "data")
    / "parsed"
)


@dataclass
class Overlay:
    """The user's own facts about a parsed syllabus item. These win over
    the parsed data on any conflict (e.g. a booked slot inside an exam
    window)."""

    text: str = ""
    booked: date | None = None
    booked_time: str = ""
    start_after: date | None = None
    # Only check-ins remain: the displacement check skips the task.
    checkins_only: bool = False
    major_academic: bool = False
    completed: bool = False
    # Fixed hours of a not-splittable block (e.g. a lecture's 3h).
    block_hours: float | None = None
    optional: bool = False
    # Lead time before the due date, same principle as a hard-date buffer.
    buffer_days: int = 0


LECTURE_STUDY = "About 3h to watch and study, self-paced within its week window."
GROUP_DELIVERABLE = "Group deliverable."

PREDICTIVE_OVERLAY = {
    "pred-exam-1": Overlay(
        text="Study for Lectures 2 to 5; I've listened to all of them. Start "
        "prep after the Prescriptive midterm on Oct 3.",
        booked=date(2026, 10, 10),
        booked_time="11:30am",
        start_after=date(2026, 10, 4),
        major_academic=True,
    ),
    "pred-exam-2": Overlay(
        booked=date(2026, 12, 5), booked_time="11:30am", major_academic=True
    ),
    "pred-deliverable-final-project-proposal": Overlay(
        text=f"{GROUP_DELIVERABLE} My part is done; the group submission is "
        "due Oct 14 and only check-ins remain.",
        start_after=date(2026, 10, 4),
        checkins_only=True,
        major_academic=True,
    ),
    "pred-deliverable-preliminary-project-report": Overlay(
        text=GROUP_DELIVERABLE, major_academic=True
    ),
    "pred-deliverable-final-project-report": Overlay(
        text=GROUP_DELIVERABLE, major_academic=True
    ),
    **{
        f"pred-lecture-{n}": Overlay(text=LECTURE_STUDY, block_hours=3)
        for n in range(6, 10)
    },
    **{f"pred-lecture-{n}": Overlay(completed=True) for n in range(1, 6)},
    "pred-deliverable-preliminary-project-proposal": Overlay(completed=True),
    "pred-lecture-project-work": Overlay(completed=True),
    "pred-deliverable-sas-server-access": Overlay(completed=True),
    "pred-deliverable-form-groups": Overlay(completed=True),
    "pred-deliverable-virtual-learning-launchpad": Overlay(completed=True),
}

# Overlay problems found while merging (unknown ids, booked dates outside
# the syllabus window). Kept for inspection, not sent to the model.
EXAM_WINDOW = "Exam window: 8:00am-11:59pm CT."
HW_DUE = "Due by 11:59 PM CT; up to 12 hours late for partial credit."

AGENTIC_OVERLAY = {
    "agai-deliverable-mini-project": Overlay(
        text="Solo build: one small agentic AI tool, presented live in class "
        "in a few minutes (the problem, how you built it, what it does). 20% "
        "of grade. Ongoing work — needs parallel time blocks for both "
        "continued building and presentation prep/testing.",
        major_academic=True,
    ),
    "agai-deliverable-proposal-presentations": Overlay(
        text="Team of three (teams assigned in Class 5). Short pitch "
        "presented live in class: the problem, why it genuinely needs an "
        "agent, the intended tools and approach, and where it lives in the "
        "repository. 10% of grade.",
        major_academic=True,
    ),
}

# From the Agentic AI syllabus's "Class Hours & Location" (outside the
# schedule table, so not parsed): "Mondays 7:00pm - 9:45pm, Online".
AGENTIC_CLASS_TIME = "7:00-9:45pm, online"
AGENTIC_CLASS = f"Agentic AI class — Mondays {AGENTIC_CLASS_TIME}"
AGENTIC_CLASS_HOURS = 2.75

PRESCRIPTIVE_OVERLAY = {
    "pres-exam-1": Overlay(
        text="5 lectures and 2 homeworks to review, plus mock practice "
        f"questions. {EXAM_WINDOW}",
        start_after=date(2026, 9, 29),
        major_academic=True,
        completed=True,
    ),
    "pres-exam-2": Overlay(text=EXAM_WINDOW, major_academic=True),
    "pres-exam-3": Overlay(text=EXAM_WINDOW, major_academic=True),
    # Homework submissions need 2 days of lead time before the due date.
    "pres-deliverable-hw-1": Overlay(text=HW_DUE, buffer_days=2, completed=True),
    "pres-deliverable-hw-2": Overlay(text=HW_DUE, buffer_days=2, completed=True),
    "pres-deliverable-hw-3": Overlay(text=HW_DUE, buffer_days=2),
    "pres-deliverable-hw-4": Overlay(text=HW_DUE, buffer_days=2),
    "pres-deliverable-hw-5": Overlay(text=HW_DUE, buffer_days=2),
    "pres-deliverable-hw-6-optional": Overlay(
        text=HW_DUE, buffer_days=2, optional=True
    ),
    **{f"pres-lecture-week-{n}": Overlay(completed=True) for n in range(1, 6)},
    **{
        f"pres-lecture-week-{n}": Overlay(block_hours=3)
        for n in (7, 8, 9, 10, 12, 13, 15)
    },
}

# FAKE — sample mode only. Made-up facts for the made-up course in
# sample/parsed/sample_course.json, merged the same way as a real overlay.
SAMPLE_OVERLAY = {
    **{f"samp-lecture-{n}": Overlay(completed=True) for n in range(1, 7)},
    **{
        f"samp-lecture-{n}": Overlay(text=LECTURE_STUDY, block_hours=3)
        for n in range(7, 11)
    },
    "samp-deliverable-hw-1": Overlay(completed=True),
    "samp-deliverable-hw-2": Overlay(completed=True),
    "samp-deliverable-hw-3": Overlay(
        text="Problem set, about 2h. Not started.", buffer_days=2
    ),
    "samp-deliverable-hw-4": Overlay(text="Problem set, about 2h.", buffer_days=2),
    "samp-deliverable-extra-credit-hw": Overlay(
        text="Extra credit, about 1h.", optional=True
    ),
    "samp-exam-1": Overlay(
        text="Covers Lectures 1 to 6; I've watched all of them. About 6h of "
        "review left.",
        major_academic=True,
    ),
    "samp-exam-2": Overlay(major_academic=True),
    "samp-deliverable-project-proposal": Overlay(
        text="Solo, one page. About 2h to write. Not started.",
        major_academic=True,
    ),
    "samp-deliverable-final-project-report": Overlay(major_academic=True),
}

OVERLAY_WARNINGS: list[str] = []


@dataclass
class ContextNote:
    """A parsed calendar fact that isn't a task (a break, or a live class
    session). Shown to the model as context only; never listed in Must-Do,
    Can-Do or Upcoming."""

    course: str
    kind: str
    title: str
    start: date
    end: date
    time: str = ""


CONTEXT_NOTES: list[ContextNote] = []
# Each parsed syllabus's semester, by name, as saved in its parsed file.
SEMESTERS: dict[str, tuple[date, date]] = {}


def parsed_item_to_task(item: dict, overlay: Overlay, course: str) -> Task:
    start = date.fromisoformat(item["date_start"]) if item["date_start"] else None
    end = date.fromisoformat(item["date_end"]) if item["date_end"] else None

    title = item["title"]
    week = re.search(r"-lecture-week-(\d+)$", item["id"])
    if item["kind"] == "lecture" and week:
        # Unnumbered lectures (ids by week): say so in the task text itself.
        title = f"Week {week.group(1)} lecture: {title}"
    parts = [f"{course} — {title}{'' if title[-1] in '.!?' else '.'}"]
    if overlay.booked:
        when = f"{fmt_date(overlay.booked)}, {overlay.booked_time}".rstrip(", ")
        parts.append(f"Booked: {when}.")
        if start and not start <= overlay.booked <= end:
            OVERLAY_WARNINGS.append(
                f"{item['id']}: booked {overlay.booked} is outside the "
                f"syllabus window {start} to {end}; using the booked date"
            )
    if overlay.text:
        parts.append(overlay.text)

    if overlay.booked:
        due = overlay.booked
    elif item["kind"] in ("lecture", "project_work"):
        due = end  # self-paced, finish by the end of its week
    else:
        due = start

    return Task(
        " ".join(parts),
        due=due,
        window_start=start if start != end else None,
        window_end=end if start != end else None,
        start_after=overlay.start_after,
        checkins_only=overlay.checkins_only,
        major_academic=overlay.major_academic,
        completed=overlay.completed,
        source_id=item["id"],
        title=title,
        kind=item["kind"],
        block_hours=overlay.block_hours,
        optional=overlay.optional,
        buffer_days=overlay.buffer_days,
        needs_review=item["needs_review"],
    )


def load_parsed_tasks(
    name: str, course: str, overlays: dict[str, Overlay], session_time: str = ""
) -> list[Task]:
    path = PARSED_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run scripts/extract_syllabus.py {name} first."
        )
    parsed = json.loads(path.read_text())
    SEMESTERS[name] = (
        date.fromisoformat(parsed["semester"]["start"]),
        date.fromisoformat(parsed["semester"]["end"]),
    )
    items = parsed["items"]
    known = {item["id"] for item in items}
    for key in overlays:
        if key not in known:
            OVERLAY_WARNINGS.append(f"overlay id {key!r} not in {path.name}")
    tasks = []
    for item in items:
        if item["kind"] in ("break", "session"):
            CONTEXT_NOTES.append(
                ContextNote(
                    course,
                    item["kind"],
                    item["title"],
                    date.fromisoformat(item["date_start"]),
                    date.fromisoformat(item["date_end"]),
                    session_time if item["kind"] == "session" else "",
                )
            )
            continue
        overlay = overlays.get(item["id"], Overlay())
        tasks.append(parsed_item_to_task(item, overlay, course))
    return tasks


def load_personal_tasks() -> list[Task]:
    """Confirmed tasks from the weekly personal task file (made by
    import_personal_tasks.py). The task text is the user's own line,
    unedited. Missing file: warn and run without them."""
    path = PARSED_DIR / "personal.json"
    if not path.exists():
        print(
            f"Note: no {path.name} yet — run scripts/import_personal_tasks.py. "
            "Running without personal tasks from the file.",
            file=sys.stderr,
        )
        return []
    return [
        Task(
            item["source_line"],
            due=date.fromisoformat(item["date"]) if item["date"] else None,
            title=item["title"],
        )
        for item in json.loads(path.read_text())["items"]
    ]


def fmt_date(d: date) -> str:
    return d.strftime("%a %b %d").replace(" 0", " ")


def sample_tasks() -> list[Task]:
    """Sample mode's whole task list: the fake course and the fake
    personal tasks. None of the hand-typed tasks below."""
    return [
        *load_parsed_tasks("sample_course", "Sample 101", SAMPLE_OVERLAY),
        *load_personal_tasks(),
    ]


TASKS = sample_tasks() if SAMPLE else [
    # Coursework — Predictive and Prescriptive come from their parsed
    # syllabi plus the overlays above; everything else is still hand-typed.
    *load_parsed_tasks("predictive", "Predictive", PREDICTIVE_OVERLAY),
    *load_parsed_tasks("prescriptive", "Prescriptive", PRESCRIPTIVE_OVERLAY),
    *load_parsed_tasks(
        "agentic", "Agentic AI", AGENTIC_OVERLAY, session_time=AGENTIC_CLASS_TIME
    ),
    # Not parsed: the date (Nov 30 or Dec 7) is assigned randomly and isn't
    # known yet. Due on the earlier date until it is. Hidden until the
    # user's own start date, Oct 13 (the day after the mini-project).
    Task(
        "Agentic AI Final Presentation — date TBD: Nov 30 or Dec 7, team of "
        "three (assigned Class 5), 40% of grade.",
        due=date(2026, 11, 30),
        start_after=date(2026, 10, 13),
        hidden_until=date(2026, 10, 13),
        major_academic=True,
        kind="deliverable",
        title="Agentic AI Final Presentation",
    ),
    # Personal — one-off tasks come from the weekly file
    # (data/personal_tasks.txt, via import_personal_tasks.py). Moo and
    # recurring tasks stay here: they need structured state a line can't
    # carry.
    *load_personal_tasks(),
    Task(
        "Moo series — episode 1 posted Thu Oct 1, then recurring every "
        "Tue/Thu after. Episodes 1-5 fully complete (script, video, "
        "editing done). Episode 6: script and video generation done, "
        "roughly 2h of editing left.",
        series=MOO_SERIES,
    ),
    # Posting is split from production: one generated task for the next
    # Moo post, recomputed from TODAY on every run.
    Task("", posting=MOO_SERIES),
    Task(
        "Talk to parents — recurring roughly every 3 days.",
        last_done=date(2026, 9, 27),
        every_days=3,
    ),
]


def describe_gap(d: date) -> str:
    days = (d - TODAY).days
    if days == 0:
        return f"{fmt_date(d)} (TODAY)"
    if days == 1:
        return f"{fmt_date(d)} (tomorrow, 1 day away)"
    if days > 0:
        return f"{fmt_date(d)} ({days} days away)"
    return f"{fmt_date(d)} ({-days} day(s) ago — this date has PASSED)"


def series_post_date(series: Series, episode: int) -> date:
    """Posting date of the given episode (1-indexed)."""
    d, count = series.first_post, 1
    while count < episode:
        d += timedelta(days=1)
        if d.weekday() in series.weekdays:
            count += 1
    return d


def fmt_hours(low: float, high: float | None = None) -> str:
    def one(h: float) -> str:
        return f"{round(h * 60)}min" if h < 1 else f"{h:g}h"

    if high is None or low == high:
        return one(low)
    return f"{one(low)}-{one(high)}"


def episode_production_facts(series: Series, episode: int) -> list[str]:
    """Remaining work, finish-by window, and academic conflicts for one
    episode, from the series' template."""
    template = series.template
    post = series_post_date(series, episode)
    min_lead, max_lead = template.finish_lead_days
    ideal_finish = post - timedelta(days=max_lead)
    latest_finish = post - timedelta(days=min_lead)

    hours_left = series.next_episode_hours_left
    stage_parts, low_total, high_total = [], 0.0, 0.0
    for i, stage in enumerate(template.stages):
        note = ""
        if hours_left is None:
            low, high = stage.min_hours, stage.max_hours
            if stage.note:
                note = f" ({stage.note})"
        elif stage.name in hours_left:
            low = high = hours_left[stage.name]
            note = " (user's own estimate of hours left)"
        else:
            stage_parts.append(f"{i + 1}) {stage.name} — DONE")
            continue
        low_total += low
        high_total += high
        stage_parts.append(
            f"{i + 1}) {stage.name} — ~{fmt_hours(low, high)} left{note}"
        )
    facts = [
        f"Episode {episode} remaining work, stages in order: "
        f"{'; '.join(stage_parts)}. Total remaining: "
        f"~{fmt_hours(low_total, high_total)}.",
        f"Episode {episode} must be fully finished (all stages) {min_lead}-"
        f"{max_lead} days before posting, to leave time to review prior "
        "episodes' performance analytics and revise before it posts: "
        f"ideally by {describe_gap(ideal_finish)}, at the latest by "
        f"{describe_gap(latest_finish)}.",
    ]

    conflicts = [
        t
        for t in TASKS
        if t.major_academic
        and not t.completed
        and t.due is not None
        and TODAY <= t.due <= latest_finish
    ]
    if conflicts:
        names = "; ".join(
            f"{t.description.split(' — ')[0]} on {describe_gap(t.due)}"
            for t in conflicts
        )
        facts.append(
            f"Major academic deadline(s) inside episode {episode}'s "
            f"production window (today through {fmt_date(latest_finish)}): "
            f"{names}. Production must start earlier so it doesn't compete "
            "with that coursework close to crunch time."
        )
    return facts


def before_start_after(task: Task) -> bool:
    return task.start_after is not None and TODAY < task.start_after


def is_overdue(task: Task) -> bool:
    return (
        task.due is not None
        and task.due < TODAY
        and not task.completed
        and not task.optional
    )


def is_expired(task: Task) -> bool:
    """An optional task whose date has passed: it just drops off."""
    return task.optional and task.due is not None and task.due < TODAY


def next_post(series: Series) -> tuple[int, date]:
    """The series' next post on or after TODAY that hasn't been posted
    yet: (episode, date)."""
    episode = series.episodes_posted + 1
    while series_post_date(series, episode) < TODAY:
        episode += 1
    return episode, series_post_date(series, episode)


def describe(task: Task) -> str:
    if task.posting:
        episode, _ = next_post(task.posting)
        return (
            f"Post {task.posting.name} episode {episode} on three platforms. "
            "Posting takes about 5-10 minutes per platform, 15-30 minutes "
            "total."
        )
    return task.description


def posting_facts(series: Series) -> list[str]:
    episode, post = next_post(series)
    done = series.episodes_done >= episode
    return [
        f"Post date: episode {episode}, {describe_gap(post)}.",
        f"Episode {episode} production: "
        + (
            f"DONE (episodes 1-{series.episodes_done} are complete)."
            if done
            else f"NOT done (only episodes 1-{series.episodes_done} are "
            "complete)."
        ),
    ]


# Days before a deliverable's due date on which a status check-in fact is
# shown (0 = the due date itself).
CHECK_IN_DAYS_BEFORE = (10, 3, 1, 0)


def lecture_at_window_end(task: Task) -> bool:
    """An uncompleted lecture within LECTURE_END_BUFFER_DAYS of the end of
    its week window."""
    if task.kind != "lecture" or task.window_end is None or task.completed:
        return False
    return 0 <= (task.window_end - TODAY).days <= LECTURE_END_BUFFER_DAYS


def check_in_fact(task: Task) -> str | None:
    """On a deliverable's check-in days, a fact asking the user to confirm
    status. A fact only — not a new work block."""
    if task.kind != "deliverable" or task.due is None or task.optional:
        return None
    days = (task.due - TODAY).days
    if days not in CHECK_IN_DAYS_BEFORE:
        return None
    name = task.title or short_title(task)
    if days == 0:
        return (
            f"Check-in: {name} is due today, confirm you're on track to "
            "submit. This is a status confirmation, not new work — if it's "
            "not ready, that's what today's check-in is for."
        )
    unit = "day" if days == 1 else "days"
    return f"Check-in: {days} {unit} until {name} is due, confirm status."


def computed_facts(task: Task) -> list[str]:
    if task.posting:
        return posting_facts(task.posting)
    facts = []
    if task.major_academic:
        facts.append("MAJOR ACADEMIC task (exam or big submission).")
    if task.due:
        facts.append(f"Date: {describe_gap(task.due)}.")
    if task.window_start:
        facts.append(
            f"Syllabus window: {describe_gap(task.window_start)} through "
            f"{describe_gap(task.window_end)}."
        )
    if task.source_id and task.due is None and task.window_start is None:
        facts.append(
            "No date found in the syllabus (flagged for review, not yet "
            "confirmed)."
        )
    if before_start_after(task):
        facts.append(
            f"User chose to start this on {describe_gap(task.start_after)}, "
            "a stated plan."
        )
    if task.hard_date:
        facts.append(
            f"Hard action date: {describe_gap(task.hard_date)} — the day "
            "this actually has to happen, not a soft deadline."
        )
    if task.buffer_days:
        anchor = task.hard_date or task.due
        target = anchor - timedelta(days=task.buffer_days)
        facts.append(
            f"With the required {task.buffer_days}-day buffer, the "
            f"finish-by target is {describe_gap(target)}."
        )
    if task.series:
        s = task.series
        episode, post = next_post(s)
        facts.append(
            f"Next post in the series: episode {episode}, {describe_gap(post)}."
        )
        next_unfinished = s.episodes_done + 1
        facts.append(
            f"First episode not yet complete: episode {next_unfinished}, "
            f"posts {describe_gap(series_post_date(s, next_unfinished))}."
        )
        if s.template:
            facts.extend(episode_production_facts(s, next_unfinished))
    if task.weekend:
        sat = TODAY + timedelta(days=(SAT - TODAY.weekday()) % 7)
        facts.append(f"Next weekend starts {describe_gap(sat)}.")
    check_in = check_in_fact(task)
    if check_in:
        facts.append(check_in)
    if task.optional:
        facts.append(
            "OPTIONAL: extra credit with no real deadline behind it — never "
            "put this in must_do; can_do at most."
        )
    if task.kind == "lecture":
        facts.append(
            "NOT SPLITTABLE: a lecture is studied in one continuous block; "
            "when scheduled, it gets its full "
            + (f"{task.block_hours:g}h " if task.block_hours else "hours ")
            + "in a single day."
        )
    if lecture_at_window_end(task):
        days = (task.window_end - TODAY).days
        facts.append(
            f"END OF WINDOW: {days} day(s) left in this lecture's window "
            f"(ends {fmt_date(task.window_end)}) and it isn't marked "
            "completed, so it must be in must_do today."
        )
    if task.last_done and task.every_days:
        since = (TODAY - task.last_done).days
        next_due = task.last_done + timedelta(days=task.every_days)
        facts.append(
            f"Last done {fmt_date(task.last_done)}, {since} day(s) ago. "
            f"Roughly every {task.every_days} days, so next due "
            f"{describe_gap(next_due)}."
        )
    return facts

SYSTEM_PROMPT = """You are the judgment core of a personal coursework-and-life \
triage agent. The person you're helping is prone to overwhelm, so your job is \
NOT to list every task that could theoretically count as urgent — it's to \
decide what actually needs their attention today, and protect them from the \
rest until it matters.

Fixed context, always true:
- Coursework and appointments (exams, fixed-time commitments) always outrank \
informal or content-creation tasks.
- Content-posting tasks (reels, series) are low-effort per unit — don't treat \
them as heavy just because they're recurring. This never overrides the buffer \
rule below.
- Never assume a task is already done unless the task list says so.
- Multi-day tasks need to be paced across the days before their deadline, not \
crammed at the end — a task can belong on today's list even if its deadline \
is many days away, if today is when its next slice of work needs to happen \
to stay on pace within the daily time budget.

Only state facts that are present in the task list. Do not state anything as \
true — e.g. that the person is already signed up, has already started, or has \
already finished something — unless the task list actually says so. If \
something is unknown, either leave it out or say plainly that it's unknown.

You will be given: today's date, a daily time budget in hours, and the \
person's full raw task list (unedited descriptions — infer effort, urgency, \
and pacing yourself; nothing has been pre-scored for you). You may also get \
"Calendar context" lines (e.g. a break with no classes, or an upcoming live \
class session). These are not tasks: use them for pacing, but never list \
them in must_do, can_do or upcoming.

FIXED COMMITMENTS: you may get "Fixed commitments today" (e.g. a live \
class). Code has already placed them in must_do with their hours, and they \
count against the budget. They are not weighed against other tasks — do not \
list them yourself; plan the rest of must_do within the remaining budget \
given.

Some tasks come with "Computed facts" lines: day counts, buffer targets, and \
next-occurrence dates for recurring series, calculated in code from today's \
date. These are already correct. Use them exactly as given and do not \
recalculate any date gap, weekday, or recurring date yourself. If you refer \
to when something happens, use these numbers. If a task has no computed \
facts, it has no known date — do not invent one.

BUFFER RULE — takes priority over all effort-based reasoning: if a task's \
computed facts give a finish-by target from a required buffer (e.g. before a \
hard action date like a posting day), that target is the real deadline. \
Judge the task's urgency by the days left until the finish-by target, not \
until the hard date or stated due date, and never push it past the target or \
defer it on the grounds that it's small, quick, or low-effort per unit.

END-OF-WINDOW RULE — same force as the buffer rule: a lecture whose computed \
facts say END OF WINDOW must be in must_do today, regardless of your own \
priority reasoning.

OPTIONAL RULE: a task marked OPTIONAL is never put in must_do, however \
close its date is. At most it goes in can_do.

LECTURE BLOCK RULE: a lecture marked NOT SPLITTABLE is studied in one \
continuous block, unlike splittable work such as series editing. Whenever a \
lecture is in must_do or can_do, its hours_today is its full remaining hours \
(e.g. the whole 3h), never a partial slice spread across days. A lecture \
may still wait until later in its window; when it is scheduled, it is \
scheduled whole.

SERIES PRODUCTION RULE: for a series episode, the computed "must be fully \
finished" date is binding the same way a buffer target is — it is the real \
deadline for all production stages, not the posting date. Stages must be \
done in order (script, then video generation + editing), but every stage \
can be worked in any chunk size and spread across days — only the finish \
date and the stage order constrain scheduling. Use the computed hours \
remaining exactly as given: for a partly done episode they are the user's \
own estimate and replace the template; template maximums are already worst \
cases, so don't add slack or multipliers on top. If \
you allocate production time today, state which stage and how many hours. \
If a major academic \
deadline falls inside the episode's production window, front-load \
production onto earlier days so it doesn't compete with that coursework \
close to crunch time.

COURSEWORK-FIRST RULE: a task marked MAJOR ACADEMIC must not be put in \
can_do in favor of content or personal work while the day's budget still has \
room for its hours today. Allocate coursework needs first; content \
production (scripts, video, editing) only gets hours after coursework needs \
are covered. Never justify deferring coursework by saying the day is full \
unless the must_do hours really do use up the budget. Exception: if a \
task's computed facts say the user chose to start it on a later date, that \
is a stated plan — respect it and don't schedule the task before that date.

Return ONLY a JSON object (no markdown fences, no commentary) with this shape:
{
  "must_do": [
    {"id": <the task's number from the input list>, "title": "<the task's \
[title: ...] copied exactly>", "task": "<short task name, matching the \
input>", "hours_today": \
<number: your honest estimate of hours this task needs TODAY — today's slice \
only, not the whole task>, "claims": [<claims: why this, why today, \
referencing the actual date/effort/stakes involved>]}
  ],
  "can_do": [
    {"id": <task number>, "title": "<[title: ...] copied exactly>", "task": \
"<short task name>", "hours_today": <number: honest estimate of the hours \
today's slice would take if it were done today>, "claims": [<claims: why it is genuinely fine to wait today — \
its actual deadline, remaining effort, and roughly when it will need to \
start. Same rigor as a must_do reason; not a generic 'not urgent'>]}
  ],
  "upcoming": [
    {"id": <task number>, "title": "<[title: ...] copied exactly>", "task": \
"<short task name>", "when": "<date/time \
as given in the task \
list>", "claims": [<claims: why no action is needed today>]}
  ]
}

CLAIMS — each item's reasoning is a short list of claims (usually 1-3). \
Code joins their "text" fields into the sentence the person reads, so write \
each text as a short sentence that reads naturally in sequence. Each claim \
is one of:
- {"type": "fact", "text": "...", "quote": "<words copied exactly from the \
source>", "source_id": <optional>} — anything stated as true about a task \
(its stakes, what's done or not, effort given in the text, a plan the \
person stated, a computed-facts tag like MAJOR ACADEMIC or END OF WINDOW). \
"quote" must be copied character for character from the source: the task's \
own line or its "Computed facts:" line. source_id defaults to the item's own \
task number; if the fact is about ANOTHER task, set source_id to that \
task's number and quote its text; if it comes from the "Calendar context" \
or "Fixed commitments today" lines, set source_id to "context".
- {"type": "date", "text": "..."} — pure date math (a day count, a weekday, \
a window) taken from the computed facts. No quote.
- {"type": "judgment", "text": "..."} — your own assessment (e.g. it fits \
the remaining budget, front-loading is safer). No quote.
Code checks every fact's quote against its source. Never label a fact as \
"judgment" or "date" to avoid that check; if you can't quote a source for a \
fact, don't state it.

Categories:
- must_do: SHORT — only what's genuinely needed today to stay on track \
without causing stress. Do not include every task that is merely eligible; \
narrow it down.
- can_do: work the person could actually do TODAY if there's spare time, \
but that can safely wait. A task with a "by" deadline that can be done any \
time before it (e.g. "change handle and bio by Sep 30") belongs here, not in \
upcoming. These are a reward tier for after must_do is done.
- upcoming: anything whose action is tied to a specific future date (e.g. a \
posting day, an appointment), and recurring tasks that aren't due yet. These \
must NOT appear in can_do. They are neither obligations nor rewards — just a \
heads-up. If the date is today, or it needs preparation today, it belongs in \
must_do instead. A recurring task moves to must_do or can_do on or after its \
computed next-due date.

Estimate hours_today honestly. Do not shrink estimates to make must_do fit \
the daily budget — if the must_do list genuinely needs more than the budget, \
the total should show that.

Each task line starts with its number and "[title: ...]". In every item, \
"id" is that number and "title" is the text inside [title: ...], copied \
character for character — code checks both against the input.

Every task number in the input must appear exactly once across the three \
lists — never zero times, never twice. A task is one unit: do not split it \
into parts or list it in two categories, even when it has several aspects \
(e.g. a recurring series with a finished upcoming post AND unfinished future \
work). Pick the single category for the aspect that needs attention soonest \
and mention the other aspect in that item's claims."""


def relevant_date(task: Task) -> date | None:
    """The earliest date that makes a task worth showing: its window
    start, start_after, or due date — or, for tasks without those, a
    series episode's ideal finish date, a posting task's post date, or a
    recurring task's next due date. None if the task has no date at all."""
    candidates = [task.window_start, task.start_after, task.due]
    if task.posting:
        candidates.append(next_post(task.posting)[1])
    if task.series:
        # Production only: the current episode's ideal finish date. The
        # next post date belongs to the separate posting task.
        s = task.series
        post = series_post_date(s, s.episodes_done + 1)
        if s.template:
            post -= timedelta(days=max(s.template.finish_lead_days))
        candidates.append(post)
    if task.last_done and task.every_days:
        candidates.append(task.last_done + timedelta(days=task.every_days))
    dates = [d for d in candidates if d is not None]
    return min(dates) if dates else None


def within_horizon(task: Task) -> bool:
    """Tasks with no date at all can't be judged by the horizon, so they
    are always shown (unless hidden_until says otherwise)."""
    if task.hidden_until and TODAY < task.hidden_until:
        return False
    rel = relevant_date(task)
    return rel is None or (rel - TODAY).days <= HORIZON_DAYS


def active_tasks() -> dict[int, Task]:
    """Tasks the model ranks, keyed by stable 1-based id. Completed tasks
    are dropped; overdue ones go to the confirm-status list instead, so
    the model never gets a chance to assume they're done; tasks beyond
    the horizon are held back until they come within it."""
    return {
        i + 1: task
        for i, task in enumerate(TASKS)
        if not task.completed
        and not is_overdue(task)
        and not is_expired(task)
        and within_horizon(task)
    }


def hidden_tasks() -> list[Task]:
    """Open tasks held back by the horizon, soonest first."""
    hidden = [
        task
        for task in TASKS
        if not task.completed
        and not is_overdue(task)
        and not is_expired(task)
        and not within_horizon(task)
    ]
    return sorted(hidden, key=relevant_date)


def short_title(task: Task) -> str:
    if task.posting:
        return f"Post {task.posting.name} episode {next_post(task.posting)[0]}"
    return task.title or task.description.split(" — ")[0]


def overdue_tasks() -> list[Task]:
    return [task for task in TASKS if is_overdue(task)]


def agentic_class_today() -> bool:
    """Mondays within the Agentic AI semester, except its break days.
    Never in sample mode, which has no Agentic AI course."""
    if SAMPLE:
        return False
    start, end = SEMESTERS["agentic"]
    on_break = any(
        note.course == "Agentic AI"
        and note.kind == "break"
        and note.start <= TODAY <= note.end
        for note in CONTEXT_NOTES
    )
    return TODAY.weekday() == MON and start <= TODAY <= end and not on_break


def fixed_commitments() -> list[dict]:
    """Must-Do items placed by code, not ranked by the model: always in
    Must-Do on their day, and counted against the budget like any other
    Must-Do item."""
    if not agentic_class_today():
        return []
    return [
        {
            "id": None,
            "fixed": True,
            "task": AGENTIC_CLASS,
            "hours_today": AGENTIC_CLASS_HOURS,
            "reasoning": "Scheduled weekly class — a fixed commitment, not "
            "weighed against other tasks.",
        }
    ]


def with_fixed(result: dict) -> dict:
    """Put today's fixed commitments at the top of the model's Must-Do."""
    model_must_do = [i for i in result.get("must_do", []) if not i.get("fixed")]
    result["must_do"] = fixed_commitments() + model_must_do
    return result


def context_lines() -> list[str]:
    """Breaks until they're over; class sessions only within the horizon."""
    lines = []
    for note in CONTEXT_NOTES:
        when = describe_gap(note.start).split("(", 1)[1].rstrip(")")
        if note.kind == "session":
            if 0 <= (note.start - TODAY).days <= HORIZON_DAYS:
                time = f", {note.time}" if note.time else ""
                lines.append(
                    f"- {note.course}: class session {fmt_date(note.start)}"
                    f"{time} ({note.title}). Starts {when}."
                )
        elif note.end >= TODAY:
            span = (
                f"on {fmt_date(note.start)}"
                if note.start == note.end
                else f"{fmt_date(note.start)} through {fmt_date(note.end)}"
            )
            lines.append(
                f"- {note.course}: no classes {span} "
                f"({note.title.strip(' !')}). Starts {when}."
            )
    return lines


def build_user_message() -> str:
    lines = []
    for task_id, task in active_tasks().items():
        lines.append(f"{task_id}. [title: {short_title(task)}] {describe(task)}")
        facts = computed_facts(task)
        if facts:
            lines.append(f"   Computed facts: {' '.join(facts)}")
    task_list = "\n".join(lines)
    return (
        f"Today's date: {TODAY.strftime('%A, %B %d, %Y')}\n"
        f"Daily time budget: {DAILY_BUDGET_HOURS} hours\n\n"
        f"Full active task list:\n{task_list}"
    ) + context_block()


def context_block() -> str:
    """The fixed-commitment and calendar-context part of the prompt.
    Claims quote it with source_id "context"."""
    message = ""
    fixed = fixed_commitments()
    if fixed:
        fixed_hours = sum(item["hours_today"] for item in fixed)
        message += (
            "\n\nFixed commitments today (already placed in must_do by code "
            "— do not list them):\n"
            + "\n".join(f"- {i['task']}: {i['hours_today']:g}h" for i in fixed)
            + f"\nThat leaves {DAILY_BUDGET_HOURS - fixed_hours:g}h of the "
            f"{DAILY_BUDGET_HOURS}h budget for everything else."
        )
    lines = context_lines()
    if lines:
        message += "\n\nCalendar context (not tasks):\n" + "\n".join(lines)
    return message


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


# Estimated dollar cost of each model call this run, for the run total.
CALL_COSTS: list[float] = []


def log_usage(label: str, usage) -> None:
    """Print one call's tokens and estimated cost, and record the cost."""
    cost = (
        usage.input_tokens * INPUT_PRICE_PER_MTOK
        + usage.output_tokens * OUTPUT_PRICE_PER_MTOK
    ) / 1_000_000
    CALL_COSTS.append(cost)
    print(
        f"[{label}] input {usage.input_tokens:,} tokens, output "
        f"{usage.output_tokens:,} tokens, est. ${cost:.4f}"
    )


def print_cost_total() -> None:
    calls = len(CALL_COSTS)
    print(
        f"\nAPI cost this run: est. ${sum(CALL_COSTS):.4f} "
        f"({calls} call{'s' if calls != 1 else ''})"
    )


def get_judgment(messages: list[dict], label: str) -> tuple[dict, str]:
    """Returns the parsed judgment and the raw response text (needed to
    continue the conversation for a correction pass)."""
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            messages=messages,
        )
    except anthropic.AuthenticationError:
        raise SystemExit(
            "Anthropic API rejected the key in .env — check ANTHROPIC_API_KEY "
            "is correct and active."
        )
    except anthropic.APIConnectionError:
        raise SystemExit("Network error reaching the Anthropic API.")

    # Logged before any exit below: a refused or unparseable call still costs.
    log_usage(label, response.usage)

    if response.stop_reason == "refusal":
        raise SystemExit(f"Model declined to respond: {response.stop_details}")

    text = next(
        (block.text for block in response.content if block.type == "text"), None
    )
    if text is None:
        raise SystemExit(f"No text in model response: {response.content}")

    try:
        return parse_json_response(text), text
    except json.JSONDecodeError:
        raise SystemExit(f"Could not parse model output as JSON:\n{text}")


def must_do_hours(result: dict) -> float:
    return sum(item["hours_today"] for item in result.get("must_do", []))


def displaced_coursework(result: dict) -> list[dict]:
    """Can-Do items for MAJOR ACADEMIC tasks due within
    DISPLACEMENT_WINDOW_DAYS (not before their start_after, and not
    checkins_only), while at least MIN_SLICE_HOURS of budget is
    left after Must-Do. Study work can be split, so even a
    partial slice counts as room. Budget arithmetic is done here, not by
    the model."""
    remaining = DAILY_BUDGET_HOURS - must_do_hours(result)
    if remaining < MIN_SLICE_HOURS:
        return []
    tasks = active_tasks()
    return [
        item
        for item in result.get("can_do", [])
        if item.get("id") in tasks
        and tasks[item["id"]].major_academic
        and (tasks[item["id"]].due - TODAY).days <= DISPLACEMENT_WINDOW_DAYS
        and not before_start_after(tasks[item["id"]])
        and not tasks[item["id"]].checkins_only
    ]


def displacement_problems(result: dict) -> list[str]:
    remaining = DAILY_BUDGET_HOURS - must_do_hours(result)
    return [
        f"task {item['id']} ({item['task']}) is in Can-Do with "
        f"{remaining:g}h of budget remaining"
        for item in displaced_coursework(result)
    ]


def over_budget(result: dict) -> float:
    """Hours by which Must-Do exceeds the daily budget (0 if it fits)."""
    return max(0.0, round(must_do_hours(result) - DAILY_BUDGET_HOURS, 2))


def item_label(item: dict) -> str:
    return "fixed, placed by code" if item.get("fixed") else f"task {item['id']}"


def correction_message(result: dict) -> str:
    """One correction request covering every check the first pass failed
    (displaced coursework and/or over budget), so a single retry fixes
    them together instead of one fix undoing the other."""
    total = round(must_do_hours(result), 2)
    remaining = round(DAILY_BUDGET_HOURS - total, 2)
    parts = [
        "Code check on your answer (computed in code — these numbers are "
        "correct, do not recalculate them):",
        f"- Your must_do totals {total:g}h against a {DAILY_BUDGET_HOURS}h "
        "budget.",
    ]

    displaced = displaced_coursework(result)
    if displaced:
        names = "\n".join(
            f"  - task {item['id']} ({item['task']})" for item in displaced
        )
        parts.append(
            f"- {remaining:g}h of the budget remains, and these MAJOR "
            f"ACADEMIC tasks are in can_do:\n{names}\n"
            "  Study work can be split, so move each of these into must_do "
            "with a partial slice for today — all slices together fitting "
            f"within the remaining {remaining:g}h — and set hours_today to the "
            "slice, not the full task."
        )

    missing = lectures_missing_at_window_end(result)
    if missing:
        names = "\n".join(
            f"  - task {task_id} ({short_title(task)}): "
            f"{(task.window_end - TODAY).days} day(s) left, window ends "
            f"{fmt_date(task.window_end)}"
            for task_id, task in missing
        )
        parts.append(
            "- These lectures are at the end of their window and not "
            f"completed, so they must be in must_do today:\n{names}\n"
            "  Move each into must_do with honest hours_today."
        )

    mismatched = lecture_block_mismatches(result)
    if mismatched:
        names = "\n".join(
            f"  - task {task_id} ({short_title(task)}): scheduled "
            f"{hours:g}h in {category}, block is exactly "
            f"{task.block_hours:g}h"
            for task_id, task, hours, category in mismatched
        )
        parts.append(
            "- These lectures aren't scheduled as their exact full block, "
            f"but lectures are NOT SPLITTABLE:\n{names}\n"
            "  Give each exactly its block hours today, or move it out of "
            "must_do/can_do whole (unless it is END OF WINDOW)."
        )

    optional = optional_in_must_do(result)
    if optional:
        names = "\n".join(
            f"  - task {task_id} ({short_title(task)})"
            for task_id, task in optional
        )
        parts.append(
            "- These tasks are OPTIONAL (no real deadline) but are in "
            f"must_do:\n{names}\n"
            "  Move each out of must_do (to can_do at most)."
        )

    overage = over_budget(result)
    if overage:
        items = "\n".join(
            f"  - {item_label(item)} ({item['task']}): {item['hours_today']:g}h"
            for item in result.get("must_do", [])
        )
        parts.append(
            f"- That is {overage:g}h OVER the budget. Your must_do list with "
            f"hours:\n{items}\n"
            f"  Give a corrected answer whose must_do fits within "
            f"{DAILY_BUDGET_HOURS}h. Every existing rule still applies: the "
            "coursework-first rule, hard finish-by dates (buffer targets, "
            "series finish dates, and lectures marked END OF WINDOW), and "
            "posts or appointments happening today stay in must_do. Fixed "
            "commitments placed by code are already counted and can't move. You "
            "choose what moves out of must_do, and you may shrink hours_today "
            "for work that can be split (never a lecture marked NOT "
            "SPLITTABLE — it stays whole or moves out whole, unless it is END "
            "OF WINDOW). When choosing "
            "what to shrink or move out, reduce content production and "
            "personal tasks first, reduce coursework (MAJOR ACADEMIC) last, "
            "and never move a post or appointment happening today. If the day "
            "can't fit without breaking one of those rules, keep the rule, "
            "say so in the claims of the affected items, and leave the "
            "total over budget — don't hide the overage with estimates "
            "smaller than honest."
        )

    integrity = id_integrity_problems(result)
    if integrity:
        names = "\n".join(f"  - {problem}" for problem in integrity)
        parts.append(
            "- These items don't match a task in the input:\n"
            f"{names}\n"
            "  Every item must use a task number from the input and copy its "
            "[title: ...] exactly. Drop any item that isn't an input task."
        )

    ungrounded = ungrounded_claims(result)
    if ungrounded:
        names = "\n".join(
            f"  - task {item['id']} ({item.get('task')}): {reason}"
            for _, item, reason in ungrounded
        )
        parts.append(
            "- These claims aren't grounded in the input:\n"
            f"{names}\n"
            "  For each: replace the quote with words copied exactly from the "
            "source text, cite the right source_id (another task's number, or "
            '"context"), or drop the claim. Don\'t relabel a fact as judgment '
            "or date to get past this check."
        )

    parts.append(
        "Fix everything above in one answer. Return the full corrected JSON "
        "in the same shape."
    )
    return "\n".join(parts)


def coverage_problems(result: dict) -> list[str]:
    """Check that every active task appears in exactly one list."""
    seen: dict[int, list[str]] = {}
    for category in ("must_do", "can_do", "upcoming"):
        for item in result.get(category, []):
            if item.get("fixed"):
                continue
            seen.setdefault(item.get("id"), []).append(category)
    problems = []
    for task_id in active_tasks():
        categories = seen.pop(task_id, [])
        if not categories:
            problems.append(f"task {task_id} is missing from every list")
        elif len(categories) > 1:
            problems.append(
                f"task {task_id} appears in more than one list: "
                f"{', '.join(categories)}"
            )
    for task_id in seen:
        problems.append(f"unknown task id {task_id} in output")
    return problems


def normalize(text: str) -> str:
    """For quote matching: whitespace collapsed, case ignored."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def task_source_text(task: Task) -> str:
    """A task's quotable text, exactly as sent: its line plus its
    "Computed facts:" line (decision 1: computed facts can be quoted)."""
    facts = computed_facts(task)
    text = describe(task)
    if facts:
        text += f" Computed facts: {' '.join(facts)}"
    return normalize(text)


def build_reasoning(result: dict) -> dict:
    """The displayed reasoning is built in code from the claim texts."""
    for category in ("must_do", "can_do", "upcoming"):
        for item in result.get(category, []):
            if item.get("fixed"):
                continue
            claims = item.get("claims") or []
            texts = (c.get("text", "").strip() for c in claims if isinstance(c, dict))
            item["reasoning"] = " ".join(as_sentence(t) for t in texts if t)
    return result


def as_sentence(text: str) -> str:
    """Capitalized, ending in punctuation, so joined claims read as
    separate sentences."""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def ungrounded_claims(result: dict) -> list[tuple[str, dict, str]]:
    """Part 2 of the no-invented-tasks check: (category, item, reason)
    for every claim that isn't grounded. Exists-check only — the quote
    must appear in its source text; whether it actually supports the
    claim isn't checked (see Known Issues in CLAUDE.md)."""
    tasks = active_tasks()
    context = normalize(context_block())
    found = []
    for category in ("must_do", "can_do", "upcoming"):
        for item in result.get(category, []):
            if item.get("fixed") or item.get("id") not in tasks:
                continue  # fixed: placed by code; unknown id: Part 1's job
            claims = item.get("claims")
            if not claims:
                found.append((category, item, "no claims given"))
                continue
            for claim in claims:
                kind = claim.get("type") if isinstance(claim, dict) else None
                if kind in ("date", "judgment"):
                    continue
                if kind != "fact":
                    found.append((category, item, f"claim has unknown type {kind!r}"))
                    continue
                text = claim.get("text", "")
                quote = normalize(claim.get("quote") or "")
                if not quote:
                    found.append((category, item, f"fact {text!r} has no quote"))
                    continue
                source = claim.get("source_id", item["id"])
                if source == "context":
                    source_text, name = context, "the calendar context"
                elif source in tasks:
                    source_text = task_source_text(tasks[source])
                    name = f"task {source}"
                else:
                    found.append(
                        (category, item, f"fact {text!r} cites unknown source_id "
                         f"{source!r}")
                    )
                    continue
                if quote not in source_text:
                    found.append(
                        (category, item, f"fact {text!r}: quote "
                         f"{claim.get('quote')!r} is not in {name}'s text")
                    )
    return found


def id_integrity_problems(result: dict) -> list[str]:
    """Part 1 of the no-invented-tasks check: every scheduled item (not
    a code-placed fixed commitment) names a real active task — its id
    exists and its copied title matches that task's title exactly."""
    tasks = active_tasks()
    problems = []
    for category in ("must_do", "can_do", "upcoming"):
        for item in result.get(category, []):
            if item.get("fixed"):
                continue
            task = tasks.get(item.get("id"))
            if task is None:
                problems.append(
                    f"{category} item {item.get('task')!r} has id "
                    f"{item.get('id')!r}, which is not a task in the input"
                )
            elif item.get("title") != short_title(task):
                problems.append(
                    f"task {item['id']} in {category}: title "
                    f"{item.get('title')!r} doesn't match the input title "
                    f"{short_title(task)!r}"
                )
    return problems


def lectures_missing_at_window_end(result: dict) -> list[tuple[int, Task]]:
    """Active lectures in the last LECTURE_END_BUFFER_DAYS of their window
    that the model left out of Must-Do."""
    in_must_do = {item.get("id") for item in result.get("must_do", [])}
    return [
        (task_id, task)
        for task_id, task in active_tasks().items()
        if lecture_at_window_end(task) and task_id not in in_must_do
    ]


def lecture_block_mismatches(result: dict) -> list[tuple[int, Task, float, str]]:
    """Lectures in must_do or can_do whose hours_today isn't exactly their
    fixed block (e.g. a partial slice of a not-splittable task)."""
    tasks = active_tasks()
    found = []
    for category in ("must_do", "can_do"):
        for item in result.get(category, []):
            task = tasks.get(item.get("id"))
            if (
                task
                and task.kind == "lecture"
                and task.block_hours
                and item.get("hours_today", 0) != task.block_hours
            ):
                found.append((item["id"], task, item["hours_today"], category))
    return found


def optional_in_must_do(result: dict) -> list[tuple[int, Task]]:
    tasks = active_tasks()
    return [
        (item["id"], tasks[item["id"]])
        for item in result.get("must_do", [])
        if item.get("id") in tasks and tasks[item["id"]].optional
    ]


def print_check_failures(result: dict) -> None:
    problems = coverage_problems(result)
    if problems:
        print("\n⚠ OUTPUT CHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")

    displacement = displacement_problems(result)
    if displacement:
        print(
            "\n⚠ OUTPUT CHECK FAILED: coursework displaced with budget "
            "remaining"
        )
        for problem in displacement:
            print(f"  - {problem}")

    missing = lectures_missing_at_window_end(result)
    if missing:
        print(
            "\n⚠ OUTPUT CHECK FAILED: lecture at end of its window not in "
            "Must-Do"
        )
        for task_id, task in missing:
            print(
                f"  - task {task_id} ({short_title(task)}): "
                f"{(task.window_end - TODAY).days} day(s) left"
            )

    mismatched = lecture_block_mismatches(result)
    if mismatched:
        print(
            "\n⚠ OUTPUT CHECK FAILED: lecture not scheduled as its exact "
            "full block"
        )
        for task_id, task, hours, category in mismatched:
            print(
                f"  - task {task_id} ({short_title(task)}): {hours:g}h in "
                f"{category}, block is exactly {task.block_hours:g}h"
            )

    integrity = id_integrity_problems(result)
    if integrity:
        print("\n⚠ OUTPUT CHECK FAILED: item doesn't match an input task")
        for problem in integrity:
            print(f"  - {problem}")

    ungrounded = ungrounded_claims(result)
    if ungrounded:
        print("\n⚠ OUTPUT CHECK FAILED: claim not grounded in the input")
        for category, item, reason in ungrounded:
            print(f"  - task {item['id']} in {category}: {reason}")

    optional = optional_in_must_do(result)
    if optional:
        print("\n⚠ OUTPUT CHECK FAILED: optional task in Must-Do")
        for task_id, task in optional:
            print(f"  - task {task_id} ({short_title(task)})")


def print_judgment(result: dict, first_pass: dict | None = None) -> None:
    print(
        f"Daily Briefing — {TODAY.strftime('%A, %B %d, %Y')} "
        f"(budget: {DAILY_BUDGET_HOURS}h)\n"
    )

    if first_pass is not None:
        print("--- First pass failed a check; sent back for correction ---")
        print_check_failures(first_pass)
        overage = over_budget(first_pass)
        if overage:
            print(
                f"\n⚠ OUTPUT CHECK FAILED: Must-Do over budget — "
                f"{round(must_do_hours(first_pass), 2):g}h of "
                f"{DAILY_BUDGET_HOURS}h (over by {overage:g}h)"
            )
        print("--- Corrected output below ---\n")

    overdue = overdue_tasks()
    if overdue:
        print("=== OVERDUE — CONFIRM STATUS ===")
        for task in overdue:
            print(f"- {describe(task)}")
            print(
                f"    Date was {describe_gap(task.due)}, and it isn't marked "
                "completed. Done, still to do, or drop?"
            )
        print()

    must_do = result.get("must_do", [])
    total_hours = must_do_hours(result)

    flags: dict[int, list[str]] = {}
    for _, item, reason in ungrounded_claims(result):
        flags.setdefault(id(item), []).append(reason)

    def print_reasoning(item: dict) -> None:
        print(f"    {item['reasoning']}")
        for reason in flags.get(id(item), []):
            print(f"    ⚠ UNGROUNDED: {reason}")

    print("=== MUST-DO ===")
    for item in must_do:
        print(f"- {item['task']} (~{item['hours_today']}h)")
        print_reasoning(item)
    print(f"  Total: {round(total_hours, 2):g}h of {DAILY_BUDGET_HOURS}h budget")
    if total_hours > DAILY_BUDGET_HOURS:
        print(
            f"  ⚠ OVER BUDGET by {round(total_hours - DAILY_BUDGET_HOURS, 2):g}h — "
            "Must-Do doesn't fit in today's budget."
        )

    print("\n=== CAN-DO (reward tier — after Must-Do is done) ===")
    for item in result.get("can_do", []):
        print(f"- {item['task']} (~{item.get('hours_today', '?')}h)")
        print_reasoning(item)

    print("\n=== UPCOMING (no action needed today) ===")
    for item in result.get("upcoming", []):
        print(f"- {item['task']} — {item['when']}")
        print_reasoning(item)

    print_check_failures(result)

    hidden = hidden_tasks()
    if hidden:
        nxt = hidden[0]
        days = (relevant_date(nxt) - TODAY).days
        print(
            f"\n{len(hidden)} later items not shown (next: {short_title(nxt)}, "
            f"in {days} days)."
        )


def main() -> None:
    load_env(ENV_PATH)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing ANTHROPIC_API_KEY in .env")

    if SAMPLE:
        print(
            "SAMPLE MODE — fake data from sample/. Today is pinned to "
            f"{TODAY.strftime('%A, %B %d, %Y')}.\n"
        )
    first_pass, result = judge()
    print()
    print_judgment(result, first_pass)
    print_cost_total()


def needs_retry(result: dict) -> bool:
    """Any check that joins the shared single retry failed."""
    return bool(
        displaced_coursework(result)
        or over_budget(result)
        or lectures_missing_at_window_end(result)
        or lecture_block_mismatches(result)
        or optional_in_must_do(result)
        or id_integrity_problems(result)
        or ungrounded_claims(result)
    )


def judge() -> tuple[dict | None, dict]:
    """Returns (first_pass, final). first_pass is None when no retry ran.

    Budget arithmetic depends on the model's own Must-Do picks, so it can
    only be done after a first pass. If coursework was displaced or
    Must-Do is over budget, the computed numbers go back to the model in
    ONE retry (both problems together if both apply). The final output is
    checked again, but never retried a second time — any remaining
    failure is printed as a warning instead."""
    messages = [{"role": "user", "content": build_user_message()}]
    result, text = get_judgment(messages, "first pass")
    result = with_fixed(build_reasoning(result))
    if not needs_retry(result):
        return None, result

    messages += [
        {"role": "assistant", "content": text},
        {"role": "user", "content": correction_message(result)},
    ]
    final, _ = get_judgment(messages, "retry")
    return result, with_fixed(build_reasoning(final))


if __name__ == "__main__":
    main()
