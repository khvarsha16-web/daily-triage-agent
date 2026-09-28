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

Still using a hardcoded task pool for now — no syllabus parsing wired
in yet. Prints to stdout only; nothing is saved or sent to Telegram."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import anthropic

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

MODEL = "claude-sonnet-5"
DAILY_BUDGET_HOURS = 6
# Smallest useful study slice: coursework left in Can-Do while at least
# this much budget remains counts as displaced.
MIN_SLICE_HOURS = 0.5
# The displacement check only applies to coursework due within this many
# days. A guardrail from two data points (see notes/meeting-notes.md,
# 2026-09-28): the midterm at 8 days must be caught; tasks at 12-16 days
# shouldn't be forced into Must-Do.
DISPLACEMENT_WINDOW_DAYS = 10

# Real current date. The task pool below is still hardcoded (updated
# by hand 2026-09-25) until syllabus parsing and a real to-do source are
# wired in, so it will drift out of date as tasks get done.
TODAY = date.today()

TUE, THU, SAT = 1, 3, 5  # date.weekday() values


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

    first_post: date
    weekdays: tuple[int, ...]
    episodes_done: int
    template: EpisodeTemplate | None = None
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


TASKS = [
    # Coursework
    Task(
        "Predictive exam — Oct 10, 11:30am. 5 lectures already listened "
        "to, need studying/review before exam. Start prep after the "
        "Prescriptive midterm on Oct 3.",
        due=date(2026, 10, 10),
        start_after=date(2026, 10, 4),
        major_academic=True,
    ),
    Task(
        "Predictive final project proposal (group) — due Oct 14. "
        "Multi-step group deliverable needing early coordination, a "
        "mid-point check, and a final check before submission — no fixed "
        "checkpoint dates given, that pacing is for the model to decide. "
        "Group coordination is mostly messaging the group and takes about "
        "15-20 minutes.",
        due=date(2026, 10, 14),
        major_academic=True,
    ),
    Task(
        "Agentic AI mini-project — ongoing work, presentation Oct 12. "
        "Needs parallel time blocks for both continued building and "
        "presentation prep/testing.",
        due=date(2026, 10, 12),
        major_academic=True,
    ),
    Task(
        "Prescriptive midterm exam — Oct 3. 5 lectures + 2 HWs to review, "
        "plus mock practice questions. I'm starting midterm prep on Tue "
        "Sep 29 by my own choice, not today.",
        due=date(2026, 10, 3),
        start_after=date(2026, 9, 29),
        major_academic=True,
    ),
    Task(
        "Prescriptive HW2 (group) — due Sep 27. Assignment submissions "
        "should be treated as needing ~2 days buffer before the actual "
        "deadline.",
        due=date(2026, 9, 27),
        buffer_days=2,
        completed=True,
    ),
    # Personal
    # Eval case 2026-09-25: the compiled video posts Sep 30, a hard action
    # date, so the compile needs a 2-day buffer before it.
    Task(
        "ANT series — compile full video (ep 1-12) by Sep 29.",
        due=date(2026, 9, 29),
        hard_date=date(2026, 9, 30),
        buffer_days=2,
        completed=True,
    ),
    Task(
        "ANT series — episode 12, due Sep 29, needs CTA check.",
        due=date(2026, 9, 29),
    ),
    Task(
        "Post the ANT compiled video on Sep 30. The video is finished and "
        "only posting remains.",
        due=date(2026, 9, 30),
    ),
    Task(
        "Moo series — change handle name and bio, by Sep 30.",
        due=date(2026, 9, 30),
    ),
    Task(
        "Moo series — episode 1 posts Thu Oct 1, then recurring every "
        "Tue/Thu after. Episodes 1-4 fully complete (script, video, "
        "editing done). Episode 5: script and video generation done, "
        "roughly 2h of editing left.",
        series=Series(
            first_post=date(2026, 10, 1),
            weekdays=(TUE, THU),
            episodes_done=4,
            template=MOO_EPISODE,
            next_episode_hours_left={"video generation + editing": 2},
        ),
    ),
    Task(
        "Personal appointment — Oct 5 (Monday), 11am. One-off date, not "
        "recurring (these happen roughly every 2 weeks but not on a fixed "
        "day).",
        due=date(2026, 10, 5),
    ),
    Task(
        "Volunteering signup — Sep 26, 10-11:30am.",
        due=date(2026, 9, 26),
        completed=True,
    ),
    Task(
        "Buy shoes — no fixed date, do on a weekend soon.",
        weekend=True,
        completed=True,
    ),
    Task("Grocery run — no fixed date, zero onions needed."),
    Task(
        "Talk to parents — recurring roughly every 3 days.",
        last_done=date(2026, 9, 27),
        every_days=3,
    ),
]


def fmt_date(d: date) -> str:
    return d.strftime("%a %b %d").replace(" 0", " ")


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
    return task.due is not None and task.due < TODAY and not task.completed


def computed_facts(task: Task) -> list[str]:
    facts = []
    if task.major_academic:
        facts.append("MAJOR ACADEMIC task (exam or big submission).")
    if task.due:
        facts.append(f"Date: {describe_gap(task.due)}.")
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
        episode = 1
        while series_post_date(s, episode) < TODAY:
            episode += 1
        facts.append(
            f"Next post in the series: episode {episode}, "
            f"{describe_gap(series_post_date(s, episode))}."
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
and pacing yourself; nothing has been pre-scored for you).

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
    {"id": <the task's number from the input list>, "task": "<short task \
name, matching the input>", "hours_today": \
<number: your honest estimate of hours this task needs TODAY — today's slice \
only, not the whole task>, "reasoning": "<one fresh sentence: why this, why \
today, referencing the actual date/effort/stakes involved>"}
  ],
  "can_do": [
    {"id": <task number>, "task": "<short task name>", "hours_today": \
<number: honest estimate of the hours today's slice would take if it were \
done today>, "reasoning": "<one sentence on why it is \
genuinely fine to wait today — cite its actual deadline, remaining effort, \
and roughly when it will need to start. Same rigor as a must_do reason; not \
a generic 'not urgent'>"}
  ],
  "upcoming": [
    {"id": <task number>, "task": "<short task name>", "when": "<date/time \
as given in the task \
list>", "reasoning": "<one sentence: why no action is needed today>"}
  ]
}

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

Every task number in the input must appear exactly once across the three \
lists — never zero times, never twice. A task is one unit: do not split it \
into parts or list it in two categories, even when it has several aspects \
(e.g. a recurring series with a finished upcoming post AND unfinished future \
work). Pick the single category for the aspect that needs attention soonest \
and mention the other aspect in that item's reasoning."""


def active_tasks() -> dict[int, Task]:
    """Tasks the model ranks, keyed by stable 1-based id. Completed tasks
    are dropped; overdue ones go to the confirm-status list instead, so
    the model never gets a chance to assume they're done."""
    return {
        i + 1: task
        for i, task in enumerate(TASKS)
        if not task.completed and not is_overdue(task)
    }


def overdue_tasks() -> list[Task]:
    return [task for task in TASKS if is_overdue(task)]


def build_user_message() -> str:
    lines = []
    for task_id, task in active_tasks().items():
        lines.append(f"{task_id}. {task.description}")
        facts = computed_facts(task)
        if facts:
            lines.append(f"   Computed facts: {' '.join(facts)}")
    task_list = "\n".join(lines)
    return (
        f"Today's date: {TODAY.strftime('%A, %B %d, %Y')}\n"
        f"Daily time budget: {DAILY_BUDGET_HOURS} hours\n\n"
        f"Full active task list:\n{task_list}"
    )


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


def get_judgment(messages: list[dict]) -> tuple[dict, str]:
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
    DISPLACEMENT_WINDOW_DAYS, while at least MIN_SLICE_HOURS of budget is
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
    ]


def displacement_problems(result: dict) -> list[str]:
    remaining = DAILY_BUDGET_HOURS - must_do_hours(result)
    return [
        f"task {item['id']} ({item['task']}) is in Can-Do with "
        f"{remaining:g}h of budget remaining"
        for item in displaced_coursework(result)
    ]


def correction_message(result: dict) -> str:
    total = must_do_hours(result)
    remaining = DAILY_BUDGET_HOURS - total
    displaced = "\n".join(
        f"- task {item['id']} ({item['task']})"
        for item in displaced_coursework(result)
    )
    return (
        "Code check on your answer (computed in code — these numbers are "
        "correct, do not recalculate them):\n"
        f"- Your must_do totals {total:g}h, so {remaining:g}h of the "
        f"{DAILY_BUDGET_HOURS}h budget remains today.\n"
        "- These MAJOR ACADEMIC tasks are in can_do while that budget "
        f"remains:\n{displaced}\n\n"
        "Study work can be split, so move each of these into must_do with a "
        f"partial slice for today — all slices together fitting within the "
        f"remaining {remaining:g}h — and set hours_today to the slice, not "
        "the full task. Keep the rest of must_do unchanged. Return the full "
        "corrected JSON in the same shape."
    )


def coverage_problems(result: dict) -> list[str]:
    """Check that every active task appears in exactly one list."""
    seen: dict[int, list[str]] = {}
    for category in ("must_do", "can_do", "upcoming"):
        for item in result.get(category, []):
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


def print_judgment(result: dict, first_pass: dict | None = None) -> None:
    print(
        f"Daily Briefing — {TODAY.strftime('%A, %B %d, %Y')} "
        f"(budget: {DAILY_BUDGET_HOURS}h)\n"
    )

    if first_pass is not None:
        print("--- First pass failed a check; sent back for correction ---")
        print_check_failures(first_pass)
        print("--- Corrected output below ---\n")

    overdue = overdue_tasks()
    if overdue:
        print("=== OVERDUE — CONFIRM STATUS ===")
        for task in overdue:
            print(f"- {task.description}")
            print(
                f"    Date was {describe_gap(task.due)}, and it isn't marked "
                "completed. Done, still to do, or drop?"
            )
        print()

    must_do = result.get("must_do", [])
    total_hours = must_do_hours(result)

    print("=== MUST-DO ===")
    for item in must_do:
        print(f"- {item['task']} (~{item['hours_today']}h)")
        print(f"    {item['reasoning']}")
    print(f"  Total: {total_hours}h of {DAILY_BUDGET_HOURS}h budget")
    if total_hours > DAILY_BUDGET_HOURS:
        print(
            f"  ⚠ OVER BUDGET by {total_hours - DAILY_BUDGET_HOURS}h — "
            "Must-Do doesn't fit in today's budget."
        )

    print("\n=== CAN-DO (reward tier — after Must-Do is done) ===")
    for item in result.get("can_do", []):
        print(f"- {item['task']} (~{item.get('hours_today', '?')}h)")
        print(f"    {item['reasoning']}")

    print("\n=== UPCOMING (no action needed today) ===")
    for item in result.get("upcoming", []):
        print(f"- {item['task']} — {item['when']}")
        print(f"    {item['reasoning']}")

    print_check_failures(result)


def main() -> None:
    load_env(ENV_PATH)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing ANTHROPIC_API_KEY in .env")

    messages = [{"role": "user", "content": build_user_message()}]
    result, text = get_judgment(messages)

    # Remaining budget depends on the model's own Must-Do picks, so it can
    # only be computed after a first pass. If coursework was displaced,
    # hand the model the computed numbers and ask once for a correction.
    first_pass = None
    if displaced_coursework(result):
        first_pass = result
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": correction_message(result)},
        ]
        result, _ = get_judgment(messages)

    print_judgment(result, first_pass)


if __name__ == "__main__":
    main()
