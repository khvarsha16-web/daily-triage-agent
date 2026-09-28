# Daily Triage Agent

## Project Overview

A Telegram bot that merges course syllabi with a personal to-do list into a
daily briefing. The core feature is deadline-aware task prioritization:
the agent reads what's due (from syllabi) and what's on the user's plate
(from their to-do list), reasons about urgency/importance, and delivers a
single daily briefing via Telegram that tells the user what to work on and
when.

## Core Pipeline

1. **Inputs** — Raw source material: course syllabi (PDFs, docs, or
   structured files) and the user's personal to-do list (format TBD).
2. **Parsing** — Extract structured data from raw inputs: assignments,
   due dates, weights/importance, course metadata, and personal tasks
   with their own deadlines/priorities.
3. **Judgment Core** — The prioritization engine. Combines parsed
   coursework and personal tasks into a short, capacity-aware selection
   of what to work on today. Model-based (see "Judgment Core Design"
   below).
4. **Verify-and-Correct** — A validation pass over the judgment core's
   output to catch errors (e.g., misparsed dates, mis-prioritized items)
   before anything reaches the user.
5. **Telegram Delivery** — Formats and sends the daily briefing to the
   user via a Telegram bot.

Status (2026-09-25): Parsing (v0, print-only), Judgment Core
(model-based) and Telegram Delivery exist but are not yet wired together
end-to-end. The judgment core still runs on a hardcoded task list.
Verify-and-Correct is not started.

## Judgment Core Design

The core requirement is **daily capacity-aware task selection**, not
deadline eligibility sorting. Showing every task that's *eligible* to be
must-do recreates the overwhelm this project exists to solve. Given the
full active task list and a daily time budget (~6 hrs), the judgment
core decides which *few* tasks to surface TODAY. That includes pacing:
a multi-day task (e.g. the Moo series — scripts, filming, 2+ stored
reels before Oct 1) can belong on today's list well before its deadline
if today is when its next slice of work needs to happen.

**Current implementation (`scripts/judgment_core.py`, 2026-09-25):**
model-based. Sends today's date, the daily budget, and the full raw task
list (unedited descriptions, nothing pre-scored) to Claude Sonnet 5 via
the Anthropic API, which reasons fresh each run about effort, stakes,
and pacing rather than following fixed thresholds. Returns JSON with a
short Must-Do list (each with `hours_today`, totalled against the budget
with a visible over-budget warning), a Can-Do list framed as a reward
tier for after Must-Do is done, and an Upcoming list of fixed future
events needing no action today — each item with one sentence of
reasoning, and each task in exactly one list (checked in code after
every run). Needs `ANTHROPIC_API_KEY` in `.env`. `TODAY` is the real
current date; the task list is still hardcoded (15 real coursework +
personal tasks, updated by hand 2026-09-25) until parsing and a to-do
source are wired in.

Deterministic work happens in code, not the model (see the "Date Math
Belongs in Code" design insight in `notes/meeting-notes.md`): day gaps,
buffer finish-by targets (incl. `hard_date` + `buffer_days`),
recurring-series post dates, loosely recurring tasks (`last_done` +
`every_days` → days since and next due), the user's stated start dates,
the Moo episode template (below), and overdue detection. These are
passed to the model as "Computed facts" it must not recalculate.
Overdue tasks (date passed, not `completed`) never go to the model —
they're listed in an "Overdue — confirm status" section for the user to
confirm. Completed tasks are dropped entirely; if a remaining task
depends on one, the fact goes in that task's own text.

**Moo episode template (`MOO_EPISODE`):** two stages in order — script
(~30 min), then video generation + editing combined (2-3h, where 3h is
the worst case and already includes AI tool failures, remakes and
text-to-speech restitching, so no multiplier or extra slack). Any stage
can be worked in any chunk size; only the finish date and stage order
matter. Each episode must be fully finished 7-10 days before its
posting date, to leave time to review prior episodes' analytics and
revise. For an episode in progress, the user's own hours remaining
(`next_episode_hours_left`) replace the template estimate. Major
academic deadlines inside the production window are flagged so
production starts earlier.

**`start_after`:** the user's own stated plan not to start a task
before a date. While today is before it, the model gets the fact "User
chose to start this on [date] (N days away), a stated plan," and the
displacement check skips the task.

**Output checks (run in code after every model call):**
- Every active task appears in exactly one list.
- Coursework displacement: flags any `major_academic` task in Can-Do
  while ≥0.5h of budget remains (`MIN_SLICE_HOURS`), limited to tasks
  due within 10 days (`DISPLACEMENT_WINDOW_DAYS` — a guardrail from two
  data points, may need revisiting) and not before their `start_after`.
  On a failure the model gets one correction pass with the remaining
  budget computed in code, and is asked to give that task a slice. It
  does not catch content winning when the budget is genuinely full —
  that's left to model judgment and eval cases.

**Archived: `scripts/judgment_core_rules_v0.py`** — the rule-based v0
(Must-Do / Checkpoint / Can-Do by eligibility thresholds). Kept as a
baseline to compare the model's calls against, and because the eval
cases in `notes/meeting-notes.md` were first tested against it.

**Evaluation approach** (per Prof. Cavallo): write eval cases where the
right answer is already known, then check whether the agent agrees.
Cases are logged under "Eval Cases" in `notes/meeting-notes.md`.

## Next Session

Spec designed 2026-09-25 but **not built yet**:

- **Blackout window, Oct 1-12:** pauses all NEW content production
  (script/video/editing). Does not affect already-scheduled posting
  dates (Oct 1, 6, 8, etc.), which stay fixed. The Agentic AI
  mini-project presentation (Oct 12) counts as coursework and falls
  inside this window like the exams.
- ~~**Revised effort model**~~ — built 2026-09-28 (see "Moo episode
  template" above). The splittable / not-splittable distinction was
  dropped entirely: every stage can be done in any chunk size.
- **Weekly capacity structure:** weekdays get a 6-hour budget covering
  everything (coursework, postings, scripting, video/editing). Weekends
  get a 2-3 hour budget, also general-purpose, not reel-exclusive.
  Not built — `DAILY_BUDGET_HOURS = 6` is still a single constant for
  every day.
- **Open decision, not yet resolved:** whether to surface upcoming
  episodes (ep 5, ep 6) before the current one is marked done, given
  the real weekly workload this implies (~7.5-13.5h/week ongoing).
  Needs a deliberate decision next session — not built yet.

Still outstanding from earlier:

- ~~Fix the coursework-displacement bug~~ — done 2026-09-28: prompt
  coursework-first rule, plus a code check that flags any
  `major_academic` task in Can-Do while ≥0.5h of budget remains and
  sends the model one correction pass with the computed remaining
  hours. Content winning when the budget is genuinely full is left to
  model judgment + eval cases (see `notes/meeting-notes.md`, 2026-09-28).
- Rewire `send_briefing.py` to the new `judgment_core` output format
  (see Known Issues).
- Connect syllabus parsing (still v0, unconnected).
- Commit today's work — nothing has been committed all session.

## Standing Rules

- Always show the user the diff/changes and ask for confirmation before
  running `git commit` or `git push`. Never commit or push automatically.
  Commit and push are separate actions — each needs its own explicit
  confirmation, even if both are requested in the same message.

## Known Issues

- **`send_briefing.py` is broken by the judgment core switch**: it still
  imports `classify` (and the Must-Do / Checkpoint / Can-Do format) from
  `judgment_core`, which no longer exists in the model-based version.
  Needs updating to call `get_judgment()` and format the new
  must_do / can_do JSON before the Telegram briefing works again.
- **Overdue-phrasing bug (v0 only, now in `judgment_core_rules_v0.py`)**:
  once a task's due date
  is in the past, the reason text still reads like it's upcoming (e.g.
  "Due in -13 day(s) — too close to defer.") instead of flagging it as
  overdue. Found while testing that Must-Do/Can-Do labels correctly
  shift over time as `TODAY` advances (confirmed 2026-09-22: they do,
  via the existing threshold rules, no new rule needed). Not fixed yet —
  needs addressing before this runs unattended day-over-day, since
  overdue tasks will otherwise read confusingly in the daily briefing.
  The model-based core generates its reasoning fresh, so it may not
  have this problem — check with a `TODAY` past some due dates before
  closing this out.

## Session Log

- 2026-09-19: Project setup. Created this CLAUDE.md, initialized git repo,
  created `notes/` (gitignored, for working/meeting notes). No feature
  code written yet.
- 2026-09-22: Added `data/` (gitignored, for raw syllabi) and
  `scripts/parse_syllabus.py` (v0 PDF text/date/deadline extraction,
  print-only). Built the first working version of the judgment core
  (`scripts/judgment_core.py`) — rule-based, not hardcoded prose: task
  attributes (due date, fixed-time, high-stakes, effort hours,
  recurring-setup) feed rules that sort into Must-Do / Checkpoint /
  Can-Do with generated reasoning. Tested against 10 real personal
  tasks. Corrected twice based on own judgment: (1) added hard-action-
  date buffer logic after catching that "due by X" doesn't mean "safe
  to do on X" when X is an actual posting day; (2) added a third
  category, Checkpoint, for soft check-in dates that aren't hard
  deadlines. Confirmed the urgency rules naturally shift task status
  over time by re-running with different "today" dates (tested Oct 7/8
  against the Predictive proposal). Connected the judgment core to the
  Telegram bot — first real daily briefing sent end-to-end. Known issue
  logged, not fixed: overdue tasks show negative day counts with
  confusing phrasing (see Known Issues above). Professor confirmed via
  email: keep the personal 3-syllabus scope, start core logic now
  rather than waiting for Classes 5-6; the "extension" ideas are
  future-work material for the presentation.
- 2026-09-25: Renamed the project to Daily Triage Agent throughout
  (CLAUDE.md title, GitHub repo `khvarsha16-web/daily-triage-agent`,
  Telegram test message). Logged the first formal eval case in
  `notes/meeting-notes.md`: the ANT series full compile (due Sep 30) was
  called Can-Do by v0 (8 days of runway, ~1hr effort) but should be
  Must-Do, because Sep 30 is the actual posting day — a hard action
  date, not a soft deadline — so it needs 2 days of buffer beforehand;
  fixed in v0 with a `hard_date` flag + `buffer_days` so effective
  runway is days_left − buffer_days. Switched the judgment core from
  rule-based to model-based (Claude Sonnet 5 via the Anthropic API,
  reasoning fresh each run instead of following thresholds) to deliver
  capacity-aware daily selection — this resolves the Sep 22 "sorts by
  eligibility, not capacity" correction. Rule-based version archived
  as `scripts/judgment_core_rules_v0.py`. Added `requirements.txt`.
  Later the same day: replaced the stale Sep 22 task list with 15 real
  current tasks and `TODAY = date.today()`; added the Upcoming list,
  per-task hour estimates with a budget total, and a facts-only prompt
  rule (after a hallucination eval case); moved all date math into code
  as "Computed facts" (after two date-math eval cases); added a buffer
  rule that overrides effort-based reasoning, code-side overdue
  detection with a confirm-status list, a one-list-per-task output
  check, and the Moo episode production template. Verified the Oct 7
  test: the Predictive proposal now escalates to Must-Do. Logged a new
  eval case (coursework displaced by content work, with a fabricated
  justification) — not fixed. See "Next Session" for what's designed
  but not built.
