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

Status (2026-10-01): **the full pipeline runs end to end.**
- **Parsing:** all three syllabi (Predictive, Prescriptive, Agentic AI)
  are parsed once by `scripts/extract_syllabus.py` and merged with an
  overlay per course. One-off personal tasks come from a weekly text
  file via a manual import (`scripts/import_personal_tasks.py`). Still
  hand-typed in `judgment_core.py`: the Moo production and posting
  tasks, the recurring "talk to parents" task, and the Agentic AI Final
  Presentation (date not known yet).
- **Judgment Core:** model-based, with code-side checks and a single
  correction retry (a first form of Verify-and-Correct). Every API call
  prints its tokens and estimated cost.
- **Telegram Delivery:** working again — `scripts/send_briefing.py`
  sends a short version of the briefing (see "Telegram Delivery").
- **Personal import:** first real import run and accepted 2026-10-01
  (the Oct 5 appointment). Re-run after each edit of the task file.

**Personal-data dependency:** `judgment_core.py` loads
`data/parsed/predictive.json`, `prescriptive.json` and `agentic.json` at
import time. Those files are gitignored, and so are the syllabus PDFs
they're extracted from (`data/` holds personal course documents). A
fresh clone must put the three PDFs in `data/` and run the extractor
for each — `python3 scripts/extract_syllabus.py <predictive|prescriptive|agentic>`
— before the
judgment core will start; otherwise it exits with a "Missing
data/parsed/…json" message. `data/parsed/personal.json` is optional:
without it the core prints a note and runs with no file-based personal
tasks.

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
current date. Coursework comes from the parsed syllabi plus overlays,
and one-off personal tasks from the weekly file (both below); the rest
(Moo, "talk to parents", the Agentic AI Final Presentation) is
hand-typed in `TASKS`.

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

**Moo posting is split from production.** The production task's
horizon date is the current episode's ideal finish date. Posting is a
separate task generated in code for the *next* post only (Tue/Thu from
Thu Oct 1), with computed facts for the episode number, days until it,
and whether that episode's production is done; ~15-30 min total across
three platforms. Once a post date passes, the following post replaces
it — **a skipped post is not tracked** and never shows as overdue.
`episodes_posted` on the series marks posts already made, so a post
done earlier on its own posting day moves the task to the following
episode (episode 1 posted Oct 1, so the next is episode 2, Tue Oct 6).

**`start_after`:** the user's own stated plan not to start a task
before a date. While today is before it, the model gets the fact "User
chose to start this on [date] (N days away), a stated plan," and the
displacement check skips the task.

### Syllabus parsing (`scripts/extract_syllabus.py`)

Model extraction plus code checks, chosen over improving the regex
parser because flattened PDF tables (wrapped cells, stacked rows,
merged columns) can't be reconstructed line by line. The schedule
section is sent once to Claude Sonnet 5 with a strict JSON schema
(`output_config.format`): each item has `id`, `course`, `kind`
(lecture, project_work, deliverable, exam, break), `title`,
`date_start`, `date_end` and a verbatim `source_text` quote. The model
may not estimate hours, rate importance, or add anything not in the
text; unclear dates come back null rather than guessed.

Code then checks every item: dates parse and fall inside the semester
(Aug 24 to the syllabus's own end date — see below), ids are unique, `source_text` is a verbatim
quote (whitespace-normalized), and the item's date is written in its
own quote. Items with no date are flagged "no date found; confirm
against the PDF". Failing items are kept and marked `needs_review`
with reasons. Items the user confirms by hand get `confirmed_by_user`,
which skips the quote-date and no-date checks.

Parse once: the result is saved to `data/parsed/<name>.json`
(gitignored); later runs read the file and never call the model.
`--recheck` re-runs the code checks after hand edits, still with no
model call. Configured syllabi: `predictive`, `prescriptive`, `agentic`.

**Per-syllabus kinds.** A config can set its own `kinds` list, which
becomes that syllabus's schema enum; the others keep the default five.
Agentic AI uses `session`, `deliverable`, `break`: `session` is a live
weekly class meeting (calendar context, never a task). A
`session_rows` flag adds a prompt note saying which rows are sessions,
which are presentation deliverables (one item per row), and which are
breaks.

**`skip_title_pattern`** drops matching items in code after extraction,
not via the prompt (so the model's "only what's in the text" rule stays
strict). Dropped items are saved under `skipped_by_config` and printed
in the summary, never loaded. Agentic AI uses it for the two Final
Presentations rows.

**Per-syllabus semester end dates.** The shared `SEMESTER_END` is Dec 9
(from the Agentic AI syllabus). A syllabus whose final exam falls in
finals week sets its own `semester_end` — Prescriptive uses Dec 12, for
Exam 3. The end date used is stored in each parsed file.

**Week-end dates are filled in code, not by the model.** Some syllabi
(Prescriptive) give week rows only a "Week of" start date. For those,
the model returns just the start date and a null end; `fill_week_ends`
computes the end (start + 6 days) for lectures, project work and
breaks, and marks the item `date_end_computed`. The date-in-quote check
skips computed ends, since they're never written in the text. Same
principle as the "Date Math Belongs in Code" design insight.

**Ids for unnumbered lectures use the week number** (e.g.
`pres-lecture-week-8`), per the extractor prompt.

### Parsed course data + overlays

`judgment_core.py` turns each parsed Predictive item into a task.
`PREDICTIVE_OVERLAY`, keyed by parsed id, adds the user's own facts,
which win over parsed data on any conflict: booked exam slots (Exam 1
Oct 10, Exam 2 Dec 5, both 11:30am, inside their syllabus windows),
`start_after`, `major_academic` for the exams and project deliverables,
group-deliverable and effort text, completed state, and `block_hours`
for lectures. Exams are due on the booked date, lectures and project
work at the end of their week window, deliverables on their date. A
code check warns if an overlay id isn't in the parsed file or a booked
date falls outside its syllabus window.

**Prescriptive** is merged the same way, with `PRESCRIPTIVE_OVERLAY`
(the hand-typed Prescriptive midterm and HW2 tasks were removed and
their facts moved here): Exam 1 has `start_after` Sep 29 and review-
scope text; all three exams are `major_academic` with "Exam window:
8:00am-11:59pm CT."; weeks 1-5 lectures and HW1-HW2 are completed.
- **Homework 2-day buffer:** HW1-HW6 carry "Due by 11:59 PM CT; up to 12
  hours late for partial credit." and `buffer_days=2` — submissions
  need 2 days of lead time, the same principle as the ANT compile's
  hard-date buffer, so each gets a "finish-by target" computed fact.
- **Week-labeled lectures:** unnumbered lectures are titled "Week N
  lecture: …" in the task text (and the footer), so the text itself says
  it's a lecture. Prescriptive lectures (weeks 7-10, 12, 13, 15) have
  `block_hours=3`, same as Predictive.

**Optional tasks** (`optional=True`, currently only Prescriptive HW6):
there's no real deadline behind them, so they are never put in
Must-Do — an "OPTIONAL" computed fact, an OPTIONAL prompt rule, and a
code check ("optional task in Must-Do", which joins the shared retry)
all enforce it. They get no deliverable check-ins, never show as
overdue, and simply drop off after their date (`is_expired`).

**Agentic AI** (`AGENTIC_OVERLAY`; the syllabus is one clean date per
row, so no week-of or stacked-row handling): the Mini-Project
Presentation (Oct 12, solo, 20%) and Proposal Presentation (Oct 26,
team of three, 10%) are `major_academic` deliverables, so they get
check-ins. The 10 class sessions are calendar context only — a
"class session [date], 7:00-9:45pm, online ([title])" line, and only
within the 14-day horizon. The old hand-typed mini-project task was
removed.
- **Fixed Monday class:** on Mondays inside the Agentic semester and not
  on a break day (so not Nov 23), code puts "Agentic AI class — Mondays
  7:00-9:45pm, online" (2.75h, `fixed: True`) at the top of Must-Do
  after every model call (`with_fixed`). It isn't ranked by the model:
  the prompt gets a "Fixed commitments today" line with the remaining
  budget and is told not to list it. It counts in the Must-Do total and
  so in the over-budget and displacement checks; the exactly-once check
  skips it.
- **Final Presentation** stays hand-typed until the random date (Nov 30
  or Dec 7) is known: due Nov 30, `major_academic`, deliverable kind
  (check-ins Nov 20, 27, 29, 30), `start_after` Oct 13 and
  `hidden_until` Oct 13. `hidden_until` is needed because `start_after`
  alone would bring it inside the 14-day horizon from Sep 29; with it,
  the task is fully hidden (footer only) until Oct 13.

**Breaks are context only.** Thanksgiving (Predictive) and Fall Break
(Prescriptive), both Nov 23-29, are not tasks. Each is passed as a
"Calendar context (not tasks)" line — "no classes Nov 23-29" — which
the prompt says to use for pacing and never list in Must-Do, Can-Do or
Upcoming. They drop out once the break is over.

**Deliverable check-ins.** For deliverable-kind tasks with a due date,
code adds a check-in fact on four days: 10, 3 and 1 day before the due
date, and the due date itself ("Check-in: N days until [title] is due,
confirm status"; on the due date: "…is due today, confirm you're on
track to submit. This is a status confirmation, not new work — if it's
not ready, that's what today's check-in is for."). These are facts, not
new work blocks; the model places them with the normal category rules.

### Weekly personal task file (`scripts/import_personal_tasks.py`)

`data/personal_tasks.txt` (gitignored): one freeform task per line;
blank lines and `#` lines are skipped. Run the import by hand after
editing; it's never automatic.
- **Change-only re-import:** lines identical to a confirmed line in
  `data/parsed/personal.json` keep their result and aren't re-reviewed
  or sent to the model. Only new or edited lines go to Claude Sonnet 5
  (strict JSON schema), which returns a short `title`, a `date` only if
  one is written in the line, and the verbatim `date_text` it came from
  — no hours, no importance, never an invented date. Lines deleted from
  the file are listed and dropped on save. Ids (`personal-<slug>`) are
  made in code, unique.
- **Weekday rule:** a bare weekday is the nearest occurrence on or after
  the import date (today counts); "next [weekday]" is that date + 7.
- **Code checks** (`check_item`): `date_text` is in the line; a written
  date (`Oct 5`, `10/3`) matches, and agrees with any weekday written in
  the line; a weekday expression resolves to exactly the rule's date AND
  (checked separately) falls on that weekday; today/tomorrow match;
  anything else ("this weekend") is flagged to confirm by hand; a line
  that looks dated but got no date is flagged; past dates are flagged
  as "will show as overdue".
- **Batch review before saving:** each line next to its title, date and
  flags; `a` accepts all (asks again if anything is still flagged),
  `e N` edits a title or date (a hand-set date is marked
  `date_set_by_user` and skips the date checks), `q` quits without
  saving.
- **Merge:** each confirmed item becomes a `Task` whose description is
  the user's line unedited, with `title` and `due` from the import.
- **Ids** are `personal-<title slug>`, with a leading "personal" in the
  title not repeated (e.g. "Personal appointment" →
  `personal-appointment`).

### 14-day horizon

A task is sent to the model only if its relevant date is within 14 days
(`HORIZON_DAYS`), or it's overdue (overdue tasks are listed separately,
never sent). The relevant date is the earliest of window start,
`start_after` and due date; for the Moo production task, the current
episode's ideal finish date; for a posting task, the post date; for a
recurring task, its next due date. Tasks with no date at all are always
shown. Hidden tasks are left out of the prompt and the exactly-once
check, and a footer printed in code says "N later items not shown
(next: [title], in X days)." 14 is a guardrail chosen from the report
lead time and a crowded Sep 28 output, and may need revisiting.

### Lecture rules

- **End-of-window rule:** an uncompleted lecture with 2 or fewer days
  left in its week window (`LECTURE_END_BUFFER_DAYS`) gets an "END OF
  WINDOW" fact and must be in Must-Do — same pattern as the hard-date
  buffer. A prompt rule gives it the buffer rule's force, and a code
  check enforces it.
- **Lecture block rule:** a lecture is NOT SPLITTABLE — studied in one
  continuous block of exactly 3h (`block_hours=3`). It may wait until
  later in its window, but when it's in Must-Do or Can-Do its
  `hours_today` must be exactly 3h, never a partial slice. A prompt
  rule states this and a code check enforces it.

### Output checks and the single retry

Run in code after every model call:
- Every active task appears in exactly one list.
- **Coursework displacement:** flags any `major_academic` task in
  Can-Do while ≥0.5h of budget remains (`MIN_SLICE_HOURS`), limited to
  tasks due within 10 days (`DISPLACEMENT_WINDOW_DAYS` — a guardrail
  from two data points, may need revisiting) and not before their
  `start_after`. It does not catch content winning when the budget is
  genuinely full — that's left to model judgment and eval cases.
- **Over budget:** Must-Do hours summed in code against the 6h budget.
- **End of window:** a lecture with ≤2 window days left is missing from
  Must-Do.
- **Lecture block:** a lecture in Must-Do or Can-Do isn't exactly its
  3h block.
- **Optional:** an optional task is in Must-Do.
- **Id integrity:** an item's id isn't an active task, or its copied
  `title` doesn't match (see "No-invented-tasks check").
- **Claim grounding:** a `fact` claim has no quote, an unknown
  `source_id`, or a quote that isn't in its source text; still-flagged
  items print a ⚠ UNGROUNDED line under their reasoning.

**One retry per run, shared by every check except the first.** (The
exactly-once check only warns; it doesn't trigger a retry.) If any of
them fails on the first pass, the model gets a single
correction message with every failure and the numbers computed in code
(Must-Do total, budget, overage, remaining hours, its own Must-Do list
with hours). For over budget, the model chooses what moves, but must
reduce content production and personal tasks first, coursework
(`major_academic`) last, never move a post or appointment happening
today or an END OF WINDOW lecture, and never shrink a lecture block; if
the day can't fit without breaking a rule, it must say so rather than
hide the overage. The final output is checked again but never retried a
second time — any remaining failure prints as a warning. A single
retry keeps two fixes from undoing each other.

**Archived: `scripts/judgment_core_rules_v0.py`** — the rule-based v0
(Must-Do / Checkpoint / Can-Do by eligibility thresholds). Kept as a
baseline to compare the model's calls against, and because the eval
cases in `notes/meeting-notes.md` were first tested against it.

**Evaluation approach** (per Prof. Cavallo): write eval cases where the
right answer is already known, then check whether the agent agrees.
Cases are logged under "Eval Cases" in `notes/meeting-notes.md`.

### Token and cost logging

After every model call in `judgment_core.py` (first pass and, if it
fires, the retry), code prints one line — `[first pass] input N tokens,
output N tokens, est. $X` — and the run ends with "API cost this run:
est. $X (N calls)". Priced at Sonnet 5's list rates, $2 / $10 per
million input / output tokens (`INPUT_PRICE_PER_MTOK`,
`OUTPUT_PRICE_PER_MTOK`); thinking tokens are billed as output and are
already in the output count. A call is logged before the refusal and
JSON checks, so a failed call's cost still shows. Printed only, never
saved. A typical run is ~4k in / ~4k out, about $0.05.

## Telegram Delivery (`scripts/send_briefing.py`)

Runs the judgment once. The terminal gets the full briefing exactly as
`judgment_core.py` prints it (reasoning, Upcoming, check warnings, cost
lines), then the Telegram text, then sends it. Telegram is the short
version: today's date, then numbered lists —
- **OVERDUE** (only when there is any): name and "was due [date]", no
  hours (overdue tasks never go to the model, so there's no estimate).
  The name is the first sentence of the task text; lectures are cut to
  course plus label, e.g. "Prescriptive — Week 8 lecture".
- **MUST-DO** and **CAN-DO** (always shown, "(none)" when empty): the
  model's short task name, hours, and "due [date]" when the task has a
  due date (a posting task shows its post date). The Monday class and
  Moo production intentionally show no date.

No reasoning, no Upcoming, and no Must-Do total or over-budget warning
in Telegram — those stay in the terminal.

## Next Session

What's left:

- **README** for the repo.
- **Demo practice** for the presentation.
- **Optional stretch — tap-to-complete:** mark a task done from the
  Telegram message itself, instead of editing code or the task file.

### No-invented-tasks check (spec agreed and BUILT 2026-10-01)

Both parts are built in `judgment_core.py` (`id_integrity_problems`,
`ungrounded_claims`, `build_reasoning`) and tested offline and live; see
the end of this subsection for what the live runs showed.

Goal: the briefing never shows a task that isn't in the input, and no
reason given for a task rests on a fact that isn't in the input. Two
parts, built in order — Part 2 attaches quotes to ids, so it depends
on Part 1. Before this, the only related check was the exactly-once
check, which warns on an unknown id but never compares names and never
triggers the retry.

**Part 1 — id integrity.** Every scheduled item (must_do, can_do,
upcoming; not the code-placed fixed commitments) must have an id that
exists in today's active task list, and its name must match that task.
Matching the model's display name doesn't work (it paraphrases, e.g.
"Prescriptive Exam 1 prep" for the title "Exam 1"), so:
- The input gives each task a title (`short_title(task)`) on its line.
- Each output item has a `title` field the model copies exactly from
  the input; code requires `item["title"] == short_title(task)`. The
  model's own `task` name stays for display.
- A failure (unknown id or title mismatch) joins the shared single
  retry, with each problem listed in the correction message.

**Part 2 — claim grounding.** The free-text `reasoning` field is
replaced by a `claims` list per item; code builds the displayed
reasoning by joining the claim texts (so the model writes claims as
short sentences that read in sequence). Each claim:
- `type: "fact"` — something stated as true about a task. Must carry a
  `quote`: words copied verbatim from the source text. Optional
  `source_id`: defaults to the item's own id; another task's id when
  the fact is about that task (decision 2); `"context"` for the
  calendar-context and fixed-commitment lines.
- `type: "date"` — pure date math (day gaps, weekdays, windows). No
  quote needed; dates are already computed in code.
- `type: "judgment"` — the model's own assessment (e.g. "fits today's
  budget", "front-loading is safer"). No quote needed. The prompt says
  never to relabel a fact as judgment to avoid the check.

The three decisions (confirmed 2026-10-01):
1. **Computed facts can be quoted.** A task's source text is its
   description plus its "Computed facts:" line (MAJOR ACADEMIC,
   OPTIONAL, END OF WINDOW, check-ins, …), exactly as sent.
   Calendar-context and fixed-commitment lines are quotable the same
   way, via `source_id: "context"` (same reasoning: code-made text the
   model was given).
2. **A claim can cite another task's text if it names that task's id**
   in `source_id` (e.g. "today's budget goes to the Prescriptive exam"
   on the Moo item, quoting the exam task).
3. **Exists-check only.** Code checks the quote appears in the source
   text (whitespace-normalized, case-insensitive), the same kind of
   check as syllabus `source_text`. No second model call checks that
   the quote actually supports the claim — logged under Known Issues.

Code flags an item when: it has no claims; a claim has an unknown type;
a fact has no quote; the `source_id` isn't an active task id or
"context"; or the quote isn't in that source's text. Flagged items join
the shared single retry ("replace the quote with exact words from the
source, cite the right `source_id`, or drop the claim — don't relabel a
fact as judgment"). If an item is still flagged after the retry, it
stays scheduled but the terminal shows a ⚠ with the reason under it —
flagged, never silently passed. Telegram has no reasoning, so it's
unaffected.

Built as specced, plus: code capitalizes and punctuates each claim text
before joining (`as_sentence`), since the model often writes claims as
lowercase fragments. Live results (Oct 1): titles copied exactly and
every fact claim quoted real text, so nothing was flagged; the model
quoted computed facts heavily (e.g. "MAJOR ACADEMIC task", "User chose
to start this on…"). Output tokens roughly doubled (~4k → ~6.5-8k per
first pass, ~$0.05 → ~$0.08-0.09). The known gap showed up right away:
a Moo claim said only "video generation and editing" was left while
its (real) quote says video generation is done — the exists-check
passes it.

Upkeep: when the Agentic AI Final Presentation date (Nov 30 or Dec 7)
is assigned, update its `due` in `judgment_core.py`.

### Parked (designed earlier, not on the current list)

- **Blackout window, Oct 1-12:** pause all NEW content production
  (script/video/editing); already-scheduled posts stay fixed. The
  Agentic AI mini-project presentation (Oct 12) counts as coursework.
- **Weekly capacity structure:** 6h budget on weekdays, 2-3h on
  weekends, both general-purpose. `DAILY_BUDGET_HOURS = 6` is still a
  single constant for every day.
- **Open decision:** whether to surface upcoming episodes (ep 5, ep 6)
  before the current one is marked done, given the real weekly workload
  (~7.5-13.5h/week ongoing).
- Replays against the answer key in `notes/meeting-notes.md` still
  disagree on some dates (e.g. Oct 11 draft reminder); they don't carry
  progress between days, so treat them as spot checks.

## Standing Rules

- Always show the user the diff/changes and ask for confirmation before
  running `git commit` or `git push`. Never commit or push automatically.
  Commit and push are separate actions — each needs its own explicit
  confirmation, even if both are requested in the same message.

## Known Issues

- **Claim grounding checks existence, not support** (by design,
  2026-10-01): code confirms each `fact` claim's quote appears in the
  cited task's text, but not that the quote actually supports the
  claim, and the model could still label a fact as `judgment` to skip
  the check. Closing that would need a second model call per run to
  judge claims; deliberately not built. Same root problem as the
  unsourced-claims eval cases in `notes/meeting-notes.md` ("Eval Case
  (summary): unsourced claims across testing"): the model states
  reasons with full confidence whether or not the data backs them, so
  enforcement lives in code — and this code check only goes as far as
  "the quoted words exist".

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
- 2026-09-28: Committed and pushed the Sep 25 work (`43afc52`) after
  rewording personal task details. Built the model-based
  syllabus extractor with code checks and connected the parsed
  Predictive data to the judgment core via an overlay (hand-typed
  Predictive tasks removed). Added: coursework displacement fix with
  `start_after`, a 10-day displacement window, the 14-day horizon, Moo
  posting split from production, the over-budget check with a shared
  single retry (shrink content first, coursework last), deliverable
  check-ins, the lecture end-of-window rule, the 3h lecture block rule,
  and Thanksgiving as context only. Many eval cases and an answer key
  for replays logged in `notes/meeting-notes.md`. Committed and pushed
  as `9bdd8f4`. Later: parsed and merged the Prescriptive syllabus
  (per-syllabus semester end, week-end dates filled in code, week-
  numbered ids) with its overlay, optional-task handling for HW6, a
  2-day homework buffer, and week-labeled lectures.
- 2026-10-01: Parsed and merged the Agentic AI syllabus: per-syllabus
  kinds with a new `session` kind (the 10 class sessions are calendar
  context only, within the 14-day horizon), Final Presentations rows
  dropped in code via `skip_title_pattern`, and an overlay making the
  Mini-Project (Oct 12, solo) and Proposal (Oct 26, team of three)
  presentations `major_academic` deliverables; the hand-typed
  mini-project task was removed. Added the Monday class as a fixed
  Must-Do item placed by code (2.75h, counted against the budget, not
  ranked by the model, skipped on Fall Break), and the hand-typed Final
  Presentation (due Nov 30 until the real date is known, `start_after`
  and new `hidden_until` Oct 13). Added token and cost logging to every
  judgment-core API call, with a run total. Rewired `send_briefing.py`
  to the model-based output and sent a live briefing: short Telegram
  version (numbered OVERDUE / MUST-DO / CAN-DO, task, hours, date),
  full reasoning in the terminal only. Built the weekly personal task
  file and manual import (freeform lines, weekday rule with code checks,
  change-only re-review, batch review before saving), tested on scratch
  copies only; the Oct 5 appointment moved from code into the file.
  Marked done and then removed eight completed hand-typed tasks (four
  ANT items, volunteering, shoes, grocery, Moo handle rename). Moo:
  `episodes_posted` added (episode 1 posted Oct 1, next post is
  episode 2 on Oct 6). Predictive Final Project Proposal kept open
  (coordination done, `start_after` Oct 4). Specced (in this file
  first) and built the no-invented-tasks check: Part 1 id integrity
  (copied titles), Part 2 claim grounding (fact/date/judgment claims,
  exists-check on quotes, shared retry, ⚠ flag); logged the
  existence-not-support gap under Known Issues. Not committed.
