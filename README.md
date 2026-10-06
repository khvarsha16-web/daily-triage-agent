# Daily Triage Agent

A command-line tool and Telegram bot that turns course syllabi and a
personal to-do list into one short daily briefing: what to work on today,
for how many hours, and why.

## Project layout

- `scripts/` — the code: `judgment_core.py` (the briefing),
  `send_briefing.py` (Telegram), `extract_syllabus.py` and
  `import_personal_tasks.py` (inputs).
  Two older files are also there. `parse_syllabus.py` is the first
  regex-based syllabus parser; its parsing is no longer used, but
  `extract_syllabus.py` and `import_personal_tasks.py` still import the
  `data/` path from it, and the extractor also imports its PDF text
  reader. `judgment_core_rules_v0.py` is the earlier rule-based version
  of the briefing, kept for comparison; nothing imports it.
- `sample/` — fake data for the sample run.
- `requirements.txt` — the two Python packages to install.
- `data/` — private course documents and the personal task file. Not
  in the repository.

## The problem

A semester's deadlines plus a personal to-do list add up to more tasks
than fit in a day. Seeing all of them at once is overwhelming, and
anything without a near deadline ends up in an "if time allows" pile that
never gets sorted. A list of everything that is due soon does not help,
because it is still too long. What is needed each day is a short list
that fits the hours available.

## What it does

Each run produces a briefing for today with:

- **Overdue — confirm status:** tasks whose date has passed and are not
  marked completed.
- **Must-Do:** a short list for today, each item with hours for today and
  the reasons. The hours are totalled against a 6-hour daily budget.
- **Can-Do:** work that could be done today but can wait, with reasons.
- **Upcoming:** items tied to a future date that need nothing today.
- A count of later items that are not shown yet (anything more than 14
  days out).

The full briefing prints in the terminal. `send_briefing.py` also sends a
short version to Telegram: the date and numbered Overdue, Must-Do and
Can-Do lists with task name, hours and due date, without the reasons.

Inputs:

- **Course syllabi.** `extract_syllabus.py` reads a syllabus PDF once and
  saves its schedule as JSON (lectures, deliverables, exams, breaks, with
  dates). Later runs read the saved file.
- **A personal task file.** One freeform line per task.
  `import_personal_tasks.py` gives each line a title and a date (only if
  one is written in the line) and shows the batch for review before
  saving.
- **Facts added in code.** Per-course "overlays" in `judgment_core.py`
  hold what a syllabus does not say: which items are completed, booked
  exam slots, planned start dates, effort notes. A few tasks are typed
  directly in that file.

## How it works

The work is split between code and a model (`claude-sonnet-5` through the
Anthropic API).

**Code computes** everything that has one right answer:

- Day counts to each date, finish-by targets for tasks that need lead
  time, status check-in days for deliverables (10, 3 and 1 day before the
  due date, and the due date), and recurring dates.
- Which tasks are overdue, completed, or beyond the 14-day horizon. None
  of these are sent to the model.
- The Must-Do hours total and the comparison with the daily budget.
- For a recurring content series, whether a major academic deadline falls
  inside the current episode's production window. When one does, code
  flags it and the model is told to start production earlier.
- The checks on the model's answer (listed below).

These results are passed to the model as "Computed facts", which it is
told to use as given and not recalculate.

**The model decides** which tasks get today's hours and how many, and
puts every task in Must-Do, Can-Do or Upcoming. It gives its reasons as
a list of claims. A claim that states a fact about a task must include a
quote copied from that task's text.

**Calls per run:** one model call. If a check fails, the failures and the
numbers computed in code are sent back in one retry. There is never a
second retry; anything still failing after it is printed as a warning.

The syllabus extractor and the personal task import are separate scripts
that make their own model calls, once per syllabus and once per new or
edited task line. A briefing run does not call them.

## How to run it

Requirements: Python 3 (developed on Python 3.9.6) and an Anthropic API
key.

```
pip install -r requirements.txt
```

Create a file named `.env` in the project root:

```
ANTHROPIC_API_KEY=your-key
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

`ANTHROPIC_API_KEY` is needed for every run. The two Telegram keys are
needed only for `send_briefing.py`. The repository contains no keys, so
anyone running it, including a grader, needs their own API key, and each
run is billed to that key.

### Sample run (no personal data needed)

```
python3 scripts/judgment_core.py --sample
```

This uses the fake data in `sample/parsed/`: one made-up course
("SAMPLE 101") and four made-up personal tasks. Everything in that folder
is fake. Sample mode:

- pins "today" to Wednesday, March 10, 2027, and prints that date at the
  top, so the fake dates always make sense;
- leaves out the tasks typed in code for the real data, and the fixed
  weekly class;
- prints to the terminal only. `send_briefing.py --sample` exits without
  sending anything;
- uses the same prompt, rules and checks as a normal run.

The sample shows the format of the briefing and the kinds of items in
it, not an exact result. The model's choices and wording vary between
runs. The parts computed in code are the same every time: the overdue
lecture, the day counts, and "5 later items not shown".

One run printed this (shortened):

```
SAMPLE MODE — fake data from sample/. Today is pinned to Wednesday, March 10, 2027.

=== OVERDUE — CONFIRM STATUS ===
- Sample 101 — Lecture 7: Multiple Regression. ...

=== MUST-DO ===
- HW 3 problem set (~2h)
    It's a 2h problem set that hasn't been started yet. The required
    buffer puts the real finish-by target at tomorrow. ...
  Total: 5h of 6h budget

=== CAN-DO (reward tier — after Must-Do is done) ===
- Watch/study Lecture 8 (~3h)
    ...

=== UPCOMING (no action needed today) ===
- Attend friend's birthday dinner — Sat Mar 20, 7pm
    ...

5 later items not shown (next: Lecture 10: Clustering, in 19 days).
```

Every run prints its token counts and an estimated cost. These are the
script's own estimates, worked out from the token counts at prices set
in the code. On the current version a run is about $0.09 to $0.18,
higher when the retry fires.

### With your own data

The real-data path is set up for three specific courses. Their syllabus
PDFs and the personal task file are in `data/`, which is not in the
repository, so a fresh clone cannot run this path as is. To use other
courses, the syllabus entries in `scripts/extract_syllabus.py` and the
overlays and task list in `scripts/judgment_core.py` have to be edited.

```
# once per syllabus (PDF in data/); calls the model, then saves JSON
python3 scripts/extract_syllabus.py <predictive|prescriptive|agentic>

# after each edit of data/personal_tasks.txt; review, then save
python3 scripts/import_personal_tasks.py

# briefing in the terminal only
python3 scripts/judgment_core.py

# briefing in the terminal, short version sent to Telegram
python3 scripts/send_briefing.py
```

`scripts/send_test_message.py` sends one test message to confirm the
Telegram keys work.

## Checks and known limits

Checks run in code on every model answer:

- Every task appears in exactly one list. (This one prints a warning
  only; it does not trigger the retry.)
- Must-Do hours do not exceed the daily budget.
- A major coursework task due within 10 days is not left in Can-Do while
  at least half an hour of budget remains.
- A lecture with two or fewer days left in its week is in Must-Do.
- A lecture is scheduled as its full 3-hour block, never a part of it.
- An optional task is not in Must-Do.
- Every item's id is a task that was sent, and its title matches that
  task.
- Every fact claim has a quote, and the quote appears in the text of the
  task it cites.

The syllabus extractor has its own checks: each item's source text must
be a verbatim quote from the PDF text, and its date must appear in that
quote. Items that fail are kept and marked for review.

Known limits:

- **The quote check confirms that a quote exists, not that it is used
  honestly.** Code checks that the quoted words appear in the task's
  text. It does not check that the quote supports the claim, and the
  model could label a fact as its own judgment to avoid the check.
- **Tap-to-complete is not built.** To mark a personal task done,
  delete its line from `data/personal_tasks.txt` and re-run the import.
  For any other task, set `completed` in its overlay or task entry in
  `judgment_core.py`.
  The recurring content series and the recurring personal
  task track progress in their own fields (`episodes_done`,
  `episodes_posted`, `last_done`) rather than `completed`.
- **Some tasks are typed in code**, not read from a file: a recurring
  content series, one recurring personal task, and one presentation
  whose date is not assigned yet.
- **One budget for every day.** The 6 hours is a single constant;
  weekdays and weekends are not treated differently.
- **One retry only.** A check that still fails after the retry is shown
  as a warning in the terminal, and the briefing is still printed.
- **Runs are started by hand.** There is no scheduler, so the morning
  ping is not automatic.
- **A skipped post in the recurring series is not tracked.** Once its
  date passes, the next post takes its place.
- **The 14-day horizon and the 10-day window are guardrails** chosen
  from a few observed cases, and may need adjusting.
- **Telegram gets no warnings.** Check failures and the over-budget
  warning are shown in the terminal only.

## What I'd add next

- Tap-to-complete: mark a task done from the Telegram message.
- Separate weekday and weekend budgets.
- A scheduled morning run.
