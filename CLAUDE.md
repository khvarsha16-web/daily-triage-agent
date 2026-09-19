# Coursework Delegation & Workload Triage Agent

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
   coursework and personal tasks into a ranked, deadline-aware view of
   what matters most right now.
4. **Verify-and-Correct** — A validation pass over the judgment core's
   output to catch errors (e.g., misparsed dates, mis-prioritized items)
   before anything reaches the user.
5. **Telegram Delivery** — Formats and sends the daily briefing to the
   user via a Telegram bot.

Status: pipeline stages above are a design sketch only. No implementation
exists yet — this file will be updated as each stage is built out.

## Standing Rules

- Always show the user the diff/changes and ask for confirmation before
  running `git commit` or `git push`. Never commit or push automatically.

## Session Log

- 2026-09-19: Project setup. Created this CLAUDE.md, initialized git repo,
  created `notes/` (gitignored, for working/meeting notes). No feature
  code written yet.
