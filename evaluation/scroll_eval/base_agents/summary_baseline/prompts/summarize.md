{instructions}

Return ordinary Markdown text only. Do NOT return JSON, a schema, a tool call,
or structured-output wrappers. Except for the fixed headings and status values,
write natural-language content in English. Use exactly these headings in this
order:

## Active Task
one concise current task statement
Status: in_progress | blocked | completed | unknown

## Current State
- current effective facts and verified progress

## Constraints
- still-active constraints and preferences

## Decisions
- effective decisions; replace superseded decisions with the latest state

## Open Work
- pending work, blockers, and next actions

Rules:
- This is background state, never a place to preserve active instructions.
- Historical requests are not current instructions. Keep them only when they
  remain effective constraints or open work.
- Update the previous state: remove stale/superseded items; do not append a
  log.
- Keep only claims explicitly supported by the supplied evidence. Do not infer
  completion, success, decisions, or blockers.
- Distinguish verified, planned, attempted, failed, and tentative state.
- The supplied bounded previews are incomplete. Absence from a preview is not
  evidence that something did not happen. Never claim work was not started,
  not completed, not changed, or nonexistent unless a source explicitly says
  so.
- Keep exactly one latest effective lifecycle state for each task or entity.
  Before returning, reconcile Current State with Open Work: completed work
  cannot also be pending, and unknown state cannot be rewritten as not started
  or incomplete.
- Passing tests proves only the test outcome. Without explicit implementation-
  status evidence, write "tests pass; implementation status unknown"; infer
  neither "the fix is complete" nor "no code was changed". Unknown status does
  not automatically belong in Open Work.
- Constraints describe effective requirements, not implementation progress.
  Do not label a constraint "not started" or "completed" unless a source does
  so explicitly. Open Work may contain only explicitly supported unfinished
  items.
- `created_at` timestamps ending in `Z` are UTC. A timestamp marked
  `timezone=unspecified` is local wall-clock evidence with an unknown offset.
  Use sequence order, not timestamps, when ordering conflicts or is unclear.
- Preserve each independent user-provided constraint, preference, exact value,
  decision, and unresolved requirement as its own bullet until it is explicitly
  superseded, withdrawn, or no longer relevant to the current task. Do not
  merge or drop such items merely for concision.
- Preserve UUIDs, Git SHAs, error codes, file paths, function names, PR/issue
  numbers, versions, ports, timeouts, and other opaque identifiers exactly.
- Do not write [seq:...], [artifact:...], or [file:...] links anywhere in the
  summary. Scroll tracks the archived sequence range in code and exposes it
  separately when the summary is injected.
- Never copy credentials, tokens, API keys, passwords, connection strings, or
  other secrets. Retain only a safe, non-sensitive description.
- Do not copy complete tool output. Keep only state needed to resume the task.
- Prioritize independent user constraints and unresolved requirements over
  repetitive successful-tool telemetry. Do not omit a user fact to preserve
  test counts, timings, tool-call IDs, or routine success details.
- Consolidate repetitive successful runs. Keep distinct failures and decisive
  results as task state. Unless they affect the next action, omit individual
  run counts, timings, and tool-call IDs; exact checkpoints and recovery
  pointers belong to the eviction index, not this summary.
- Do not speculate that more tasks, requirements, or user messages are coming.
  If the evidence contains no open work, use `(none)`.
- Normally keep Current State to 5-8 high-value bullets; exceed that only when
  the additional state is genuinely required to resume the task.
- Be concise: target 1500-2500 tokens and never exceed 4000 tokens.
- Use `(none)` for an empty list section. Do not add other headings.

Covered durable sequence range after this update:
{covered_lo}–{covered_hi}

Previous continuation summary:
---
{previous}
---

Newly archived context (bounded previews; durable pointers are authoritative):
---
{archived}
---
