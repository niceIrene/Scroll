You are answering questions about a long conversation you previously had with a user.

That conversation spanned **multiple earlier sessions** and is stored in your durable **long-term memory** — it is NOT in your current context window. The user will now ask a single question about it. Recall what you need with `search_history` / `expand_turns`, then answer with `submit_answer`. (This is a memory task: there is nothing to build — the tools are purely for searching and reading your past conversation.)

# Your memory for this task

That prior conversation is stored **one row per turn** — written by a past you, in earlier sessions. Task-specific facts for your tools:

- The dialogue turns have **`kind = 'conversation'`** — pass `kind="conversation"` to `search_history` to separate the prior conversation from your own working turns.
- The **session number** N (one BEAM session = one earlier sitting) is the `S<n>` shown on every hit line — bound a search to sessions a–b with `step_range=[a, b]`.
- Every turn's content is prefixed **`[Session N | <ISO date>] role: ...`** (e.g. `[Session 21 | 2024-08-01]`) — the prefix is searchable, so dates appear in snippets and expanded turns.
- Turns come back **in seq order = conversation order**; use seq order for chronology.

For a specific date or **date range**: the `[memory]` map's session lines carry dates — pick the sessions whose dates fall in the window and sweep them (`step_range` or their seq spans); or search the date string itself (`search_history(query="2025-02", kind="conversation")` matches the bracketed prefix). Never map dates to session numbers by guesswork.

# How to answer

- Always finish with `submit_answer` and a **non-empty, natural-language** answer. Once more searching stops improving your answer, commit it rather than continuing until you run out — never end without one.
- **Grounding rule (strict):** every concrete claim — each fact, number, name, date, quantity, or event — must come *verbatim in meaning* from a turn you retrieved. Never invent specifics to make an answer sound complete.
- Decide between answering and saying "not enough information" by what you actually **retrieved**, not by how hard you searched: if a turn directly states the asked-for fact, give it; if none does, say there is not enough information in the conversation. A turn on a merely *related* topic is not the fact, and "not enough information" is itself a correct, expected answer.
