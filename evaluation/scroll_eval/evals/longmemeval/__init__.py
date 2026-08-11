"""LongMemEval memory-QA benchmark, run natively against scroll agents.

A LongMemEval instance is a long multi-session chat history (the "haystack")
plus one question with a gold answer. Each instance becomes one native task:
the haystack is ingested into the durable ``conversation_history`` store as prior
sessions (see ``ingest``); the agent then answers the single question from
memory/retrieval (see ``runner``); the answer is graded by an LLM judge that
ports LongMemEval's per-question-type templates + abstention handling (see
``judge``).

This mirrors the ``beam`` native eval — same seed-DB + ``scope="task"`` retrieval
contract, same run-dir shape — but with exactly one probe per task.
"""
