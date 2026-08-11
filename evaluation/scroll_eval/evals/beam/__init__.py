"""BEAM long-term-memory benchmark, run natively against scroll agents.

A BEAM conversation is ingested into the durable ``conversation_history`` store
as prior sessions (see ``ingest``); the agent then answers each probing question
from memory/retrieval (see ``runner``); answers are graded by a faithful port of
BEAM's LLM-as-judge metrics (see ``judge``).
"""
