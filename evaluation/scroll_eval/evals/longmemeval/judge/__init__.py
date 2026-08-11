"""LongMemEval LLM-as-judge grading.

A faithful port of the relaxed per-question-type judge templates from the
original Scroll LongMemEval benchmark (which themselves mirror the upstream
``evaluate_qa.py`` templates, loosened to cut false negatives on
format/tense/unit variation and false positives on abstention). One QA per task
→ one 0/1 verdict → the task reward. Unlike BEAM there is no Kendall-tau /
event-ordering path, so this judge needs no ``scipy``/``json_repair``.
"""
