"""Faithful port of BEAM's LLM-as-judge grading.

Vendored from BEAM (MIT-licensed; https://github.com/mohammadtavakoli78/... ),
trimmed to the code path actually exercised by ``run_evaluation``:

- 9 of 10 question types are graded identically: for each rubric item, ask the
  judge model (``unified_llm_judge_base_prompt``) for a score in {0.0, 0.5, 1.0},
  then average → ``llm_judge_score`` (``metrics.judge_rubric``).
- ``event_ordering`` additionally computes ``event_ordering_score`` (Kendall-tau
  + F1) over the response, using the ``align_type="llm"`` path (LLM equivalence).
  One deliberate divergence: upstream feeds the aligner ``extract_facts`` output
  and then dead-assigns ``llm_response.split("\\n")`` over it, so its *executed*
  input is raw lines (blanks/preambles included) and formatting dominates the
  score. ``align_input="facts"`` (our default) restores the intended
  fact-extraction input; ``align_input="lines"`` (CLI ``--align-input lines`` or
  ``SCROLL_JUDGE_ALIGN_INPUT=lines``) reproduces upstream's executed behavior
  for score comparability with published BEAM numbers.

Dependencies are intentionally minimal: ``scipy`` (kendalltau) + ``json_repair``.
BEAM's BLEU/ROUGE/sentence-transformer helpers and the ``semantic`` align path
are NOT on the evaluation dispatch and are omitted.
"""
