"""LongMemEval judge: verdict parsing, template selection, and grade() aggregation."""
from __future__ import annotations

from scroll_eval.evals.longmemeval.judge import metrics
from scroll_eval.evals.longmemeval.judge.__main__ import grade


def test_parse_verdict_prefers_region_after_thinking() -> None:
    # The thinking block may itself contain 'no'; the verdict is what follows.
    text = "<judge_thinking>the answer says no date but...</judge_thinking>\nyes"
    assert metrics.parse_judge_verdict(text) == 1.0
    assert metrics.parse_judge_verdict("<judge_thinking>looks right</judge_thinking>\nno") == 0.0


def test_parse_verdict_falls_back_to_last_token() -> None:
    assert metrics.parse_judge_verdict("reasoning... final: yes") == 1.0
    assert metrics.parse_judge_verdict("") == 0.0          # empty -> failure
    assert metrics.parse_judge_verdict("mid-thought, no tag") == 0.0  # no verdict -> failure


def test_template_selection_by_qtype_and_abstention() -> None:
    # Abstention overrides qtype; preference uses the rubric-framed template;
    # unknown qtype falls back to default.
    abs_p = metrics.judge_prompt("multi-session", "q", "a", "r", abstention=True)
    assert "UNANSWERABLE" in abs_p
    pref_p = metrics.judge_prompt("single-session-preference", "q", "a", "r", abstention=False)
    assert "RECOMMENDATION" in pref_p and "Rubric:" in pref_p
    ku_p = metrics.judge_prompt("knowledge-update", "q", "a", "r", abstention=False)
    assert "REVISED" in ku_p
    default_p = metrics.judge_prompt("single-session-user", "q", "a", "r", abstention=False)
    assert "semantic equivalence" in default_p


class _FakeModel:
    """Returns a verdict based on whether the response contains the gold answer."""

    def invoke(self, prompt: str):
        # The prompt embeds "Model Response: <resp>" and "Correct Answer: <ans>".
        import re

        ans = re.search(r"(?:Correct Answer|Rubric|Explanation[^:]*): (.+?)\n\nModel Response:", prompt, re.S)
        resp = prompt.split("Model Response:", 1)[1]
        verdict = "yes" if ans and ans.group(1).strip() and ans.group(1).strip() in resp else "no"
        return type("R", (), {"content": f"<judge_thinking>...</judge_thinking>\n{verdict}"})()


def test_grade_aligns_by_type_and_computes_reward() -> None:
    questions = {
        "multi-session": [
            {"question": "q1", "answer": "PARIS", "question_type": "multi-session", "is_abstention": False},
            {"question": "q2", "answer": "TOKYO", "question_type": "multi-session", "is_abstention": False},
        ]
    }
    answers = {
        "multi-session": [
            {"id": "a1", "question": "q1", "llm_response": "It is PARIS for sure."},   # correct
            {"id": "a2", "question": "q2", "llm_response": "I think it is Berlin."},    # wrong
        ]
    }
    out = grade(questions, answers, _FakeModel(), workers=1)
    assert out["n_probes"] == 2
    assert out["per_type"]["multi-session"]["mean"] == 0.5
    assert out["overall_reward"] == 0.5
    assert out["per_type"]["multi-session"]["probes"][0]["primary"] == 1.0
