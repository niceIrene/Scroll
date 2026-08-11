"""Judge grading: rubric scoring, oracle ~1.0, per-type aggregation shape."""
from __future__ import annotations

from scroll_eval.evals.beam.judge.__main__ import grade


class StubModel:
    """Deterministic judge: every rubric item scores 1.0; no network."""

    def __init__(self, content: str = '{"score": 1.0, "reason": "stub"}') -> None:
        self._content = content

    def invoke(self, _x):
        class R:
            content = self._content
        R.content = self._content
        return R()


def test_grade_oracle_full_marks_non_ordering() -> None:
    questions = {
        "abstention": [{"rubric": ["state no info"]}],
        "information_extraction": [{"rubric": ["names the date", "names the city"]}],
    }
    answers = {
        "abstention": [{"id": "abstention-0", "question": "q", "llm_response": "no info"}],
        "information_extraction": [
            {"id": "information_extraction-0", "question": "q", "llm_response": "March 15, Paris"}
        ],
    }
    scores = grade(questions, answers, StubModel())
    assert scores["n_probes"] == 2
    assert scores["per_type"]["abstention"]["mean"] == 1.0
    assert scores["per_type"]["information_extraction"]["mean"] == 1.0
    assert scores["overall_reward"] == 1.0


def test_event_ordering_uses_tau_norm_headline() -> None:
    questions = {"event_ordering": [{"rubric": ["first event", "second event", "third event"]}]}
    answers = {"event_ordering": [
        {"id": "event_ordering-0", "question": "order?",
         "llm_response": "first event\nsecond event\nthird event"}
    ]}
    scores = grade(questions, answers, StubModel())
    result = scores["per_type"]["event_ordering"]["probes"][0]["result"]
    # event_ordering carries the tau metrics, and its headline (primary) is tau_norm.
    assert "tau_norm" in result and "f1" in result
    assert scores["per_type"]["event_ordering"]["mean"] == result["tau_norm"]


class OrderingStubModel:
    """Dispatching stub for the three LLM roles in event_ordering grading.

    - fact extraction (prompt mentions "semantic fact units"): returns a fixed
      numbered fact list, regardless of the answer's formatting;
    - equivalence (messages list): YES iff both snippets share an "eventN" token;
    - rubric judge (string prompt): always full marks.
    """

    FACTS = '1. "event1 happened"\n2. "event2 happened"\n3. "event3 happened"'

    def invoke(self, x):
        class R:
            content = ""
        if isinstance(x, list):  # equivalence: [system, user] messages
            user = x[1]["content"]
            import re
            tokens = re.findall(r"event\d", user)
            R.content = "YES" if len(set(tokens)) == 1 and len(tokens) >= 2 else "NO"
        elif "semantic fact units" in x:  # fact extraction
            R.content = self.FACTS
        else:  # rubric judge
            R.content = '{"score": 1.0, "reason": "stub"}'
        return R()


def test_event_ordering_facts_mode_aligns_paragraph_answers() -> None:
    """The default facts path scores a paragraph answer by content, not layout.

    Under the upstream-executed "lines" behavior this same answer is ONE
    unmatchable line (plus nothing else), so alignment fails; extract_facts
    restores the intended input and all three events align in order.
    """
    questions = {"event_ordering": [{"rubric": ["event1 first", "event2 second", "event3 third"]}]}
    answers = {"event_ordering": [
        {"id": "event_ordering-0", "question": "order?",
         "llm_response": "It went: (1) event1, then (2) event2, then (3) event3."}
    ]}
    scores = grade(questions, answers, OrderingStubModel())
    result = scores["per_type"]["event_ordering"]["probes"][0]["result"]
    assert result["align_input"] == "facts"
    assert result["precision"] == 1.0 and result["recall"] == 1.0
    assert result["tau_norm"] == 1.0

    # The same probe under align_input="lines" (upstream-executed behavior):
    # the single paragraph line has multiple distinct eventN tokens, so the
    # equivalence stub rejects it and nothing aligns.
    scores_lines = grade(questions, answers, OrderingStubModel(), align_input="lines")
    result_lines = scores_lines["per_type"]["event_ordering"]["probes"][0]["result"]
    assert result_lines["align_input"] == "lines"
    assert result_lines["recall"] == 0.0
    assert result_lines["tau_norm"] < result["tau_norm"]


def test_event_ordering_facts_mode_strips_rubric_instruction_prefix() -> None:
    """Facts mode aligns against rubric content, not the embedded judge phrasing."""
    questions = {"event_ordering": [{"rubric": [
        "LLM response should mention: event1 first",
        "LLM response should mention: event2 second",
    ]}]}
    answers = {"event_ordering": [
        {"id": "event_ordering-0", "question": "order?",
         "llm_response": "(1) event1, then (2) event2."}
    ]}
    scores = grade(questions, answers, OrderingStubModel())
    result = scores["per_type"]["event_ordering"]["probes"][0]["result"]
    # extraction yields 3 stub facts; the first two align to the stripped refs
    assert result["recall"] == 1.0
    assert result["tau_norm"] == 1.0


def test_event_ordering_facts_mode_falls_back_on_empty_extraction() -> None:
    class EmptyFactsModel(OrderingStubModel):
        FACTS = "\n\n"  # extraction returns nothing usable

    questions = {"event_ordering": [{"rubric": ["event1 first"]}]}
    answers = {"event_ordering": [
        {"id": "event_ordering-0", "question": "q", "llm_response": "event1 event1"}
    ]}
    # Falls back to line-split: the single line has one distinct token repeated,
    # so the equivalence stub matches it and the probe still scores.
    scores = grade(questions, answers, EmptyFactsModel())
    result = scores["per_type"]["event_ordering"]["probes"][0]["result"]
    assert result["recall"] == 1.0


def test_empty_rubric_does_not_crash() -> None:
    scores = grade(
        {"abstention": [{"rubric": []}]},
        {"abstention": [{"id": "abstention-0", "question": "q", "llm_response": "x"}]},
        StubModel(),
    )
    assert scores["per_type"]["abstention"]["mean"] == 0.0
