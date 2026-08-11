"""Migration produces the expected task layout with held-out separation."""
from __future__ import annotations

import json
from pathlib import Path

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "migrate_beam", Path(__file__).parent.parent.parent / "scripts" / "migrate_beam.py"
)
migrate_beam = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate_beam)


def _make_source(root: Path) -> None:
    conv = root / "1"
    (conv / "probing_questions").mkdir(parents=True)
    chat = [
        {
            "batch_number": 1,
            "time_anchor": None,
            "turns": [
                [
                    {"role": "user", "id": 0, "time_anchor": "March-15-2024",
                     "question_type": "main_question", "content": "Help me plan. ->-> 1,1"},
                    {"role": "assistant", "id": 1, "content": "Sure, here is a plan."},
                ]
            ],
        }
    ]
    (conv / "chat.json").write_text(json.dumps(chat), encoding="utf-8")
    probing = {
        "abstention": [
            {"question": "Q-abs?", "ideal_response": "No info.", "rubric": ["state no info"]}
        ],
        "instruction_following": [
            {"question": "Q-if?", "expected_compliance": "use code blocks",
             "rubric": ["uses code blocks"]}
        ],
    }
    (conv / "probing_questions" / "probing_questions.json").write_text(
        json.dumps(probing), encoding="utf-8"
    )


def test_migrate_layout_and_heldout(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_source(src)
    dest = tmp_path / "out"

    written = migrate_beam.migrate(src, dest, "100K")
    assert written == ["100K-1"]

    task = dest / "100K-1"
    for rel in ("task.toml", "chat.json", "questions.json",
                "solution/answers.json", "tests/probing_questions.json", "tests/test.sh"):
        assert (task / rel).exists(), f"missing {rel}"
    # Standing guidance now lives in the package system prompt, not per task.
    assert not (task / "instruction.md").exists()

    # questions.json is agent-visible: only id/type/question, no rubric/gold leak.
    questions = json.loads((task / "questions.json").read_text())
    assert {k for q in questions for k in q} == {"id", "type", "question"}
    assert len(questions) == 2

    # Held-out file keeps the rubrics.
    held = json.loads((task / "tests" / "probing_questions.json").read_text())
    assert held["abstention"][0]["rubric"] == ["state no info"]

    # Oracle gold uses the right per-type field (ideal_response / expected_compliance).
    gold = json.loads((task / "solution" / "answers.json").read_text())
    assert gold["abstention"][0]["llm_response"] == "No info."
    assert gold["instruction_following"][0]["llm_response"] == "use code blocks"

    # task.toml carries the native marker + name.
    toml = (task / "task.toml").read_text()
    assert 'name = "beam/100K-1"' in toml
    assert 'runner = "native"' in toml


def _plan_wrapped_chat() -> list[dict]:
    """10M-tier shape: plan wrappers, batch numbers restarting per plan."""
    def batch(n: int, text: str) -> dict:
        return {
            "batch_number": n,
            "time_anchor": f"April-{n:02d}-2024",
            "turns": [[{"role": "user", "id": 0, "content": text},
                       {"role": "assistant", "id": 1, "content": "ok"}]],
        }
    return [
        {"plan-1": [batch(1, "first plan, first batch"), batch(2, "first plan, second")]},
        {"plan-2": [batch(1, "second plan, first batch")]},
    ]


def test_migrate_flattens_plan_wrapped_10m(tmp_path: Path) -> None:
    src = tmp_path / "src"
    conv = src / "1"
    (conv / "probing_questions").mkdir(parents=True)
    (conv / "chat.json").write_text(json.dumps(_plan_wrapped_chat()), encoding="utf-8")
    probing = {"abstention": [{"question": "Q?", "ideal_response": "A.", "rubric": ["r"]}]}
    (conv / "probing_questions" / "probing_questions.json").write_text(
        json.dumps(probing), encoding="utf-8"
    )

    migrate_beam.migrate(src, tmp_path / "out", "10M")

    chat = json.loads((tmp_path / "out" / "10M-1" / "chat.json").read_text())
    # Flat canonical batch list with GLOBAL session numbering across plans.
    assert [b["batch_number"] for b in chat] == [1, 2, 3]
    # Plan provenance survives on each batch.
    assert [(b["plan"], b["plan_batch_number"]) for b in chat] == [
        ("plan-1", 1), ("plan-1", 2), ("plan-2", 1)]
    # Content and per-batch fields are intact.
    assert chat[2]["turns"][0][0]["content"] == "second plan, first batch"
    assert all("time_anchor" in b for b in chat)
    # n_sessions counts flattened batches, not plan wrappers.
    assert "n_sessions = 3" in (tmp_path / "out" / "10M-1" / "task.toml").read_text()


def test_normalize_chat_passthrough_and_empty_guard(tmp_path: Path) -> None:
    flat = [{"batch_number": 1, "time_anchor": None, "turns": [[{"role": "user", "content": "hi"}]]}]
    assert migrate_beam.normalize_chat(flat) is flat  # flat tiers untouched (verbatim copy path)

    # An unrecognized schema (zero turns) fails migration instead of producing
    # a task that ingests into empty memory.
    src = tmp_path / "src"
    conv = src / "1"
    (conv / "probing_questions").mkdir(parents=True)
    (conv / "chat.json").write_text(json.dumps([{"mystery": []}]), encoding="utf-8")
    (conv / "probing_questions" / "probing_questions.json").write_text("{}", encoding="utf-8")
    try:
        migrate_beam.migrate(src, tmp_path / "out", "10M")
    except ValueError as e:
        assert "no turns" in str(e)
    else:
        raise AssertionError("expected ValueError for zero-turn chat")
