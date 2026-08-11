"""BEAM judge CLI: grade answers against held-out rubrics, emit a reward.

    python -m scroll_eval.evals.beam.judge \
        --questions <task>/tests/probing_questions.json \
        --answers   <run>/answers.json \
        --out       <run>/scores.json \
        --reward-file <path>

Answers and questions are aligned by type + position. Per-type score = mean of
the headline metric (tau_norm for event_ordering, else llm_judge_score); the
overall reward in [0,1] = mean of per-type means. Honors the Harbor reward
contract (writes the reward to ``--reward-file``, never raises out).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scroll_eval.evals.beam.judge.metrics import evaluate, primary_score
from scroll_eval.evals.beam.judge.model import BeamJudgeModel

# Probes are independent (own answer + rubrics), so they grade concurrently.
# Effective in-flight LLM calls ≈ workers × within-probe parallelism, so keep
# `workers` modest if the judge endpoint rate-limits.
_DEFAULT_WORKERS = 8


def grade(
    questions: dict,
    answers: dict,
    model,
    *,
    workers: int = _DEFAULT_WORKERS,
    align_input: str = "facts",
) -> dict:
    """Grade all answers concurrently. Returns {per_type, overall_reward, n_probes}.

    ``align_input`` ("facts"|"lines") selects event_ordering's alignment input;
    see metrics.evaluate_event_ordering. Other types are unaffected.
    """
    # Flatten to independent probe units, keeping type + position for reassembly.
    units = []  # (qtype, index, ans, rubric)
    for qtype, items in answers.items():
        rubrics = questions.get(qtype) or []
        for index, ans in enumerate(items):
            rubric = rubrics[index].get("rubric", []) if index < len(rubrics) else []
            units.append((qtype, index, ans, rubric))

    def _run(unit):
        qtype, index, ans, rubric = unit
        llm_response = ans.get("llm_response", "") or ""
        result = evaluate(  # within-probe parallelism inside
            qtype,
            rubric,
            llm_response,
            model,
            question=ans.get("question", "") or "",
            align_input=align_input,
        )
        return qtype, index, ans, result

    if workers > 1 and len(units) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(units))) as ex:
            evaluated = list(ex.map(_run, units))
    else:
        evaluated = [_run(u) for u in units]

    # Reassemble per type, preserving the original type order and probe index.
    grouped: dict[str, list] = defaultdict(list)
    for qtype, index, ans, result in evaluated:
        grouped[qtype].append((index, ans, result))

    per_type: dict[str, dict] = {}
    type_means: list[float] = []
    n_probes = 0
    for qtype in answers:
        probes: list[dict] = []
        primaries: list[float] = []
        for index, ans, result in sorted(grouped[qtype], key=lambda r: r[0]):
            p = primary_score(qtype, result)
            primaries.append(p)
            n_probes += 1
            probes.append(
                {
                    "id": ans.get("id", f"{qtype}-{index}"),
                    "question": ans.get("question", ""),
                    "primary": p,
                    "result": result,
                }
            )
        mean = sum(primaries) / len(primaries) if primaries else 0.0
        per_type[qtype] = {"mean": mean, "n": len(probes), "probes": probes}
        if probes:
            type_means.append(mean)

    overall = sum(type_means) / len(type_means) if type_means else 0.0
    return {"per_type": per_type, "overall_reward": overall, "n_probes": n_probes}


def _write_reward(reward_file: str | None, reward: float) -> None:
    if not reward_file:
        return
    try:
        path = Path(reward_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{reward}\n", encoding="utf-8")
    except OSError as e:  # best-effort; never crash grading over the reward sink
        print(f"[beam.judge] could not write reward file {reward_file}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade BEAM answers via LLM-as-judge.")
    ap.add_argument("--questions", required=True, help="Held-out probing_questions.json (rubrics).")
    ap.add_argument("--answers", required=True, help="answers.json: {type: [{question, llm_response}]}.")
    ap.add_argument("--out", required=True, help="Where to write scores.json.")
    ap.add_argument("--reward-file", default=None, help="Where to write the [0,1] reward.")
    ap.add_argument("--model", default=None,
                    help="Judge model (default: env SCROLL_JUDGE_MODEL, else SCROLL_MODEL).")
    ap.add_argument("--base-url", default=None, help="Judge endpoint (default: env OPENAI_BASE_URL).")
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                    help="Concurrent probes to grade (lower if rate-limited).")
    ap.add_argument("--align-input", choices=("facts", "lines"),
                    default=os.environ.get("SCROLL_JUDGE_ALIGN_INPUT", "facts"),
                    help="event_ordering alignment input: 'facts' (default) = BEAM's "
                         "intended atomic-fact extraction; 'lines' = upstream's executed "
                         "raw line-split, for comparability with published BEAM scores. "
                         "Env default: SCROLL_JUDGE_ALIGN_INPUT.")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    # Judge model resolves: --model flag > SCROLL_JUDGE_MODEL env > the agent
    # model (SCROLL_MODEL, BeamJudgeModel's own fallback). This lets a run be
    # answered by a large model but graded by a cheaper/faster one. base_url
    # defaults to the agent endpoint (OPENAI_BASE_URL) unless overridden.
    model = BeamJudgeModel(
        model=args.model or os.environ.get("SCROLL_JUDGE_MODEL"),
        base_url=args.base_url or os.environ.get("SCROLL_JUDGE_BASE_URL"),
    )

    scores = grade(questions, answers, model, workers=args.workers, align_input=args.align_input)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_reward(args.reward_file, scores["overall_reward"])
    print(
        f"[beam.judge] {scores['n_probes']} probes graded; "
        f"overall_reward={scores['overall_reward']:.4f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
