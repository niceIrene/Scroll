"""LongMemEval judge CLI: grade answers against held-out gold, emit a reward.

    python -m scroll_eval.evals.longmemeval.judge \
        --questions <task>/tests/probing_questions.json \
        --answers   <run>/answers.json \
        --out       <run>/scores.json \
        --reward-file <path>

Answers and gold are aligned by question type + position. Per-type score = mean
of the per-probe 1/0 verdict; the overall reward in [0,1] = mean of per-type
means. Mirrors the BEAM judge's output schema + Harbor reward contract (writes
the reward to ``--reward-file``, never raises out) so the shared runner /
``scroll-eval summary`` consume it unchanged. A LongMemEval task has exactly one
probe, so per_type usually holds a single entry.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scroll_eval.evals.longmemeval.judge.metrics import LmeJudgeModel, score_one

_DEFAULT_WORKERS = 8


def grade(questions: dict, answers: dict, model, *, workers: int = _DEFAULT_WORKERS) -> dict:
    """Grade all answers concurrently. Returns {per_type, overall_reward, n_probes}."""
    units = []  # (qtype, index, ans, gold)
    for qtype, items in answers.items():
        golds = questions.get(qtype) or []
        for index, ans in enumerate(items):
            gold = golds[index] if index < len(golds) else {}
            units.append((qtype, index, ans, gold))

    def _run(unit):
        qtype, index, ans, gold = unit
        score = score_one(model, gold, ans.get("llm_response", "") or "")
        return qtype, index, ans, score

    if workers > 1 and len(units) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(units))) as ex:
            evaluated = list(ex.map(_run, units))
    else:
        evaluated = [_run(u) for u in units]

    grouped: dict[str, list] = defaultdict(list)
    for qtype, index, ans, score in evaluated:
        grouped[qtype].append((index, ans, score))

    per_type: dict[str, dict] = {}
    type_means: list[float] = []
    n_probes = 0
    for qtype in answers:
        probes: list[dict] = []
        scores: list[float] = []
        for index, ans, score in sorted(grouped[qtype], key=lambda r: r[0]):
            scores.append(score)
            n_probes += 1
            probes.append(
                {
                    "id": ans.get("id", f"{qtype}-{index}"),
                    "question": ans.get("question", ""),
                    "primary": score,
                }
            )
        mean = sum(scores) / len(scores) if scores else 0.0
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
        print(f"[longmemeval.judge] could not write reward file {reward_file}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade LongMemEval answers via LLM-as-judge.")
    ap.add_argument("--questions", required=True, help="Held-out gold probing_questions.json.")
    ap.add_argument("--answers", required=True, help="answers.json: {type: [{id, question, llm_response}]}.")
    ap.add_argument("--out", required=True, help="Where to write scores.json.")
    ap.add_argument("--reward-file", default=None, help="Where to write the [0,1] reward.")
    ap.add_argument("--model", default=None,
                    help="Judge model (default: env SCROLL_JUDGE_MODEL, else SCROLL_MODEL).")
    ap.add_argument("--base-url", default=None, help="Judge endpoint (default: env OPENAI_BASE_URL).")
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                    help="Concurrent probes to grade (lower if rate-limited).")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    model = LmeJudgeModel(
        model=args.model or os.environ.get("SCROLL_JUDGE_MODEL"),
        base_url=args.base_url or os.environ.get("SCROLL_JUDGE_BASE_URL"),
    )

    scores = grade(questions, answers, model, workers=args.workers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_reward(args.reward_file, scores["overall_reward"])
    print(
        f"[longmemeval.judge] {scores['n_probes']} probe(s) graded; "
        f"overall_reward={scores['overall_reward']:.4f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
