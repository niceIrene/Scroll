#!/usr/bin/env bash
# Untrained-baseline eval for the post-training plan: Qwen3.8-27B (via Tinker)
# in the unchanged scroll harness, on BEAM + LongMemEval. Fills the missing
# "Table 2 row" for the trainee; every later SFT/RL claim is measured against
# these numbers.
#
# Usage:
#   scripts/run_tinker_baseline.sh [beam|longmemeval|longmemeval-m|all] [extra flags...]
#
# Extra flags are forwarded verbatim to the underlying `scroll-eval <benchmark>`
# command (single-benchmark targets only — `all` accepts none, since flags like
# --scale are benchmark-specific). Common ones: --task/--scale (subset),
# --concurrency, --judge-workers, --judge-model, --index/--no-index,
# --var-context/--no-var-context, --grade-only (beam), -v.
#
# Required env (put them in .env.local — the runners load it):
#   TINKER_API_KEY          agent-side sampling (tinker SDK)
#   SCROLL_JUDGE_BASE_URL   chat endpoint for the LLM judge (e.g. DashScope)
#   SCROLL_JUDGE_MODEL      judge model id (frozen judge; keep it fixed across runs)
#   DASHSCOPE_API_KEY       judge credential (judges fall back OPENAI_API_KEY ->
#                           DASHSCOPE_API_KEY; under the tinker backend
#                           OPENAI_API_KEY is deliberately NOT set from TINKER_API_KEY)
#
# Evaluate a trained checkpoint later by swapping model.name in the configs to
# its tinker://.../sampler_weights/... path — nothing else changes.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-all}"
shift $(( $# > 0 ? 1 : 0 ))
EXTRA=("$@")
if [[ "$TARGET" == "all" && ${#EXTRA[@]} -gt 0 ]]; then
  echo "extra flags need a single benchmark target (flags like --scale are" \
       "benchmark-specific): got '$TARGET ${EXTRA[*]}'" >&2
  exit 1
fi

# .env.local is loaded by the runners themselves, but validate up front so a
# missing key fails here, not 20 minutes into ingestion.
if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

# Judge endpoint: default from OPENAI_BASE_URL when it is a real chat endpoint
# (the usual DashScope dotenv). The runner overwrites OPENAI_BASE_URL with
# "tinker" in-process, so the judge needs its own var pinned before that.
if [[ -z "${SCROLL_JUDGE_BASE_URL:-}" && "${OPENAI_BASE_URL:-}" == http* ]]; then
  export SCROLL_JUDGE_BASE_URL="$OPENAI_BASE_URL"
  echo "note: SCROLL_JUDGE_BASE_URL defaulted from OPENAI_BASE_URL ($OPENAI_BASE_URL)"
fi

# Judge model: a --judge-model flag (the CLI exports it as SCROLL_JUDGE_MODEL)
# satisfies the requirement just as well as the env var.
# ${EXTRA[@]+...} guards the empty-array case: macOS bash 3.2 treats expanding
# an empty array under `set -u` as an unbound-variable error.
judge_model_flagged=false
for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do
  [[ "$arg" == --judge-model || "$arg" == --judge-model=* ]] && judge_model_flagged=true
done

missing=()
[[ -n "${TINKER_API_KEY:-}" ]] || missing+=("TINKER_API_KEY")
[[ -n "${SCROLL_JUDGE_BASE_URL:-}" ]] || missing+=("SCROLL_JUDGE_BASE_URL")
if [[ -z "${SCROLL_JUDGE_MODEL:-}" ]] && ! $judge_model_flagged; then
  missing+=("SCROLL_JUDGE_MODEL (or pass --judge-model)")
fi
[[ -n "${DASHSCOPE_API_KEY:-}${OPENAI_API_KEY:-}" ]] || missing+=("DASHSCOPE_API_KEY (judge key)")
if (( ${#missing[@]} )); then
  echo "missing required env: ${missing[*]} — set them in .env.local (KEY=value, no 'export')" >&2
  exit 1
fi

run_beam() {
  echo "=== BEAM baseline (Qwen3.8-27B via tinker) ==="
  uv run scroll-eval beam configs/baseline/tinker-qwen3.8-27b-beam.yaml ${EXTRA[@]+"${EXTRA[@]}"}
}

run_lme() {
  echo "=== LongMemEval(S) baseline (Qwen3.8-27B via tinker) ==="
  uv run scroll-eval longmemeval configs/baseline/tinker-qwen3.8-27b-longmemeval.yaml ${EXTRA[@]+"${EXTRA[@]}"}
}

run_lme_m() {
  if [[ ! -d local-tasks/longmemeval-m ]]; then
    echo "skipping longmemeval-m: local-tasks/longmemeval-m not materialized" \
         "(see configs/longmemeval-m.yaml header)" >&2
    return 0
  fi
  echo "=== LongMemEval(M) baseline (Qwen3.8-27B via tinker) ==="
  uv run scroll-eval longmemeval configs/baseline/tinker-qwen3.8-27b-longmemeval-m.yaml ${EXTRA[@]+"${EXTRA[@]}"}
}

case "$TARGET" in
  beam)          run_beam ;;
  longmemeval)   run_lme ;;
  longmemeval-m) run_lme_m ;;
  all)           run_beam; run_lme; run_lme_m ;;
  *) echo "unknown target: $TARGET (want beam|longmemeval|longmemeval-m|all)" >&2; exit 1 ;;
esac

echo "done — summaries under runs/<timestamp>__*/summary.json"
