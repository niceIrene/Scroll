#!/bin/bash
# Reproduce the LongMemEval SCROLL numbers.
#
# Runs the canonical config (qwen3.6-plus on the 500-QA _s split) and
# writes a per-question-type summary to output/longmemeval/_reproduction.json.
#
# Requires: US_DASHSCOPE_API_KEY exported (or set in .env).
# Wall-clock: ~3 hours on a single workstation with --max-parallel 4.
set -e

CONFIG="${CONFIG:-configs/longmemeval/scroll_s.json}"
SEED="${SEED:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="output/longmemeval"

# Source .env if present so the API key is available without manual export.
[ -f .env ] && set -a && source .env && set +a

if [ -z "${US_DASHSCOPE_API_KEY:-}" ]; then
    echo "ERROR: US_DASHSCOPE_API_KEY is not set." >&2
    echo "Set it in .env or export it before running." >&2
    exit 1
fi

echo "=== SCROLL LongMemEval reproduction ==="
echo "Config:       $CONFIG"
echo "Seed:         $SEED"
echo "Max parallel: $MAX_PARALLEL"
echo

python scripts/run_longmemeval.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --max-parallel "$MAX_PARALLEL" \
    --include-abstention

# Locate the most recent run dir and emit a summary JSON.
RUN_DIR=$(ls -t -d "$OUTPUT_ROOT"/scroll_* 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/summary.json" ]; then
    cp "$RUN_DIR/summary.json" "$OUTPUT_ROOT/_reproduction.json"
    echo
    echo "Summary written to $OUTPUT_ROOT/_reproduction.json"
    python -c "
import json, sys
with open('$OUTPUT_ROOT/_reproduction.json') as f:
    s = json.load(f)
print()
print('=== Per-question-type accuracy ===')
for k, v in sorted(s.items()):
    if isinstance(v, dict) and 'accuracy' in v:
        print(f'  {k:30s} acc={v[\"accuracy\"]:.3f}  (n={v.get(\"n\", \"?\")})')
"
else
    echo "WARN: could not find summary.json under $OUTPUT_ROOT" >&2
fi
