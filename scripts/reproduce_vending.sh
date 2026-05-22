#!/bin/bash
# Reproduce the Vending Machine SCROLL numbers.
#
# Runs a 30-day single-seed simulation under the canonical config and
# writes the resulting run_result.json (net worth, units sold, active
# days, probe scores) to output/vending/_reproduction.json.
#
# Requires: OPENAI_API_KEY exported (or set in .env).
# Wall-clock: ~30 minutes per seed.
set -e

CONFIG="${CONFIG:-configs/vending/scroll.json}"
SEED="${SEED:-1}"
OUTPUT_ROOT="output/vending"

[ -f .env ] && set -a && source .env && set +a

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    echo "Set it in .env or export it before running." >&2
    exit 1
fi

echo "=== SCROLL Vending reproduction ==="
echo "Config:  $CONFIG"
echo "Seed:    $SEED"
echo

Scroll --config "$CONFIG" --seed "$SEED" --fresh

# Find the latest run dir and emit a summary
RUN_DIR=$(ls -t -d "$OUTPUT_ROOT"/scroll_* 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/run_result.json" ]; then
    cp "$RUN_DIR/run_result.json" "$OUTPUT_ROOT/_reproduction.json"
    echo
    echo "Summary written to $OUTPUT_ROOT/_reproduction.json"
    python -c "
import json
with open('$OUTPUT_ROOT/_reproduction.json') as f:
    r = json.load(f)
print()
print('=== Vending run summary ===')
print(f'  net_worth:    {r.get(\"net_worth\", \"?\"):.2f}')
print(f'  units_sold:   {r.get(\"units_sold\", \"?\")}')
print(f'  active_days:  {r.get(\"active_days\", \"?\")}')
"
else
    echo "WARN: could not find run_result.json under $OUTPUT_ROOT" >&2
fi
