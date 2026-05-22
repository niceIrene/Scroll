#!/bin/bash
# Reproduce the BEAM SCROLL numbers.
#
# BEAM = "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term
# Memory in LLMs" (Tavakoli, ICLR 2026, arXiv:2510.27246).
#
# Runs the canonical config (qwen3.6-plus on the 100K scale, all 20
# chats) and writes a per-category accuracy summary to
# output/beam/_reproduction.json.
#
# Requires:
#   - US_DASHSCOPE_API_KEY  (agent)
#   - OPENAI_API_KEY        (LLM judge — BEAM scores via OpenAI's
#                            evaluation model; override via judge_*
#                            fields in the config if you prefer a
#                            different endpoint)
#
# Wall-clock: ~2 hours for the 20-chat 100K sweep at --max-parallel 4.
set -e

CONFIG="${CONFIG:-configs/beam/scroll.json}"
SEED="${SEED:-1}"
OUTPUT_ROOT="output/beam"

[ -f .env ] && set -a && source .env && set +a

if [ -z "${US_DASHSCOPE_API_KEY:-}" ]; then
    echo "ERROR: US_DASHSCOPE_API_KEY is not set." >&2
    exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARN: OPENAI_API_KEY is not set — judge will score 0.0 for every probe." >&2
fi

# Make sure the BEAM dataset submodule is initialized.
if [ ! -d external/beam/chats ]; then
    echo "Initializing BEAM submodule..."
    git submodule update --init external/beam
fi

# Available chats at the configured scale.
SCALE=$(python -c "import json; print(json.load(open('$CONFIG'))['simulation']['scale'])")
CHAT_IDS=$(python -c "from Scroll.benchmarks.beam.dataset import list_chats; print(' '.join(list_chats('external/beam/chats', '$SCALE')))")

echo "=== SCROLL BEAM reproduction ==="
echo "Config:    $CONFIG"
echo "Scale:     $SCALE"
echo "Chats:     $CHAT_IDS"
echo "Seed:      $SEED"
echo

# Run each chat as a separate Scroll invocation. We use the CLI's
# --output-dir to namespace per-chat. (BEAM doesn't have a built-in
# multi-task orchestrator yet — straightforward sequential loop.)
mkdir -p "$OUTPUT_ROOT"
for CHAT_ID in $CHAT_IDS; do
    CHAT_OUT="$OUTPUT_ROOT/scroll_${SEED}/chat_${CHAT_ID}"
    if [ -d "$CHAT_OUT/probe_results.json" ]; then
        echo "Skipping chat $CHAT_ID (already complete)."
        continue
    fi
    echo "--- chat $CHAT_ID ---"
    # Patch the config inline with the current chat_id.
    TMP_CFG=$(mktemp)
    python -c "
import json
with open('$CONFIG') as f: cfg = json.load(f)
cfg['simulation']['chat_id'] = '$CHAT_ID'
with open('$TMP_CFG', 'w') as f: json.dump(cfg, f, indent=2)
"
    Scroll --config "$TMP_CFG" --seed "$SEED" --output-dir "$CHAT_OUT" --fresh
    rm -f "$TMP_CFG"
done

# Aggregate per-category accuracy across all chats.
python -c "
import json, glob, collections
buckets = collections.defaultdict(list)
for p in glob.glob('$OUTPUT_ROOT/scroll_${SEED}/chat_*/probe_results.json'):
    with open(p) as f:
        for r in json.load(f).get('probes', []):
            qid = r.get('question_id', '')
            # qid format: <chat_id>_<category>_<idx>
            parts = qid.split('_')
            if len(parts) >= 3:
                category = '_'.join(parts[1:-1])
                buckets[category].append(r.get('score', 0.0))
out = {
    cat: {
        'accuracy': round(sum(s)/len(s), 4),
        'n': len(s),
    } for cat, s in sorted(buckets.items())
}
out['_overall'] = {
    'accuracy': round(sum(sum(v) for v in buckets.values()) / sum(len(v) for v in buckets.values()), 4),
    'n': sum(len(v) for v in buckets.values()),
}
with open('$OUTPUT_ROOT/_reproduction.json', 'w') as f:
    json.dump(out, f, indent=2)
print()
print('=== Per-category accuracy ===')
for k, v in out.items():
    print(f'  {k:30s} acc={v[\"accuracy\"]:.3f}  (n={v[\"n\"]})')
print()
print('Summary written to $OUTPUT_ROOT/_reproduction.json')
"
