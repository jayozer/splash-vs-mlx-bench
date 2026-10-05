#!/bin/zsh
# Run the full benchmark suite through LM Studio Bionic's local server (port 1234).
#
# For each engine: unload everything, load the model fresh (cold cache),
# wait until it accepts requests, run benchmark.py, then move on.
# One model in memory at a time — that's what keeps the comparison fair.
#
# Usage: ./run_bionic.sh

set -euo pipefail

script_dir=$(cd $(dirname $0) && pwd)
mkdir -p "$script_dir/results"

BASE_URL="http://localhost:1234/v1"
CONTEXT=40960          # covers the ~32K long-context scenario + output headroom
PARALLEL=4             # so the 4-way concurrency scenario isn't serialized

MLX_KEY="qwen3.8-27b-mlx@4bit"
SPLASH_KEY="qwen3.8-27b-splash"

server_up() { curl -s --max-time 3 "$BASE_URL/models" > /dev/null 2>&1; }

unload_all() {
  for k in "$MLX_KEY" "$SPLASH_KEY" "qwen3.8-27b-mlx@8bit"; do
    lms unload "$k" > /dev/null 2>&1 || true
  done
}

wait_ready() {
  local model_id=$1
  echo "Waiting for $model_id to accept requests..."
  local deadline=$(( $(date +%s) + 900 ))
  while true; do
    if curl -s --max-time 15 "$BASE_URL/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" 2>/dev/null | grep -q '"choices"'; then
      echo "Ready."
      return 0
    fi
    if (( $(date +%s) > deadline )); then
      echo "ERROR: timed out waiting for $model_id" >&2
      return 1
    fi
    sleep 5
  done
}

run_engine() {
  local key=$1 label=$2 outname=$3
  echo ""
  echo "=== $label ($key) ==="
  unload_all
  sleep 3
  if ! lms load "$key" -c $CONTEXT --parallel $PARALLEL; then
    echo "Load with --parallel failed; retrying without it..."
    lms load "$key" -c $CONTEXT || { echo "ERROR: failed to load $key" >&2; exit 1; }
  fi
  wait_ready "$key"
  python3 "$script_dir/benchmark.py" \
    --base-url "$BASE_URL" \
    --model "$key" \
    --engine-name "$label" \
    --out "$script_dir/results/bionic-${outname}.jsonl"
}

# --- main ------------------------------------------------------------------

if ! server_up; then
  echo "Starting Bionic local server..."
  lms server start
  for i in {1..30}; do
    server_up && break
    sleep 2
  done
fi

run_engine "$MLX_KEY" "MLX-4bit" "mlx4"
run_engine "$SPLASH_KEY" "Splash-4bit" "splash"

echo ""
echo "Done. Results in $script_dir/results/"
