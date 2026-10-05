#!/bin/zsh
# Run the full benchmark suite against one engine, managing its server lifecycle.
#
# Usage:
#   ./run_engine.sh mlx      # Qwen3.8-27B via mlx-lm server (port 8080)
#   ./run_engine.sh splash   # Qwen3.8-27B via Splash engine  (port 8000)
#
# First run of each engine downloads the model (~17 GB), so be patient on
# the readiness wait. The server is killed when the script exits.

set -euo pipefail

usage() { echo "Usage: $0 mlx|splash" >&2; exit 1 }
[[ $# -eq 1 ]] || usage

engine=$1
script_dir=$(cd $(dirname $0) && pwd)
mkdir -p "$script_dir/results"

case $engine in
  mlx)
    model="mlx-community/Qwen3.8-27B-4bit"
    port=8080
    base_url="http://127.0.0.1:8080/v1"
    engine_name="MLX"
    server_cmd=(mlx_lm.server --model "$model")
    ;;
  splash)
    model="incoai/Qwen3.8-27B-Splash"
    port=8000
    base_url="http://127.0.0.1:8000/v1"
    engine_name="Splash"
    server_cmd=(splash serve --model "$model")
    ;;
  *) usage ;;
esac

if curl -s --max-time 2 "$base_url/models" > /dev/null 2>&1; then
  echo "ERROR: something is already serving on port $port." >&2
  echo "Stop the other engine first — run one engine at a time for fair results." >&2
  exit 1
fi

log="$script_dir/results/${engine}-server.log"
echo "Starting $engine server (first run downloads the model)..."
"${server_cmd[@]}" > "$log" 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null || true; wait $server_pid 2>/dev/null || true' EXIT INT TERM

timeout=${READY_TIMEOUT:-1800}
echo "Waiting for readiness (up to ${timeout}s)..."
deadline=$(( $(date +%s) + timeout ))
while true; do
  if curl -s --max-time 3 "$base_url/models" | grep -q .; then
    echo "Server ready."
    break
  fi
  if ! kill -0 $server_pid 2>/dev/null; then
    echo "ERROR: server process died. Last log lines:" >&2
    tail -30 "$log" >&2
    exit 1
  fi
  if (( $(date +%s) > deadline )); then
    echo "ERROR: timed out waiting for the server. Last log lines:" >&2
    tail -30 "$log" >&2
    exit 1
  fi
  sleep 5
done

python3 "$script_dir/benchmark.py" \
  --base-url "$base_url" \
  --model "$model" \
  --engine-name "$engine_name" \
  --out "$script_dir/results/${engine}.jsonl"

echo ""
echo "Done. Results in $script_dir/results/"
