# Qwen3.8-27B: MLX vs Splash — benchmark kit

Reproducible speed comparison of the same model (Qwen3.8-27B, 4-bit) on two
local inference engines for Apple silicon:

| Engine | What it is | Model package | Server |
|---|---|---|---|
| **MLX** | Apple's general-purpose ML framework (`mlx-lm` server) | [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | `mlx_lm.server` → `127.0.0.1:8080` |
| **Splash** | Inco AI's engine, specialized per-model (kernels + DFlash 2 draft for speculative decoding) | [`incoai/Qwen3.8-27B-Splash`](https://huggingface.co/incoai/Qwen3.8-27B-Splash) (4-bit + draft, 17.4 GB) | `splash serve` → `127.0.0.1:8000` |

Both expose the OpenAI-compatible `/v1/chat/completions` endpoint with
streaming, which is what makes a like-for-like benchmark possible.

## Requirements

- Apple M3 or newer, macOS 26.4+, ≥ 36 GB unified memory (Splash minimums).
- Homebrew, Python 3.9+ (system Python is fine — the benchmark is stdlib-only).

## Setup

```bash
# MLX engine (uv keeps it in its own Python env; pipx works too)
brew install uv
uv tool install mlx-lm

# Splash engine
brew install incoai/tap/splash
```

## Run the benchmark

Run **one engine at a time** (both want the GPU and unified memory; running
them concurrently would skew results):

```bash
cd splash-vs-mlx-bench
chmod +x run_engine.sh benchmark.py   # once

./run_engine.sh mlx      # downloads ~17 GB model on first run
./run_engine.sh splash   # downloads 17.4 GB package on first run
```

Each invocation starts the engine, waits for readiness, runs the full suite,
and kills the server. Output lands in `results/`:

- `mlx.jsonl` / `splash.jsonl` — raw per-run metrics (for charts)
- `mlx.md` / `splash.md` — Markdown summary tables (paste into the blog)
- `mlx-server.log` / `splash-server.log` — engine logs

### If you already have a server running

Skip the wrapper and point the benchmark at it directly:

```bash
python3 benchmark.py --base-url http://127.0.0.1:8080/v1 \
  --model mlx-community/Qwen3.8-27B-4bit --engine-name MLX \
  --out results/mlx.jsonl
```

This also works against **LM Studio Bionic's built-in server** (default
`http://localhost:1234/v1`) if you want to measure the in-app experience with
the Splash backend selected — but for the blog, standalone servers are the
cleaner, more reproducible setup.

## What gets measured

| Metric | Definition |
|---|---|
| **TTFT** (time to first token) | Request sent → first streamed token (content or reasoning). Reported cold and cached. |
| **Decode tok/s** | Output tokens ÷ span between first and last token arrival (streaming). |
| **Prefill tok/s** | `prompt_tokens ÷ TTFT` on the cold long-context run (estimate; includes scheduling). |
| **ITL p50/p95** | Median / 95th-percentile inter-token latency — smoothness, not just average speed. |
| **Total time** | End-to-end request duration. |
| **Peak RSS** | Server process memory sampled every 0.5 s during the suite (via `lsof`/`ps`). |

### Test matrix

| Scenario | Prompt size | Max output | Runs |
|---|---|---|---|
| `short` | ~100 tokens | 300 | 3 (median reported) |
| `medium` | ~4K tokens | 300 | 3 |
| `long` | ~32K tokens | 256 | 3 — run 1 is **cold**, runs 2–3 replay the same prompt (**cached TTFT**) |
| `concurrent` | 4 × short in parallel | 256 each | aggregate tok/s + per-request TTFT |

Every scenario is preceded by a warmup run using *different* filler text, so
the first measured long-context prompt is genuinely unseen by the KV cache.
All runs use `temperature: 0` and identical prompts across engines. Actual
prompt token counts come from each server's `usage` field and are recorded in
the JSONL — use those real numbers in the post.

## Fairness notes (put these in the blog)

1. **Same model, different quantization recipes.** Both are 4-bit, but
   `mlx-community`'s 4-bit and Inco's Splash package are not byte-identical
   conversions. Speed comparison is valid; a quality delta, if any, is not an
   engine effect.
2. **Speculative decoding is part of Splash.** Each Splash package ships a
   trained DFlash 2 draft model; the decode path is draft → verify → accept as
   one fused unit. MLX runs plain autoregressive decoding here. So this is an
   *engine-vs-engine* comparison — exactly the question "is Splash worth it" —
   but not a pure kernel shootout.
3. **Sequential runs, same machine.** One engine at a time; note which ran
   first (thermal state can drift over a long session). If the gap is large,
   order effects won't matter; if it's small, repeat the suite in reverse
   order.
4. **Cache behavior is a feature being tested, not controlled away.** The
   cached-TTFT scenario measures KV-cache reuse across requests — the thing
   agentic workloads actually do. Don't restart servers between runs.
5. **Memory:** on Apple silicon, GPU allocations live in unified memory and
   show up in process RSS, so peak RSS is a fair footprint proxy.

## Blog post outline

1. **TL;DR** — verdict table: Splash vs MLX on your machine, one sentence per
   metric, and the "worth it in Bionic?" answer.
2. **The setup** — your Mac (chip/RAM/macOS), the model, what each engine is
   in one paragraph. Splash's pitch: "the engine is built around the model."
3. **Methodology** — metrics table above, test matrix, warmup/cold-cache
   design. This is what makes the post credible; link to this repo/folder.
4. **Results** — three charts from the JSONL:
   - decode tok/s by prompt length (short/medium/long)
   - TTFT: cold vs cached at 32K (the headline number for agents)
   - concurrency: aggregate tok/s at 4 parallel requests
5. **Why Splash is fast** — per-model fused kernels, DFlash 2 draft (sliding
   window keeps it cheap at long context), memory plan that budgets nearly all
   of unified memory. Cite Inco's launch post for the design details.
6. **Caveats** — only two supported models today, no generic fallback,
   experimental backend in Bionic, 36 GB floor, quantization-recipe note.
7. **Verdict** — for Qwen3.8-27B / Qwen3.6-35B-A3B agentic work on M3+ Macs
   with 48 GB+: is the speedup worth an experimental backend? For everyone
   else (other models, smaller Macs): MLX remains the general-purpose choice.

## Troubleshooting

- **`mlx_lm: command not found`** — `uv tool install mlx-lm`, then check
  `~/.local/bin` is on your PATH.
- **Splash readiness times out** — first run downloads 17.4 GB and tunes to
  your machine; raise the wait with `READY_TIMEOUT=3600 ./run_engine.sh splash`.
  Check `results/splash-server.log`.
- **TTFT looks equal to total time** — the client is buffering; make sure you
  run `benchmark.py` as-is (it reads the stream line-by-line).
- **Port already in use** — stop the other engine (or Bionic's local server if
  it grabbed port 8000/8080).
