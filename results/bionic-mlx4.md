# Benchmark summary: MLX-4bit

- base_url: `http://localhost:1234/v1`
- model: `qwen3.8-27b-mlx@4bit`
- peak server RSS: **0.7 GB** (pid 97713)
- date: 2026-10-05 11:20 PDT

| Scenario | TTFT (ms) median | Decode (tok/s) median | ITL p95 (ms) | Total (s) median |
|---|---|---|---|---|
| short | 337.9 | 27.5 | 105.6 | 11.42 |
| medium | 1494.8 | 10.2 | 276.3 | 30.11 |
| long | 1909.2 | 8.4 | 397.7 | 31.64 |

**Long-context (~32K) detail:**

- Cold TTFT: **52630.0 ms** (prefill ~n/a tok/s)
- Cached TTFT (replay): **1605.0 ms**

**Concurrency (4 parallel short):**

- Aggregate: **25.4 tok/s**
- Per-request TTFT median: 2531.6 ms
- Per-request decode median: 6.8 tok/s

Raw per-run data: `results/bionic-mlx4.jsonl`
