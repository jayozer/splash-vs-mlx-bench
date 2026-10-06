# Benchmark summary: MLX-4bit

- model: `qwen3.8-27b-mlx@4bit`
- load config: context=40960 parallel=4
- peak server RSS: **0.2 GB**
- date: 2026-10-06 11:49 PDT

| Scenario | TTFT (ms) median | Decode (tok/s) median | ITL p95 (ms) | Total (s) median |
|---|---|---|---|---|
| short | 292.7 | 31.1 | 95.8 | 9.79 |
| medium | 289.0 | 29.2 | 101.7 | 10.39 |
| long | 754.0 | 22.4 | 134.8 | 11.92 |

**Long-context (~32K) detail:**

- First long request after load (prefill probe): **40177.9 ms**
  - exact prompt size: 19984 tokens -> prefill ~497 tok/s
- Cold TTFT (streaming, unseen prompt): **36046.3 ms**
- Cached TTFT (replay): **715.2 ms**

**Concurrency (4 parallel short):**

- Aggregate: **49.8 tok/s**
- Per-request TTFT median: 1353.1 ms
- Per-request decode median: 13.3 tok/s
