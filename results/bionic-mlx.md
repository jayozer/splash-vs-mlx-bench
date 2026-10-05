# Benchmark summary: MLX-4bit

- model: `qwen3.8-27b-mlx@4bit`
- load config: context=262144 parallel=4
- peak server RSS: **0.4 GB**
- date: 2026-10-05 12:23 PDT

| Scenario | TTFT (ms) median | Decode (tok/s) median | ITL p95 (ms) | Total (s) median |
|---|---|---|---|---|
| short | 624.0 | 25.2 | 113.5 | 12.35 |
| medium | 5884.0 | 24.7 | 118.2 | 17.73 |
| long | 75093.4 | 6.8 | 542.3 | 112.41 |

**Long-context (~32K) detail:**

- First long request after load (prefill probe): **85764.0 ms**
  - exact prompt size: 19984 tokens -> prefill ~233 tok/s
- Cold TTFT (streaming, unseen prompt): **75093.4 ms**
