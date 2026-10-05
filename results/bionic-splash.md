# Benchmark summary: Splash-4bit

- model: `qwen3.8-27b-splash`
- load config: context=40960 parallel=None
- peak server RSS: **0.3 GB**
- date: 2026-10-05 12:27 PDT

| Scenario | TTFT (ms) median | Decode (tok/s) median | ITL p95 (ms) | Total (s) median |
|---|---|---|---|---|
| short | 139.7 | 63.1 | 70.0 | 4.86 |
| medium | 4817.0 | 49.9 | 70.0 | 10.79 |
| long | 36769.1 | 51.1 | 72.2 | 41.76 |

**Long-context (~32K) detail:**

- First long request after load (prefill probe): **36570.9 ms**
  - exact prompt size: 19942 tokens -> prefill ~545 tok/s
- Cold TTFT (streaming, unseen prompt): **36769.1 ms**
