# Benchmark summary: Splash-4bit

- model: `qwen3.8-27b-splash`
- load config: context=40960 parallel=None
- peak server RSS: **0.3 GB**
- date: 2026-10-06 11:53 PDT

| Scenario | TTFT (ms) median | Decode (tok/s) median | ITL p95 (ms) | Total (s) median |
|---|---|---|---|---|
| short | 148.1 | 54.0 | 74.4 | 5.67 |
| medium | 158.6 | 45.9 | 81.8 | 6.67 |
| long | 185.6 | 53.6 | 68.5 | 4.94 |

**Long-context (~32K) detail:**

- First long request after load (prefill probe): **40425.0 ms**
  - exact prompt size: 19942 tokens -> prefill ~493 tok/s
- Cold TTFT (streaming, unseen prompt): **37667.8 ms**
- Cached TTFT (replay): **182.2 ms**

**Concurrency (4 parallel short):**

- Aggregate: **59.5 tok/s**
- Per-request TTFT median: 6424.5 ms
- Per-request decode median: 61.6 tok/s
