# Combined comparison

| Metric | MLX-4bit | Splash-4bit |
|---|---|---|
| Decode short (tok/s) | 31.1 | 54.0 |
| Decode medium ~4K (tok/s) | 29.2 | 45.9 |
| Decode long ~32K (tok/s) | 22.4 | 53.6 |
| TTFT short (ms) | 292.7 | 148.1 |
| Cold TTFT ~32K (ms) | 36046.3 | 37667.8 |
| Cached TTFT ~32K (ms) | 715.2 | 182.2 |
| Prefill ~32K (ms) | 40177.9 | 40425.0 |
| Concurrency 4x aggregate (tok/s) | 49.8 | 59.5 |
