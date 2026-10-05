# Combined comparison

| Metric | MLX-4bit | Splash-4bit |
|---|---|---|
| Decode short (tok/s) | 25.2 | 63.1 |
| Decode medium ~4K (tok/s) | 24.7 | 49.9 |
| Decode long ~32K (tok/s) | 6.8 | 51.1 |
| TTFT short (ms) | 624.0 | 139.7 |
| Cold TTFT ~32K (ms) | 75093.4 | 36769.1 |
| Cached TTFT ~32K (ms) | n/a | n/a |
| Prefill ~32K (ms) | 85764.0 | 36570.9 |
| Concurrency 4x aggregate (tok/s) | n/a | n/a |
