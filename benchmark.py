#!/usr/bin/env python3
"""Streaming speed benchmark for OpenAI-compatible local LLM servers.

Compares engines (e.g. MLX vs Splash) on the same machine and model by
measuring, per request:

  * TTFT           time from sending the request to the first token
                    (first content OR reasoning token)
  * decode tok/s   tokens per second during generation, from the span
                    between first and last token arrival
  * prefill tok/s  prompt_tokens / TTFT on a cold long-context run (estimate)
  * total time     end-to-end request duration
  * peak RSS       server process memory while the suite runs (lsof + ps)

Test matrix per engine:
  short       ~100-token prompt,   up to 300 output tokens (x3)
  medium      ~4K-token prompt,    up to 300 output tokens (x3)
  long        ~32K-token prompt,   up to 256 output tokens (x3)
                    run 1 is cold; runs 2-3 replay the same prompt, so they
                    measure cached TTFT (KV-cache reuse)
  concurrent  4 parallel short requests, up to 256 output tokens each

Each scenario is preceded by a warmup run that uses DIFFERENT filler text,
so the first measured long-context prompt is genuinely unseen by the cache.

Stdlib only; works on macOS system Python 3.9+.

Usage:
  python3 benchmark.py --base-url http://127.0.0.1:8080/v1 \
      --model mlx-community/Qwen3.8-27B-4bit --engine-name MLX \
      --out results/mlx.jsonl
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

QUESTION = (
    "Explain how speculative decoding works in large language model inference, "
    "covering the draft model, the verification step, and how accepted tokens "
    "are decided. Be thorough but stay under 300 words."
)

FILLER_WORDS = (
    "the local inference engine schedules each request against the shared "
    "unified memory budget while the kernel pipeline streams activations "
    "through attention blocks and feed forward layers on the apple silicon "
    "gpu every token that survives verification advances the cache pointer "
    "and the next draft block is proposed in a single fused pass so that "
    "the measured latency reflects the hardware rather than the runtime"
).split()


def make_filler(seed, target_chars):
    """Deterministic pseudo-English filler of roughly target_chars characters."""
    rng = random.Random(seed)
    words = []
    total = 0
    while total < target_chars:
        w = rng.choice(FILLER_WORDS)
        words.append(w)
        total += len(w) + 1
    return " ".join(words)


def build_messages(filler_chars, seed):
    filler = make_filler(seed, filler_chars) if filler_chars else ""
    content = (filler + "\n\n" if filler else "") + QUESTION
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------
# Single streaming request
# --------------------------------------------------------------------------

def run_request(base_url, model, messages, max_tokens):
    """One streaming chat completion. Returns a metrics dict."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=3600)

    ttft = None
    token_times = []
    usage = None
    text_parts = []

    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if content:
                text_parts.append(content)
            if content or reasoning:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)

    total_time = time.perf_counter() - t0
    resp.close()

    n_tokens = (usage or {}).get("completion_tokens") or len(token_times)
    prompt_tokens = (usage or {}).get("prompt_tokens")

    decode_tps = None
    itls = []
    if len(token_times) >= 2:
        span = token_times[-1] - token_times[0]
        if span > 0:
            decode_tps = max(n_tokens - 1, 0) / span
        itls = [b - a for a, b in zip(token_times, token_times[1:])]

    return {
        "ttft_s": ttft,
        "total_time_s": total_time,
        "output_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tps": decode_tps,
        "itl_median_ms": (statistics.median(itls) * 1000.0) if itls else None,
        "itl_p95_ms": (percentile(itls, 95) * 1000.0) if itls else None,
        "prefill_tps": (prompt_tokens / ttft) if (ttft and prompt_tokens) else None,
        "text_head": "".join(text_parts)[:120],
    }


def percentile(values, pct):
    if not values:
        return None
    vs = sorted(values)
    k = (len(vs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


# --------------------------------------------------------------------------
# Memory sampling (server process RSS via lsof + ps, macOS)
# --------------------------------------------------------------------------

class MemSampler(threading.Thread):
    def __init__(self, port, interval=0.5):
        super(MemSampler, self).__init__()
        self.daemon = True
        self.port = port
        self.interval = interval
        self.peak_kb = 0
        self.pid = None
        self._stop = threading.Event()

    def _find_pid(self):
        try:
            out = subprocess.run(
                ["lsof", "-nP", "-iTCP:%d" % self.port, "-sTCP:LISTEN"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
            ).stdout.decode("utf-8", "replace")
        except Exception:
            return None
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 2 and parts[1].isdigit():
                return int(parts[1])
        return None

    def run(self):
        self.pid = self._find_pid()
        if not self.pid:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(self.pid)],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
                ).stdout.decode("utf-8", "replace").strip()
                if out:
                    kb = int(out.split()[0])
                    self.peak_kb = max(self.peak_kb, kb)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

# (name, filler_chars, max_tokens, measured_runs)
SCENARIOS = [
    ("short", 0, 300, 3),
    ("medium", 16000, 300, 3),
    ("long", 128000, 256, 3),
]


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    return ("%." + str(nd) + "f") % v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True,
                    help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True, help="Model id the server expects")
    ap.add_argument("--engine-name", default="engine",
                    help="Label for this engine in the output (e.g. MLX, Splash)")
    ap.add_argument("--out", default="results/bench.jsonl",
                    help="Where to write the raw JSONL results")
    args = ap.parse_args()

    port = urllib.parse.urlparse(args.base_url).port
    sampler = MemSampler(port)

    print("== Benchmark: %s @ %s (model=%s)" %
          (args.engine_name, args.base_url, args.model), flush=True)

    # Sanity check: server reachable and model known.
    try:
        with urllib.request.urlopen(
                args.base_url.rstrip("/") + "/models", timeout=10) as r:
            models = json.loads(r.read().decode("utf-8"))
        ids = [m.get("id") for m in models.get("data", [])]
        print("Server models: %s" % ", ".join(ids), flush=True)
    except Exception as e:
        print("WARNING: could not list models from server: %s" % e, flush=True)

    sampler.start()
    records = []

    for name, filler_chars, max_tokens, runs in SCENARIOS:
        # Warmup with different filler so measured prompts stay cold.
        print("  [%s] warmup ..." % name, flush=True)
        run_request(args.base_url, args.model,
                    build_messages(filler_chars, seed=9000), min(max_tokens, 64))

        for i in range(runs):
            # Same prompt every measured run. For "long", the warmup used a
            # different seed, so run 1 is cold and runs 2-3 hit the KV cache.
            msgs = build_messages(filler_chars, seed=1000)
            print("  [%s] run %d/%d ..." % (name, i + 1, runs), flush=True)
            r = run_request(args.base_url, args.model, msgs, max_tokens)
            r.update(scenario=name, run=i + 1, engine=args.engine_name)
            records.append(r)
            print("    ttft=%sms decode=%stok/s total=%ss prompt_tokens=%s" % (
                fmt((r["ttft_s"] or 0) * 1000, 1),
                fmt(r["decode_tps"], 1) if r["decode_tps"] else "n/a",
                fmt(r["total_time_s"], 2),
                r["prompt_tokens"] if r["prompt_tokens"] else "n/a",
            ), flush=True)
            time.sleep(1.0)

    # Concurrency: 4 parallel short requests (agent-style load).
    print("  [concurrent] 4x parallel short ...", flush=True)

    def one(_):
        return run_request(args.base_url, args.model,
                           build_messages(0, seed=2000), 256)

    with ThreadPoolExecutor(max_workers=4) as ex:
        conc = list(ex.map(one, range(4)))

    wall = max(r["total_time_s"] for r in conc)
    total_out = sum(r["output_tokens"] or 0 for r in conc)
    agg_tps = (total_out / wall) if wall else None
    for i, r in enumerate(conc):
        r.update(scenario="concurrent", run=i + 1, engine=args.engine_name)
    records.extend(conc)
    print("    aggregate=%stok/s over %ss (%d tokens)" % (
        fmt(agg_tps, 1) if agg_tps else "n/a", fmt(wall, 2), total_out), flush=True)

    sampler.stop()
    peak_gb = sampler.peak_kb / (1024.0 * 1024.0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    def med(vals):
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    lines = []
    lines.append("# Benchmark summary: %s" % args.engine_name)
    lines.append("")
    lines.append("- base_url: `%s`" % args.base_url)
    lines.append("- model: `%s`" % args.model)
    lines.append("- peak server RSS: **%.1f GB** (pid %s)" %
                 (peak_gb, sampler.pid or "n/a"))
    lines.append("- date: %s" % time.strftime("%Y-%m-%d %H:%M %Z"))
    lines.append("")
    lines.append("| Scenario | TTFT (ms) median | Decode (tok/s) median | "
                 "ITL p95 (ms) | Total (s) median |")
    lines.append("|---|---|---|---|---|")

    for name, _, _, _ in SCENARIOS:
        rs = [r for r in records if r["scenario"] == name]
        ttft_ms = med([r["ttft_s"] for r in rs]) * 1000.0
        dec = med([r["decode_tps"] for r in rs])
        p95 = med([r["itl_p95_ms"] for r in rs])
        tot = med([r["total_time_s"] for r in rs])
        lines.append("| %s | %s | %s | %s | %s |" % (
            name, fmt(ttft_ms, 1) if ttft_ms else "n/a",
            fmt(dec, 1) if dec else "n/a",
            fmt(p95, 1) if p95 else "n/a",
            fmt(tot, 2) if tot else "n/a"))

    # Long-context cold vs cached breakdown.
    long_rs = [r for r in records if r["scenario"] == "long"]
    if long_rs:
        cold = [r for r in long_rs if r["run"] == 1]
        warm = [r for r in long_rs if r["run"] > 1]
        lines.append("")
        lines.append("**Long-context (~32K) detail:**")
        lines.append("")
        if cold:
            c = cold[0]
            lines.append("- Cold TTFT: **%s ms** (prefill ~%s tok/s)" % (
                fmt((c["ttft_s"] or 0) * 1000, 1),
                fmt(c["prefill_tps"], 0) if c["prefill_tps"] else "n/a"))
        if warm:
            w_ttft = med([r["ttft_s"] for r in warm]) * 1000.0
            lines.append("- Cached TTFT (replay): **%s ms**" % fmt(w_ttft, 1))

    if conc:
        c_ttft = med([r["ttft_s"] for r in conc]) * 1000.0
        c_dec = med([r["decode_tps"] for r in conc])
        lines.append("")
        lines.append("**Concurrency (4 parallel short):**")
        lines.append("")
        lines.append("- Aggregate: **%s tok/s**" % (fmt(agg_tps, 1) if agg_tps else "n/a"))
        lines.append("- Per-request TTFT median: %s ms" % (fmt(c_ttft, 1) if c_ttft else "n/a"))
        lines.append("- Per-request decode median: %s tok/s" % (fmt(c_dec, 1) if c_dec else "n/a"))

    lines.append("")
    lines.append("Raw per-run data: `%s`" % args.out)

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)

    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(summary + "\n")
    print("\nSummary written to %s" % md_path, flush=True)


if __name__ == "__main__":
    main()
