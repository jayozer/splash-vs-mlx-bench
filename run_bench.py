#!/usr/bin/env python3
"""
Qwen3.8-27B engine benchmark — MLX vs Splash, through LM Studio (Bionic).

Single file, stdlib only. Copy this one script to your Mac and run:

    python3 run_bench.py                      # auto-detect models, defaults
    python3 run_bench.py --quick              # fast smoke test (~2 min)
    python3 run_bench.py --reverse            # Splash first (thermal fairness)
    CONTEXT=8192 PARALLEL=1 python3 run_bench.py   # tune for smaller machines
    python3 run_bench.py --mlx KEY --splash KEY    # override model detection

What it does
  1. Prints machine info (chip / RAM / macOS) for your blog's setup section
  2. Starts the LM Studio local server if needed (port 1234)
  3. Unloads ALL loaded models and VERIFIES the list is empty — nothing else
     may sit in memory while an engine is measured
  4. Per engine: load model fresh -> wait for ready -> run the suite
  5. Writes results/*.jsonl + results/*.md and prints a combined comparison

Suite per engine
  short       ~100-token prompt, up to 300 output tokens (x3)
  medium      ~4K-token prompt,  up to 300 output tokens (x3)
  long        ~32K-token prompt, up to 256 output tokens (x3)
                preceded by two warmups with different prompts, then a
                non-streaming prefill probe (max_tokens=1) that absorbs any
                one-time engine costs and yields the exact prefill time +
                prompt token count. Measured run 1 = cold TTFT, runs 2-3
                replay the same prompt = cached TTFT (KV-cache reuse).
  concurrent  4 parallel short requests -> aggregate tok/s

Fairness notes (for the blog)
  * Both models must be 4-bit for an apples-to-apples comparison.
  * One model in memory at a time; the script enforces this.
  * Note which engine ran first (thermal drift). If the gap is small, re-run
    with --reverse and average.
  * Speculative decoding (DFlash 2 draft) is part of the Splash package, so
    this compares engines as shipped, not raw kernels.

Requires: LM Studio (Bionic) with the `lms` CLI on PATH, and both models
downloaded. macOS only (uses lsof/ps/sysctl for memory + machine info).
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
# Configuration
# --------------------------------------------------------------------------

PORT = int(os.environ.get("LMSTUDIO_PORT", "1234"))
BASE_URL = "http://localhost:%d/v1" % PORT
CONTEXT = int(os.environ.get("CONTEXT", "40960"))   # covers ~32K prompt + output
PARALLEL = int(os.environ.get("PARALLEL", "4"))     # needed for concurrency test
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

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
# LM Studio (lms) helpers
# --------------------------------------------------------------------------

def run_cmd(cmd, timeout=60):
    """Run a command, return (rc, stdout). Never prompts (stdin=/dev/null)."""
    try:
        p = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def lms_json(args):
    rc, out = run_cmd(["lms"] + args)
    if rc != 0:
        return None, out
    start = out.find("[")
    if start < 0:
        return None, out
    try:
        return json.loads(out[start:]), ""
    except ValueError:
        return None, out


def server_up():
    try:
        with urllib.request.urlopen(BASE_URL + "/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def loaded_identifiers():
    data, _ = lms_json(["ps", "--json"])
    if not data:
        return []
    return [m["identifier"] for m in data if m.get("status") not in ("unloaded",)]


def unload_all():
    """Unload every loaded model; verify the list is empty."""
    for _ in range(5):
        ids = loaded_identifiers()
        if not ids:
            return True
        for ident in ids:
            run_cmd(["lms", "unload", ident], timeout=120)
        time.sleep(3)
    return not loaded_identifiers()


def find_models():
    """Auto-detect the 4-bit MLX model and the Splash package."""
    data, _ = lms_json(["ls", "--json"])
    mlx_key = splash_key = None
    if data:
        for m in data:
            if m.get("type") != "llm":
                continue
            path = (m.get("path") or "").lower()
            bits = (m.get("quantization") or {}).get("bits")
            if "splash" in path:
                splash_key = m["modelKey"]
            elif ("qwen3.8-27b" in path or "qwen3_5" in (m.get("architecture") or "").lower()) \
                    and bits == 4:
                mlx_key = m["modelKey"]
    return mlx_key, splash_key


def load_model(key):
    """Load with context + parallel; fall back to no --parallel if rejected."""
    print("  loading %s (context=%d, parallel=%d) ..." % (key, CONTEXT, PARALLEL),
          flush=True)
    rc, out = run_cmd(["lms", "load", key, "-c", str(CONTEXT),
                       "--parallel", str(PARALLEL)], timeout=900)
    if rc != 0:
        print("  --parallel rejected, retrying without it", flush=True)
        rc, out = run_cmd(["lms", "load", key, "-c", str(CONTEXT)], timeout=900)
    if rc != 0:
        print("  LOAD FAILED:\n" + out[-2000:], file=sys.stderr)
        return False
    # Record the effective load config for the results metadata.
    data, _ = lms_json(["ps", "--json"])
    cfg = {}
    if data:
        for m in data:
            if m.get("modelKey") == key or m.get("identifier") == key:
                cfg = {"contextLength": m.get("contextLength"),
                       "parallel": m.get("parallel")}
    print("  loaded. effective config: %s" % (cfg or "unknown"), flush=True)
    return True


def wait_ready(key, deadline_s=900):
    print("  waiting for %s to accept requests ..." % key, flush=True)
    deadline = time.time() + deadline_s
    probe = json.dumps({
        "model": key,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "stream": False,
    }).encode()
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                BASE_URL + "/chat/completions", data=probe,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if b'"choices"' in r.read():
                    print("  ready.", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(5)
    print("  TIMED OUT waiting for model", file=sys.stderr)
    return False


# --------------------------------------------------------------------------
# Request measurement (streaming + non-streaming)
# --------------------------------------------------------------------------

def run_request(model, messages, max_tokens):
    """One streaming chat completion. Returns a metrics dict."""
    url = BASE_URL + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})

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
        "text_head": "".join(text_parts)[:120],
    }


def prefill_probe(model, messages):
    """Non-streaming max_tokens=1: exact prefill time + prompt token count."""
    url = BASE_URL + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.loads(r.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    usage = body.get("usage") or {}
    return {
        "prefill_time_s": elapsed,   # includes ~1 token of decode (~30-40 ms)
        "prompt_tokens": usage.get("prompt_tokens"),
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
                    self.peak_kb = max(self.peak_kb, int(out.split()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

def fmt(v, nd=3):
    if v is None:
        return "n/a"
    return ("%." + str(nd) + "f") % v


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def run_suite(model, label, quick=False):
    runs_per = 1 if quick else 3
    records = []

    def record(scenario, run_no, r):
        r.update(scenario=scenario, run=run_no, engine=label)
        records.append(r)

    # --- short / medium ----------------------------------------------------
    for name, filler_chars, max_tokens in (("short", 0, 300), ("medium", 16000, 300)):
        print("  [%s] warmup ..." % name, flush=True)
        run_request(model, build_messages(filler_chars, seed=9000), min(max_tokens, 64))
        for i in range(runs_per):
            print("  [%s] run %d/%d ..." % (name, i + 1, runs_per), flush=True)
            r = run_request(model, build_messages(filler_chars, seed=1000), max_tokens)
            record(name, i + 1, r)
            print("    ttft=%sms decode=%stok/s" % (
                fmt((r["ttft_s"] or 0) * 1000, 1),
                fmt(r["decode_tps"], 1) if r["decode_tps"] else "n/a"), flush=True)
            time.sleep(1.0)

    # --- long context ------------------------------------------------------
    LONG_FILLER = 128000
    print("  [long] warmup 1/2 ...", flush=True)
    run_request(model, build_messages(LONG_FILLER, seed=9001), 64)
    print("  [long] warmup 2/2 ...", flush=True)
    run_request(model, build_messages(LONG_FILLER, seed=9002), 64)

    print("  [long] prefill probe (non-streaming, max_tokens=1) ...", flush=True)
    probe = prefill_probe(model, build_messages(LONG_FILLER, seed=9003))
    record("prefill_probe", 1, probe)
    print("    prefill=%sms prompt_tokens=%s" % (
        fmt((probe["prefill_time_s"] or 0) * 1000, 1),
        probe["prompt_tokens"] if probe["prompt_tokens"] else "n/a"), flush=True)

    for i in range(runs_per):
        print("  [long] run %d/%d ..." % (i + 1, runs_per), flush=True)
        r = run_request(model, build_messages(LONG_FILLER, seed=1000), 256)
        record("long", i + 1, r)
        print("    ttft=%sms decode=%stok/s" % (
            fmt((r["ttft_s"] or 0) * 1000, 1),
            fmt(r["decode_tps"], 1) if r["decode_tps"] else "n/a"), flush=True)
        time.sleep(1.0)

    # --- concurrency ---------------------------------------------------------
    if not quick:
        print("  [concurrent] 4x parallel short ...", flush=True)

        def one(_):
            return run_request(model, build_messages(0, seed=2000), 256)

        with ThreadPoolExecutor(max_workers=4) as ex:
            conc = list(ex.map(one, range(4)))
        wall = max(r["total_time_s"] for r in conc)
        total_out = sum(r["output_tokens"] or 0 for r in conc)
        agg_tps = (total_out / wall) if wall else None
        for i, r in enumerate(conc):
            record("concurrent", i + 1, r)
        print("    aggregate=%stok/s over %ss" % (
            fmt(agg_tps, 1) if agg_tps else "n/a", fmt(wall, 2)), flush=True)

    return records


def summarize(label, model, load_cfg, peak_gb, records):
    lines = []
    lines.append("# Benchmark summary: %s" % label)
    lines.append("")
    lines.append("- model: `%s`" % model)
    lines.append("- load config: context=%s parallel=%s" % (
        load_cfg.get("contextLength", "?"), load_cfg.get("parallel", "?")))
    lines.append("- peak server RSS: **%.1f GB**" % peak_gb)
    lines.append("- date: %s" % time.strftime("%Y-%m-%d %H:%M %Z"))
    lines.append("")
    lines.append("| Scenario | TTFT (ms) median | Decode (tok/s) median | "
                 "ITL p95 (ms) | Total (s) median |")
    lines.append("|---|---|---|---|---|")

    for name in ("short", "medium", "long"):
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

    probe = [r for r in records if r["scenario"] == "prefill_probe"]
    long_rs = [r for r in records if r["scenario"] == "long"]
    lines.append("")
    lines.append("**Long-context (~32K) detail:**")
    lines.append("")
    if probe:
        p = probe[0]
        pt = p.get("prompt_tokens")
        lines.append("- First long request after load (prefill probe): **%s ms**"
                     % fmt((p["prefill_time_s"] or 0) * 1000, 1))
        if pt:
            lines.append("  - exact prompt size: %d tokens -> prefill ~%s tok/s"
                         % (pt, fmt(pt / p["prefill_time_s"], 0) if p["prefill_time_s"] else "n/a"))
    if long_rs:
        cold = [r for r in long_rs if r["run"] == 1]
        warm = [r for r in long_rs if r["run"] > 1]
        if cold:
            lines.append("- Cold TTFT (streaming, unseen prompt): **%s ms**"
                         % fmt((cold[0]["ttft_s"] or 0) * 1000, 1))
        if warm:
            lines.append("- Cached TTFT (replay): **%s ms**"
                         % fmt(med([r["ttft_s"] for r in warm]) * 1000.0, 1))

    conc = [r for r in records if r["scenario"] == "concurrent"]
    if conc:
        wall = max(r["total_time_s"] for r in conc)
        total_out = sum(r["output_tokens"] or 0 for r in conc)
        lines.append("")
        lines.append("**Concurrency (4 parallel short):**")
        lines.append("")
        lines.append("- Aggregate: **%s tok/s**" % fmt(total_out / wall, 1) if wall else "n/a")
        lines.append("- Per-request TTFT median: %s ms"
                     % fmt(med([r["ttft_s"] for r in conc]) * 1000.0, 1))
        lines.append("- Per-request decode median: %s tok/s"
                     % fmt(med([r["decode_tps"] for r in conc]), 1))
    return "\n".join(lines)


def combined_table(summaries):
    """Side-by-side comparison of the two engines."""
    labels = list(summaries.keys())
    if len(labels) < 2:
        return ""

    def get(label, scenario, field):
        rs = [r for r in summaries[label]["records"] if r["scenario"] == scenario]
        return med([r.get(field) for r in rs])

    def get_probe(label):
        rs = [r for r in summaries[label]["records"] if r["scenario"] == "prefill_probe"]
        return rs[0].get("prefill_time_s") if rs else None

    rows = [
        ("Decode short (tok/s)", "short", "decode_tps"),
        ("Decode medium ~4K (tok/s)", "medium", "decode_tps"),
        ("Decode long ~32K (tok/s)", "long", "decode_tps"),
        ("TTFT short (ms)", "short", "ttft_s"),
        ("Cold TTFT ~32K (ms)", None, None),   # special: long run 1
        ("Cached TTFT ~32K (ms)", None, None), # special: long runs 2+
        ("Prefill ~32K (ms)", None, None),     # special: probe
    ]

    def cold_ttft(label):
        rs = [r for r in summaries[label]["records"]
              if r["scenario"] == "long" and r["run"] == 1]
        return rs[0]["ttft_s"] if rs else None

    def cached_ttft(label):
        rs = [r for r in summaries[label]["records"]
              if r["scenario"] == "long" and r["run"] > 1]
        return med([r["ttft_s"] for r in rs]) if rs else None

    lines = ["# Combined comparison", ""]
    header = "| Metric | " + " | ".join(labels) + " |"
    lines.append(header)
    lines.append("|---" * (len(labels) + 1) + "|")

    def row(name, vals, scale=1.0, nd=1):
        cells = []
        for v in vals:
            cells.append(fmt(v * scale, nd) if v is not None else "n/a")
        lines.append("| %s | %s |" % (name, " | ".join(cells)))

    row("Decode short (tok/s)", [get(l, "short", "decode_tps") for l in labels])
    row("Decode medium ~4K (tok/s)", [get(l, "medium", "decode_tps") for l in labels])
    row("Decode long ~32K (tok/s)", [get(l, "long", "decode_tps") for l in labels])
    row("TTFT short (ms)", [get(l, "short", "ttft_s") for l in labels], 1000.0)
    row("Cold TTFT ~32K (ms)", [cold_ttft(l) for l in labels], 1000.0)
    row("Cached TTFT ~32K (ms)", [cached_ttft(l) for l in labels], 1000.0)
    row("Prefill ~32K (ms)", [get_probe(l) for l in labels], 1000.0)

    conc_rows = []
    for l in labels:
        rs = [r for r in summaries[l]["records"] if r["scenario"] == "concurrent"]
        if rs:
            wall = max(r["total_time_s"] for r in rs)
            conc_rows.append(sum(r["output_tokens"] or 0 for r in rs) / wall if wall else None)
        else:
            conc_rows.append(None)
    row("Concurrency 4x aggregate (tok/s)", conc_rows)

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def machine_info():
    info = {}
    try:
        rc, out = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        if rc == 0:
            info["chip"] = out.strip()
    except Exception:
        pass
    try:
        rc, out = run_cmd(["sysctl", "-n", "hw.memsize"], timeout=5)
        if rc == 0:
            info["ram_gb"] = int(out.strip()) // (1024 ** 3)
    except Exception:
        pass
    try:
        rc, out = run_cmd(["sw_vers", "-productVersion"], timeout=5)
        if rc == 0:
            info["macos"] = out.strip()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(
        description="MLX vs Splash benchmark through LM Studio (single file)")
    ap.add_argument("--mlx", help="model key for the MLX model (default: auto-detect)")
    ap.add_argument("--splash", help="model key for the Splash model (default: auto-detect)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 1 run per scenario, no concurrency")
    ap.add_argument("--reverse", action="store_true",
                    help="run Splash first (thermal fairness check)")
    args = ap.parse_args()

    print("=" * 64)
    print("Qwen3.8-27B engine benchmark (MLX vs Splash) via LM Studio")
    print("=" * 64, flush=True)

    # --- preflight ---------------------------------------------------------
    rc, _ = run_cmd(["lms", "--version"], timeout=10)
    if rc != 0:
        print("ERROR: `lms` CLI not found on PATH. Install LM Studio first.",
              file=sys.stderr)
        sys.exit(1)

    info = machine_info()
    print("Machine: %s | %s GB RAM | macOS %s" % (
        info.get("chip", "?"), info.get("ram_gb", "?"), info.get("macos", "?")),
        flush=True)

    if not server_up():
        print("Starting LM Studio local server ...", flush=True)
        run_cmd(["lms", "server", "start"], timeout=60)
        for _ in range(30):
            if server_up():
                break
            time.sleep(2)
    if not server_up():
        print("ERROR: local server did not come up on port %d" % PORT, file=sys.stderr)
        sys.exit(1)
    print("Server: up on %s" % BASE_URL, flush=True)

    mlx_key = args.mlx
    splash_key = args.splash
    if not (mlx_key and splash_key):
        d_mlx, d_splash = find_models()
        mlx_key = mlx_key or d_mlx
        splash_key = splash_key or d_splash
    if not mlx_key:
        print("ERROR: could not auto-detect a 4-bit Qwen3.8-27B MLX model.\n"
              "List your models with `lms ls` and pass --mlx KEY.", file=sys.stderr)
        sys.exit(1)
    if not splash_key:
        print("ERROR: could not auto-detect the Splash model.\n"
              "List your models with `lms ls` and pass --splash KEY.", file=sys.stderr)
        sys.exit(1)

    print("MLX model:    %s" % mlx_key, flush=True)
    print("Splash model: %s" % splash_key, flush=True)
    print("Context=%d  Parallel=%d  Quick=%s" % (CONTEXT, PARALLEL, args.quick),
          flush=True)

    order = [("Splash-4bit", splash_key, "splash"),
             ("MLX-4bit", mlx_key, "mlx")] if args.reverse else \
            [("MLX-4bit", mlx_key, "mlx"),
             ("Splash-4bit", splash_key, "splash")]

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = {}

    for label, key, tag in order:
        print("")
        print("=" * 64)
        print("ENGINE: %s (%s)" % (label, key))
        print("=" * 64, flush=True)

        if not unload_all():
            print("WARNING: could not fully clear loaded models before %s —\n"
                  "results may be contaminated by other resident models!" % label,
                  file=sys.stderr)

        if not load_model(key):
            print("ERROR: skipping %s (load failed)" % label, file=sys.stderr)
            continue

        if not wait_ready(key):
            print("ERROR: skipping %s (never became ready)" % label, file=sys.stderr)
            continue

        sampler = MemSampler(PORT)
        sampler.start()
        records = run_suite(key, label, quick=args.quick)
        sampler.stop()
        peak_gb = sampler.peak_kb / (1024.0 * 1024.0)

        data, _ = lms_json(["ps", "--json"])
        load_cfg = {}
        if data:
            for m in data:
                if m.get("modelKey") == key or m.get("identifier") == key:
                    load_cfg = {"contextLength": m.get("contextLength"),
                                "parallel": m.get("parallel")}

        summary = summarize(label, key, load_cfg, peak_gb, records)
        print("\n" + summary, flush=True)

        jsonl_path = os.path.join(OUT_DIR, "bionic-%s.jsonl" % tag)
        md_path = os.path.join(OUT_DIR, "bionic-%s.md" % tag)
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(md_path, "w") as f:
            f.write(summary + "\n")

        summaries[label] = {"records": records, "model": key}
        print("\nRaw data: %s" % jsonl_path, flush=True)

    if len(summaries) >= 2:
        table = combined_table(summaries)
        print("\n" + table, flush=True)
        with open(os.path.join(OUT_DIR, "comparison.md"), "w") as f:
            f.write(table + "\n")
        print("\nComparison written to %s" % os.path.join(OUT_DIR, "comparison.md"),
              flush=True)

    print("\nDone. The last model is still loaded — `lms unload` to free memory.")
    print("Tip: if the gap between engines is small, re-run with --reverse and")
    print("average the two orders to cancel thermal drift.")


if __name__ == "__main__":
    main()
