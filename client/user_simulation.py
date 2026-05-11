"""
user_simulation.py
==================
Simulates 100, 250, 500 and 1000 concurrent users against the live
Distributed LLM System API and reports:
  • Latency    – avg / P50 / P95 / P99
  • Throughput – requests / second
  • GPU Usage  – memory %, device name (polled from /workers endpoint)
  • Success    – ok / failed counts

Usage
-----
  # 1. Make sure app.py is running:
  #       python app.py          (from project root)

  # 2. Run this script:
  #       python client/user_simulation.py

  # Optional flags:
  #       --url   http://host:port   (default: http://localhost:8000)
  #       --scales 100 500 1000      (override user counts)
  #       --timeout 120              (per-request timeout, seconds)
  #       --out results/             (directory for CSV + JSON output)
  #       --workers 64               (max thread-pool size, default 64)
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── optional colour support ────────────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    G  = colorama.Fore.GREEN
    Y  = colorama.Fore.YELLOW
    C  = colorama.Fore.CYAN
    R  = colorama.Fore.RED
    B  = colorama.Fore.BLUE
    M  = colorama.Fore.MAGENTA
    W  = colorama.Fore.WHITE
    DIM= colorama.Style.DIM
    RST= colorama.Style.RESET_ALL
    BOLD = colorama.Style.BRIGHT
except ImportError:
    G = Y = C = R = B = M = W = DIM = RST = BOLD = ""

try:
    import httpx
except ImportError:
    print("httpx is required.  Run:  pip install httpx")
    sys.exit(1)

# ── sample queries (varied so every worker gets real work) ─────────────────────
SAMPLE_QUERIES = [
    "What is load balancing in distributed systems?",
    "Explain how RAG improves LLM response accuracy.",
    "What are the benefits of GPU parallelism for AI inference?",
    "How does fault tolerance work in distributed computing?",
    "What is the difference between horizontal and vertical scaling?",
    "Explain the CAP theorem and its implications.",
    "What is a vector database and how is it used in RAG?",
    "How does CUDA enable parallel computation on GPUs?",
    "What is model sharding in distributed LLM serving?",
    "How do transformer models handle attention at scale?",
    "What is the role of a master node in a GPU cluster?",
    "Explain round robin vs least connections load balancing.",
    "What metrics are used to evaluate distributed system performance?",
    "How do heartbeat mechanisms detect node failure?",
    "What is tensor parallelism in model serving?",
    "How does quantization reduce GPU memory usage for LLMs?",
    "Explain the difference between batch and streaming inference.",
    "What is KV cache and how does it speed up LLM inference?",
    "How does consistent hashing help in distributed caching?",
    "What are the tradeoffs between throughput and latency?",
]

# ── per-request worker ─────────────────────────────────────────────────────────

def send_one(args: tuple) -> dict:
    """Fire a single /query POST and return timing + metadata."""
    base_url, user_id, timeout = args
    query = SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)]
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{base_url}/query",
            json={"query": query, "request_id": user_id},
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if r.status_code == 200:
            body = r.json()
            return {
                "ok": True,
                "latency_ms": elapsed_ms,
                "worker_id": body.get("worker_id", -1),
                "gpu_id": body.get("gpu_id", -1),
                "strategy": body.get("strategy_used", "?"),
            }
        else:
            return {"ok": False, "latency_ms": elapsed_ms, "status_code": r.status_code}

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "latency_ms": elapsed_ms, "error": str(exc)}


# ── GPU snapshot from /workers ─────────────────────────────────────────────────

def fetch_gpu_snapshot(base_url: str, timeout: float = 5.0) -> list[dict]:
    """Return worker GPU stats from the /workers endpoint."""
    try:
        r = httpx.get(f"{base_url}/workers", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def fetch_system_metrics(base_url: str, timeout: float = 5.0) -> dict:
    """Return aggregated metrics from the /metrics endpoint."""
    try:
        r = httpx.get(f"{base_url}/metrics", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


# ── single scale-level simulation ─────────────────────────────────────────────

def run_scale(base_url: str, num_users: int, timeout: float, max_workers: int) -> dict:
    """
    Fire `num_users` concurrent requests, collect results, poll GPU stats
    before and after, and return a rich result dict.
    """
    bar_width = 40

    # Pre-run GPU snapshot
    gpu_before = fetch_gpu_snapshot(base_url)

    print(f"\n{BOLD}{C}  ┌─ {num_users:>5} Users {'─' * 38}{RST}")
    print(f"{DIM}  │  Firing {num_users} concurrent requests …{RST}")

    args = [(base_url, i, timeout) for i in range(num_users)]

    results   = []
    completed = 0
    lock      = threading.Lock()

    # Progress reporter runs in a side thread
    stop_progress = threading.Event()

    def _progress():
        while not stop_progress.is_set():
            with lock:
                done = completed
            pct  = done / num_users
            fill = int(bar_width * pct)
            bar  = ("█" * fill) + ("░" * (bar_width - fill))
            print(f"\r  │  [{G}{bar}{RST}] {done:>5}/{num_users}", end="", flush=True)
            time.sleep(0.25)

    prog_thread = threading.Thread(target=_progress, daemon=True)
    prog_thread.start()

    wall_t0 = time.perf_counter()

    actual_pool = min(max_workers, num_users)
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_pool) as pool:
        futures = {pool.submit(send_one, a): a for a in args}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
            with lock:
                completed += 1

    wall_elapsed = time.perf_counter() - wall_t0

    stop_progress.set()
    prog_thread.join()
    print()  # newline after progress bar

    # Post-run GPU snapshot
    gpu_after = fetch_gpu_snapshot(base_url)

    # ── crunch numbers ──────────────────────────────────────────────────
    ok_results  = [r for r in results if r.get("ok")]
    bad_results = [r for r in results if not r.get("ok")]

    latencies_ms = sorted(r["latency_ms"] for r in ok_results)
    n_ok = len(ok_results)
    n_bad = len(bad_results)

    def pct(arr, p):
        if not arr:
            return 0.0
        idx = max(0, math.ceil(len(arr) * p / 100) - 1)
        return arr[idx]

    avg_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p50    = pct(latencies_ms, 50)
    p95    = pct(latencies_ms, 95)
    p99    = pct(latencies_ms, 99)
    p_min  = latencies_ms[0]  if latencies_ms else 0.0
    p_max  = latencies_ms[-1] if latencies_ms else 0.0

    # Throughput: total successful reqs / wall-clock time
    throughput_rps = n_ok / wall_elapsed if wall_elapsed > 0 else 0.0

    # Worker distribution
    worker_dist: dict[str, int] = {}
    for r in ok_results:
        wid = str(r.get("worker_id", "?"))
        worker_dist[wid] = worker_dist.get(wid, 0) + 1

    # GPU utilisation – use the post-run snapshot (peak-ish)
    gpu_stats = []
    for w in gpu_after:
        gpu_stats.append({
            "worker_id"      : w.get("worker_id"),
            "gpu_id"         : w.get("gpu_id"),
            "status"         : w.get("status"),
            "active_requests": w.get("active_requests", 0),
            "total_processed": w.get("total_processed", 0),
            "device_name"    : w.get("device_name", "N/A"),
            "gpu_memory_pct" : w.get("gpu_memory_pct", 0.0),
            "gpu_mem_used_gb": w.get("gpu_memory_used_gb", 0.0),
            "gpu_mem_total_gb": w.get("gpu_memory_total_gb", 0.0),
        })

    return {
        "num_users"     : num_users,
        "wall_sec"      : round(wall_elapsed, 3),
        "total"         : len(results),
        "ok"            : n_ok,
        "failed"        : n_bad,
        "success_rate"  : round(n_ok / len(results) * 100, 1) if results else 0.0,
        "avg_ms"        : round(avg_ms, 1),
        "p50_ms"        : round(p50, 1),
        "p95_ms"        : round(p95, 1),
        "p99_ms"        : round(p99, 1),
        "min_ms"        : round(p_min, 1),
        "max_ms"        : round(p_max, 1),
        "throughput_rps": round(throughput_rps, 2),
        "worker_dist"   : worker_dist,
        "gpu_stats"     : gpu_stats,
    }


# ── pretty terminal report ─────────────────────────────────────────────────────

def print_scale_summary(res: dict):
    u    = res["num_users"]
    ok   = res["ok"]
    bad  = res["failed"]
    sr   = res["success_rate"]
    rps  = res["throughput_rps"]

    # Colour-code success rate
    sr_col = G if sr >= 99 else Y if sr >= 95 else R

    print(f"\n{BOLD}{C}  ╔═══════════════════════ {u:>5} USERS ═══════════════════════╗{RST}")
    print(f"  ║ {'Requests':30} {ok:>5} ok  /  {bad:>5} failed  ({sr_col}{sr:.1f}%{RST})  ║")
    print(f"  ║ {'Wall-clock time':30} {res['wall_sec']:>8.2f} s                   ║")
    print(f"  ╠═══════════════════ LATENCY (ms) ═══════════════════════╣")
    print(f"  ║  Min  = {res['min_ms']:>8.1f}   Avg  = {res['avg_ms']:>8.1f}   Max  = {res['max_ms']:>8.1f}  ║")
    print(f"  ║  P50  = {res['p50_ms']:>8.1f}   P95  = {res['p95_ms']:>8.1f}   P99  = {res['p99_ms']:>8.1f}  ║")
    print(f"  ╠═══════════════════ THROUGHPUT ═════════════════════════╣")
    print(f"  ║  {rps:>8.2f} requests/second                               ║")

    # Worker distribution
    print(f"  ╠═══════════════════ WORKER DISTRIBUTION ════════════════╣")
    dist = res["worker_dist"]
    total_ok = sum(dist.values()) if dist else 1
    for wid, cnt in sorted(dist.items()):
        pct_w = cnt / total_ok * 100
        bar   = "█" * int(pct_w / 5)
        print(f"  ║  Worker {wid}: {cnt:>5} req  ({pct_w:5.1f}%)  {bar:<20}          ║")

    # GPU utilization
    print(f"  ╠═══════════════════ GPU UTILIZATION ════════════════════╣")
    gpu_stats = res["gpu_stats"]
    if not gpu_stats:
        print(f"  ║  (no GPU stats – /workers returned empty)               ║")
    for g in gpu_stats:
        mem_pct = g["gpu_memory_pct"]
        mem_col = G if mem_pct < 70 else Y if mem_pct < 90 else R
        dev     = str(g["device_name"])[:22]
        status  = g["status"]
        st_col  = G if status == "active" else R
        print(f"  ║  Worker {g['worker_id']} (GPU {g['gpu_id']}) [{st_col}{status:6}{RST}] {dev:<22}  ║")
        if mem_pct > 0:
            mem_bar = "█" * int(mem_pct / 5)
            print(f"  ║    Mem: {mem_col}{mem_pct:5.1f}%{RST} [{mem_bar:<20}]   "
                  f"{g['gpu_mem_used_gb']:.1f}/{g['gpu_mem_total_gb']:.1f} GB    ║")
        else:
            print(f"  ║    Mem: {mem_col}mock/CPU mode (no real GPU data){RST}        ║")
        print(f"  ║    Processed total: {g['total_processed']:>6}                          ║")

    print(f"  ╚═══════════════════════════════════════════════════════╝")


def print_comparison_table(all_results: list[dict]):
    """Print the final side-by-side comparison table."""
    print(f"\n\n{BOLD}{M}{'=' * 80}{RST}")
    print(f"{BOLD}{M}{'  MULTI-SCALE SIMULATION — COMPARISON SUMMARY':^80}{RST}")
    print(f"{BOLD}{M}{'=' * 80}{RST}")

    hdr = (
        f"  {'Users':>6} │ {'OK':>5} │ {'Fail':>5} │ {'Rate%':>6} │ "
        f"{'Avg ms':>7} │ {'P95 ms':>7} │ {'P99 ms':>7} │ {'Req/s':>8}"
    )
    sep = "  " + "─" * (len(hdr) - 2)
    print(f"\n{BOLD}{hdr}{RST}")
    print(sep)

    for r in all_results:
        sr     = r["success_rate"]
        sr_col = G if sr >= 99 else Y if sr >= 95 else R
        print(
            f"  {r['num_users']:>6} │ {r['ok']:>5} │ {r['failed']:>5} │ "
            f"{sr_col}{sr:>6.1f}{RST} │ "
            f"{r['avg_ms']:>7.1f} │ {r['p95_ms']:>7.1f} │ {r['p99_ms']:>7.1f} │ "
            f"{r['throughput_rps']:>8.2f}"
        )

    print(sep)

    # Best/Worst highlights
    if all_results:
        best_rps  = max(all_results, key=lambda x: x["throughput_rps"])
        best_lat  = min(all_results, key=lambda x: x["avg_ms"] if x["avg_ms"] > 0 else float("inf"))
        best_sr   = max(all_results, key=lambda x: x["success_rate"])

        print(f"\n  {G}🏆 Best throughput  :{RST} {best_rps['num_users']:>5} users → {best_rps['throughput_rps']:.2f} req/s")
        print(f"  {G}🏆 Lowest avg latency:{RST} {best_lat['num_users']:>5} users → {best_lat['avg_ms']:.1f} ms")
        print(f"  {G}🏆 Best success rate :{RST} {best_sr['num_users']:>5} users → {best_sr['success_rate']:.1f}%")

    print(f"\n{BOLD}{M}{'=' * 80}{RST}")


# ── save results ───────────────────────────────────────────────────────────────

def save_results(all_results: list[dict], out_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = os.path.join(out_dir, f"simulation_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV (flat, one row per scale)
    csv_path = os.path.join(out_dir, f"simulation_{ts}.csv")
    flat_keys = ["num_users", "ok", "failed", "success_rate",
                 "avg_ms", "p50_ms", "p95_ms", "p99_ms",
                 "min_ms", "max_ms", "throughput_rps", "wall_sec"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in flat_keys})

    print(f"\n  {G}✅ Results saved:{RST}")
    print(f"     JSON → {json_path}")
    print(f"     CSV  → {csv_path}")


# ── connectivity check ─────────────────────────────────────────────────────────

def check_server(base_url: str, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            status   = data.get("status", "?")
            active   = data.get("active_workers", 0)
            total    = data.get("total_workers", 0)
            st_col   = G if status == "healthy" else Y if status == "degraded" else R
            print(f"  {G}✅ Server reachable{RST} — "
                  f"status={st_col}{status}{RST}, "
                  f"workers={active}/{total} active")
            return True
        print(f"  {R}❌ Server replied {r.status_code}{RST}")
        return False
    except Exception as e:
        print(f"  {R}❌ Cannot reach {base_url}: {e}{RST}")
        return False


# ── CLI entry ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-scale user simulation for the Distributed LLM System"
    )
    p.add_argument("--url",     default="http://localhost:8000",
                   help="Base URL of app.py (default: http://localhost:8000)")
    p.add_argument("--scales",  nargs="+", type=int,
                   default=[100, 250, 500, 1000],
                   help="List of user counts to simulate (default: 100 250 500 1000)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Per-request timeout in seconds (default: 120)")
    p.add_argument("--workers", type=int, default=64,
                   help="Max thread-pool size (default: 64)")
    p.add_argument("--out",     default="results",
                   help="Output directory for JSON/CSV (default: ./results)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving results to disk")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{BOLD}{B}{'=' * 64}{RST}")
    print(f"{BOLD}{B}{'  DISTRIBUTED LLM — USER SIMULATION BENCHMARK':^64}{RST}")
    print(f"{BOLD}{B}{'=' * 64}{RST}")
    print(f"  Target  : {C}{args.url}{RST}")
    print(f"  Scales  : {Y}{args.scales}{RST}")
    print(f"  Timeout : {args.timeout}s / request")
    print(f"  Threads : up to {args.workers} per scale")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Connectivity ──────────────────────────────────────────────────
    print(f"{BOLD}  Checking server connectivity …{RST}")
    if not check_server(args.url):
        print(f"\n  {R}Aborting. Start app.py first:{RST}")
        print(f"     cd llm_distributed_system")
        print(f"     python app.py")
        sys.exit(1)

    # Show initial GPU state
    print(f"\n{BOLD}  Initial GPU/Worker state:{RST}")
    workers_initial = fetch_gpu_snapshot(args.url)
    if workers_initial:
        for w in workers_initial:
            mem_pct = w.get("gpu_memory_pct", 0.0)
            print(f"    Worker {w['worker_id']} (GPU {w['gpu_id']}) "
                  f"[{w.get('status', '?')}] "
                  f"device={w.get('device_name', 'N/A')} "
                  f"mem={mem_pct:.1f}%")
    else:
        print(f"    {Y}(no worker data returned){RST}")

    # ── Run each scale ────────────────────────────────────────────────
    all_results = []
    for scale in args.scales:
        res = run_scale(
            base_url=args.url,
            num_users=scale,
            timeout=args.timeout,
            max_workers=args.workers,
        )
        print_scale_summary(res)
        all_results.append(res)

        # Brief cooldown between runs so workers drain
        if scale != args.scales[-1]:
            cooldown = max(3, min(10, scale // 100))
            print(f"\n  {DIM}↺ Cooling down {cooldown}s before next scale …{RST}")
            time.sleep(cooldown)

    # ── Summary table ─────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save ──────────────────────────────────────────────────────────
    if not args.no_save:
        save_results(all_results, args.out)

    print(f"\n  {G}Simulation complete.{RST}  {datetime.now().strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
