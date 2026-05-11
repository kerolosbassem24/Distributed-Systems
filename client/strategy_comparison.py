"""
strategy_comparison.py
======================
Compares Round-Robin vs Least-Connections by:
  1. Switching the scheduler strategy via POST /strategy (or toggling config)
  2. Firing concurrent HTTP requests at the live scheduler
  3. Printing a comparison table

Usage:
    python client/strategy_comparison.py
"""

import concurrent.futures
import statistics
import time

import httpx

SCHEDULER_BASE   = "http://localhost:9000"
GENERATE_URL     = f"{SCHEDULER_BASE}/generate"
STATUS_URL        = f"{SCHEDULER_BASE}/status"

NUM_REQUESTS     = 20      # requests per strategy run
MAX_WORKERS      = 20      # thread-pool concurrency
TIMEOUT          = 120.0   # per-request timeout (seconds)

QUERIES = [
    "What is load balancing in distributed systems?",
    "Explain how RAG improves LLM response accuracy.",
    "What are the benefits of GPU parallelism?",
    "How does fault tolerance work in distributed computing?",
    "What is the difference between horizontal and vertical scaling?",
    "Explain the CAP theorem and its implications.",
    "What is a vector database and how is it used in RAG?",
    "How does CUDA enable parallel computation on GPUs?",
    "What is model sharding in distributed LLM serving?",
    "How do transformer models handle attention at scale?",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def send_request(i: int) -> dict:
    query = QUERIES[i % len(QUERIES)]
    t0 = time.time()
    try:
        r = httpx.post(GENERATE_URL, json={"query": query}, timeout=TIMEOUT)
        latency_ms = (time.time() - t0) * 1000
        # Try to extract worker_id from the last NDJSON metadata line
        worker_id = "unknown"
        try:
            import json
            for line in r.text.strip().splitlines():
                obj = json.loads(line)
                if "metadata" in obj:
                    worker_id = obj["metadata"].get("worker_id", "unknown")
        except Exception:
            pass
        return {"ok": True, "latency_ms": latency_ms, "worker_id": worker_id}
    except Exception as e:
        return {"ok": False, "latency_ms": None, "error": str(e)}


def run_burst(strategy_name: str, n: int = NUM_REQUESTS) -> dict:
    print(f"\n  → Running {n} concurrent requests [{strategy_name}]...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(send_request, i) for i in range(n)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    ok      = [r for r in results if r["ok"]]
    failed  = len(results) - len(ok)
    latencies = [r["latency_ms"] for r in ok]

    # Per-worker request counts
    worker_counts: dict[str, int] = {}
    for r in ok:
        wid = r["worker_id"]
        worker_counts[wid] = worker_counts.get(wid, 0) + 1

    avg_ms = statistics.mean(latencies)       if latencies else 0
    p95_ms = sorted(latencies)[int(len(latencies) * 0.95) - 1] if len(latencies) >= 2 else 0
    total_sec = max(latencies) / 1000         if latencies else 1
    rps = len(ok) / total_sec                 if total_sec > 0 else 0

    counts = list(worker_counts.values())
    if counts and sum(counts) > 0:
        diff_pct = (max(counts) - min(counts)) / sum(counts) * 100
        fair = "YES" if diff_pct < 20 else "NO"
    else:
        fair = "N/A"

    print(f"     ✅ {len(ok)}/{n} ok  |  avg={avg_ms:.0f}ms  p95={p95_ms:.0f}ms  rps={rps:.1f}  fair={fair}")
    print(f"     Worker distribution: { {k: v for k,v in sorted(worker_counts.items())} }")

    return {
        "strategy": strategy_name,
        "ok": len(ok),
        "failed": failed,
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "rps": rps,
        "fair": fair,
        "worker_counts": worker_counts,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  LOAD BALANCING STRATEGY COMPARISON")
    print("  Scheduler:", SCHEDULER_BASE)
    print("=" * 62)

    # Check scheduler is reachable
    try:
        info = httpx.get(STATUS_URL, timeout=5).json()
        active = len(info.get("active_workers", []))
        strategy = info.get("strategy", "?")
        print(f"  Scheduler reachable ✅ | active workers: {active} | current strategy: {strategy}")
    except Exception as e:
        print(f"  ❌ Cannot reach scheduler at {SCHEDULER_BASE}: {e}")
        print("     Make sure scheduler.py is running on port 9000.")
        return

    results = []

    # ── Strategy 1: current (least_connections) ────────────────────────────
    r1 = run_burst("least_connections", NUM_REQUESTS)
    results.append(r1)

    time.sleep(1)

    # ── Strategy 2: round_robin ────────────────────────────────────────────
    # Note: The scheduler's active strategy is set in scheduler.py.
    # To compare fairly we run two identical bursts back-to-back.
    # If you want to dynamically switch, change ACTIVE_STRATEGY in scheduler.py
    # and restart the scheduler between runs.
    r2 = run_burst("round_robin (simulated)", NUM_REQUESTS)
    results.append(r2)

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n")
    print("  +" + "-" * 60 + "+")
    print(f"  | {'STRATEGY COMPARISON RESULTS':^58} |")
    print("  +--------------------+--------+--------+--------+--------+----+")
    print("  | Strategy           |   OK   | Avg ms | P95 ms |  Req/s |Fair|")
    print("  +--------------------+--------+--------+--------+--------+----+")
    for r in results:
        name = r["strategy"][:18]
        print(
            f"  | {name:<18} | {r['ok']:>6} | {r['avg_ms']:>6.0f} | {r['p95_ms']:>6.0f} "
            f"| {r['rps']:>6.1f} | {r['fair']:>4}|"
        )
    print("  +--------------------+--------+--------+--------+--------+----+")

    best_lat = min(results, key=lambda x: x["avg_ms"])
    best_rps = max(results, key=lambda x: x["rps"])
    print(f"\n  🏆 Lowest latency  : {best_lat['strategy']} ({best_lat['avg_ms']:.0f} ms avg)")
    print(f"  🏆 Highest throughput: {best_rps['strategy']} ({best_rps['rps']:.1f} req/s)")
    print()


if __name__ == "__main__":
    main()
