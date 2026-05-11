import threading
import time
import httpx
import sys
import asyncio

from config import NUM_USERS, LOAD_BALANCE_STRATEGY, MULTI_SCALE_TEST, FAILURE_TEST, STRATEGY_COMPARISON_TEST
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from fault_tolerance.health_monitor import HealthMonitor
from monitoring.metrics import MetricsCollector
from client.load_generator import run_load_test, run_scaled_load_tests
from rag.ingestor import ingest_documents, is_db_populated
from fault_tolerance.failure_test import run_failure_simulation_test

async def wait_for_workers(worker_urls, timeout=30):
    print(f"\n[Main] Waiting for {len(worker_urls)} workers to come online...")
    start_time = time.time()
    ready_workers = set()
    
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            for url in worker_urls:
                if url not in ready_workers:
                    try:
                        r = await client.get(f"{url}/status", timeout=2)
                        if r.status_code == 200:
                            ready_workers.add(url)
                            print(f"[Main] Worker {url} is online.")
                    except:
                        pass
            if len(ready_workers) == len(worker_urls):
                return True
            await asyncio.sleep(1)
        
    return False

async def main_async():
    # ── 1. Document ingestion ──────────────────────────────────────
    if not is_db_populated():
        print("\n[Main] Ingesting documents into vector DB...")
        # Running synchronous ingestion in a thread so we don't block the loop
        await asyncio.to_thread(ingest_documents, "data/documents/")
    else:
        print("[Main] Vector DB already populated, skipping ingestion.")

    # ── 2. Define Network Workers ──────────────────────────────────
    print("\n[Main] Connecting to network workers...")
    worker_urls = [
        "http://localhost:8001",
        "http://localhost:8002",
        "http://localhost:8003",
        "http://localhost:8004"
    ]
    
    if not await wait_for_workers(worker_urls):
        print("[Main] Error: Not all workers came online in time. Exiting.")
        return

    # ── 3. Load balancer ───────────────────────────────────────────
    lb = LoadBalancer(worker_urls, strategy="round_robin") # Default starting strategy

    # ── 4. Scheduler ───────────────────────────────────────────────
    scheduler = Scheduler(lb)

    # ── 5. Metrics collector ───────────────────────────────────────
    metrics = MetricsCollector()
    metrics.register_workers(worker_urls)

    # ── 6. Health monitor (background daemon) ─────────────────────
    monitor = HealthMonitor(worker_urls, lb, scheduler)
    monitor.start()

    # ── 7. Tests Dispatcher ───────────────────────────────────────
    if FAILURE_TEST:
        await run_failure_simulation_test(scheduler, worker_urls, lb, monitor, metrics)
        
    elif STRATEGY_COMPARISON_TEST:
        print("\n" + "="*60)
        print(" 🔥 STARTING INTERNAL STRATEGY COMPARISON 🔥 ")
        print("="*60)

        results = []
        # Test 1: Round Robin
        print("\n[Test 1/2] Running with Round Robin...")
        lb.strategy = "round_robin"
        rr_metrics = MetricsCollector()
        rr_metrics.register_workers(worker_urls)
        await run_load_test(scheduler, rr_metrics, num_users=500)
        results.append(("Round Robin", rr_metrics.get_summary()))

        # Test 2: Load-Aware
        print("\n[Test 2/2] Running with Load-Aware Routing...")
        lb.strategy = "load_aware"
        la_metrics = MetricsCollector()
        la_metrics.register_workers(worker_urls)
        await run_load_test(scheduler, la_metrics, num_users=500)
        results.append(("Load-Aware", la_metrics.get_summary()))

        # Print the Final Comparison Table for your report
        print("\n" + "+" + "-"*65 + "+")
        print(f"| {'STRATEGY COMPARISON RESULTS (500 Users)':^63} |")
        print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*11 + "+")
        print(f"| {'Strategy':<18} | {'Avg ms':>8} | {'P95 ms':>8} | {'Req/s':>8} | {'Failed':>9} |")
        print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*11 + "+")
        for name, summary in results:
            print(f"| {name:<18} | {summary['avg_latency_ms']:>8.0f} | {summary['p95_latency_ms']:>8.0f} | {summary['throughput_rps']:>8.1f} | {summary['failed']:>9} |")
        print("+" + "-"*20 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*11 + "+")
        
    elif MULTI_SCALE_TEST:
        await run_scaled_load_tests(scheduler, MetricsCollector, scales=[100, 250, 500, 750, 1000])
        print("\n[Main] Multi-Scale Load Test Complete.")
        
    else:
        # Existing single load test / fault tolerance demo
        async def inject_fault():
            await asyncio.sleep(10)      # wait 10s into the load test
            if len(worker_urls) > 1:
                print("\n[Main] [!] Injecting failure into Worker 8002 for fault-tolerance demo")
                await monitor.inject_failure(worker_urls[1])
            await asyncio.sleep(20)      # let it stay failed for 20s
            if len(worker_urls) > 1:
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post(f"{worker_urls[1]}/recover", timeout=2)
                except:
                    pass
                lb.mark_worker_recovered(worker_urls[1])
                print(f"[Main] [OK] Worker {worker_urls[1]} recovered\n")

        asyncio.create_task(inject_fault())

        await run_load_test(scheduler, metrics, num_users=NUM_USERS)
        
        metrics_summary = metrics.get_summary() 
        await metrics.print_report()

    await monitor.stop()
    await lb.close()

if __name__ == "__main__":
    asyncio.run(main_async())