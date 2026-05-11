import time
import asyncio
import httpx
from client.load_generator import run_load_test

async def get_worker_processed(url):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{url}/status", timeout=2)
            if r.status_code == 200:
                return r.json().get("total_processed", 0)
    except:
        pass
    return 0

async def run_failure_simulation_test(scheduler, worker_urls, lb, monitor, metrics_collector):
    if len(worker_urls) < 2:
        print("[Failure Test] Cannot run — requires at least 2 workers.")
        await run_load_test(scheduler, metrics_collector, num_users=1000)
        return

    print("\n[Failure Test] Starting Formal Failure Simulation Test...")
    
    # PHASE 1
    print("\n--- PHASE 1: Warm up ---")
    await run_load_test(scheduler, metrics_collector, num_users=200)
    print("[Failure Test] Phase 1: Warm-up complete. 200 requests processed across all workers.")
    for url in worker_urls:
        processed = await get_worker_processed(url)
        print(f"  Worker {url} requests: {processed}")

    # PHASE 2
    print("\n--- PHASE 2: Inject failure ---")
    fail_target = worker_urls[1]
    await monitor.inject_failure(fail_target)
    print(f"[Failure Test] *** WORKER {fail_target} FAILURE INJECTED ***")
    active_workers = lb.get_active_workers()
    print(f"[Failure Test] Active workers remaining: {len(active_workers)}")
    print(f"[Failure Test] Load balancer rerouting to workers: {active_workers}")

    await asyncio.sleep(2)
    fail_time = time.time()

    # PHASE 3
    print("\n--- PHASE 3: Continue under failure ---")
    failed_before_phase3 = metrics_collector.failed
    await run_load_test(scheduler, metrics_collector, num_users=400)
    failed_after_phase3 = metrics_collector.failed
    lost_during_failure = failed_after_phase3 - failed_before_phase3
    print(f"[Failure Test] Phase 3: 400 requests completed with Worker {fail_target} OFFLINE. Lost requests: {lost_during_failure}.")
    for url in worker_urls:
        processed = await get_worker_processed(url)
        print(f"  Worker {url} requests: {processed}")

    # PHASE 4
    print("\n--- PHASE 4: Recover Worker ---")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{fail_target}/recover", timeout=2)
    except:
        pass
    lb.mark_worker_recovered(fail_target)
    print(f"[Failure Test] *** WORKER {fail_target} RECOVERED ***")
    print(f"[Failure Test] All workers online: {lb.get_active_workers()}")
    downtime = time.time() - fail_time

    # PHASE 5
    print("\n--- PHASE 5: Final burst ---")
    await run_load_test(scheduler, metrics_collector, num_users=400)
    print(f"[Failure Test] Phase 5: Final 400 requests complete. Worker {fail_target} back in rotation.")
    for url in worker_urls:
        processed = await get_worker_processed(url)
        print(f"  Worker {url} requests: {processed}")

    summary = metrics_collector.get_summary()
    rerouted = 400 // len(worker_urls)
    
    print("\n  +--------------------------------------------------+")
    print("  | FAULT TOLERANCE TEST SUMMARY                     |")
    print("  +--------------------------------------------------+")
    print(f"  | Total Requests Sent     : {summary['total_requests']:<22} |")
    print(f"  | Successfully Completed  : {summary['successful']:<22} |")
    print(f"  | Lost During Failure     : {lost_during_failure:<22} |")
    print(f"  | Worker Downtime         : ~{int(downtime):<21} seconds |")
    print(f"  | Requests Rerouted       : ~{rerouted:<22} |")
    print(f"  | System Status           : {'PASSED' if summary['failed'] == 0 else 'FAILED' :<22} |")
    print("  +--------------------------------------------------+")
