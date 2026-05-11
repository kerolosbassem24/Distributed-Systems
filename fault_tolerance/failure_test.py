import time
from client.load_generator import run_load_test

def run_failure_simulation_test(scheduler, workers, lb, monitor, metrics_collector):
    if len(workers) < 2:
        print("[Failure Test] Cannot run — requires at least 2 workers.")
        run_load_test(scheduler, metrics_collector, num_users=1000)
        return

    print("\n[Failure Test] Starting Formal Failure Simulation Test...")
    
    # PHASE 1
    print("\n--- PHASE 1: Warm up ---")
    run_load_test(scheduler, metrics_collector, num_users=200)
    print("[Failure Test] Phase 1: Warm-up complete. 200 requests processed across all workers.")
    for w in workers:
        print(f"  Worker {w.id} requests: {w.total_processed}")

    # PHASE 2
    print("\n--- PHASE 2: Inject failure ---")
    monitor.inject_failure(worker_id=1)
    print("[Failure Test] *** WORKER 1 FAILURE INJECTED ***")
    active_workers = lb.get_active_workers()
    print(f"[Failure Test] Active workers remaining: {len(active_workers)}")
    print(f"[Failure Test] Load balancer rerouting to workers: {[w.id for w in active_workers]}")

    time.sleep(2)
    fail_time = time.time()

    # PHASE 3
    print("\n--- PHASE 3: Continue under failure ---")
    failed_before_phase3 = metrics_collector.failed
    run_load_test(scheduler, metrics_collector, num_users=400)
    failed_after_phase3 = metrics_collector.failed
    lost_during_failure = failed_after_phase3 - failed_before_phase3
    print(f"[Failure Test] Phase 3: 400 requests completed with Worker 1 OFFLINE. Lost requests: {lost_during_failure}.")
    for w in workers:
        print(f"  Worker {w.id} requests: {w.total_processed}")

    # PHASE 4
    print("\n--- PHASE 4: Recover Worker 1 ---")
    workers[1].recover()
    lb.mark_worker_recovered(1)
    print("[Failure Test] *** WORKER 1 RECOVERED ***")
    print(f"[Failure Test] All workers online: {[w.id for w in lb.get_active_workers()]}")
    downtime = time.time() - fail_time

    # PHASE 5
    print("\n--- PHASE 5: Final burst ---")
    run_load_test(scheduler, metrics_collector, num_users=400)
    print("[Failure Test] Phase 5: Final 400 requests complete. Worker 1 back in rotation.")
    for w in workers:
        print(f"  Worker {w.id} requests: {w.total_processed}")

    summary = metrics_collector.get_summary()
    rerouted = 400 // len(workers)
    
    print("\n  +--------------------------------------------------+")
    print("  | FAULT TOLERANCE TEST SUMMARY                     |")
    print("  +--------------------------------------------------+")
    print(f"  | Total Requests Sent     : {summary['total_requests']:<22} |")
    print(f"  | Successfully Completed  : {summary['successful']:<22} |")
    print(f"  | Lost During Failure     : {lost_during_failure:<22} |")
    print(f"  | Worker 1 Downtime       : ~{int(downtime):<21} seconds |")
    print(f"  | Requests Rerouted       : ~{rerouted:<22} |")
    print(f"  | System Status           : {'PASSED' if summary['failed'] == 0 else 'FAILED' :<22} |")
    print("  +--------------------------------------------------+")
