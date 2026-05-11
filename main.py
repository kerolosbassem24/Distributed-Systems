import torch
import threading
import time
import sys

from config import NUM_USERS, LOAD_BALANCE_STRATEGY, MULTI_SCALE_TEST, FAILURE_TEST, STRATEGY_COMPARISON_TEST
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from fault_tolerance.health_monitor import HealthMonitor
from monitoring.metrics import MetricsCollector
from client.load_generator import run_load_test, run_scaled_load_tests
from rag.ingestor import ingest_documents, is_db_populated
from fault_tolerance.failure_test import run_failure_simulation_test
from client.strategy_comparison import run_strategy_comparison

def main():
    # ── 0. Hardware check ──────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    mock_mode = False
    if num_gpus == 0:
        print("[Main] No CUDA GPUs found. Falling back to Simulated CPU Mode.")
        mock_mode = True
        num_workers = 2 # Simulate 2 workers for local testing
    else:
        print(f"[Main] Detected {num_gpus} GPU(s):")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"       cuda:{i} → {props.name} ({props.total_memory/1e9:.1f} GB)")
        num_workers = num_gpus

    # ── 1. Document ingestion ──────────────────────────────────────
    if not is_db_populated():
        print("\n[Main] Ingesting documents into vector DB...")
        ingest_documents("data/documents/")
    else:
        print("[Main] Vector DB already populated, skipping ingestion.")

    # ── 2. Instantiate GPU workers ─────────────────────────────────
    print("\n[Main] Loading workers...")
    workers = [GPUWorker(gpu_id=i, mock_mode=mock_mode) for i in range(num_workers)]

    # ── 3. Load balancer ───────────────────────────────────────────
    lb = LoadBalancer(workers, strategy=LOAD_BALANCE_STRATEGY)

    # ── 4. Scheduler ───────────────────────────────────────────────
    scheduler = Scheduler(lb)

    # ── 5. Metrics collector ───────────────────────────────────────
    metrics = MetricsCollector()
    metrics.register_workers(workers)

    # ── 6. Health monitor (background daemon) ─────────────────────
    monitor = HealthMonitor(workers, lb, scheduler)
    monitor.start()

    # ── 7. Tests Dispatcher ───────────────────────────────────────
    if FAILURE_TEST:
        run_failure_simulation_test(scheduler, workers, lb, monitor, metrics)
    elif STRATEGY_COMPARISON_TEST:
        run_strategy_comparison(scheduler, lb, workers, MetricsCollector, num_users=500)
    elif MULTI_SCALE_TEST:
        run_scaled_load_tests(scheduler, MetricsCollector, scales=[100, 250, 500, 750, 1000])
        print("\n[Main] Multi-Scale Load Test Complete.")
    else:
        # Existing single load test
        def inject_fault():
            time.sleep(10)      # wait 10s into the load test
            if len(workers) > 1:
                print("\n[Main] [!] Injecting failure into Worker 1 for fault-tolerance demo")
                monitor.inject_failure(worker_id=1)
            time.sleep(20)      # let it stay failed for 20s
            if len(workers) > 1:
                workers[1].recover()
                lb.mark_worker_recovered(1)
                print("[Main] [OK] Worker 1 recovered\n")

        fault_thread = threading.Thread(target=inject_fault, daemon=True)
        fault_thread.start()

        run_load_test(scheduler, metrics, num_users=NUM_USERS)
        metrics.print_report()

if __name__ == "__main__":
    main()
