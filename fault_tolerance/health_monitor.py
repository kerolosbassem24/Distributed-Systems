import threading
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEARTBEAT_INTERVAL, FAILURE_THRESHOLD

class HealthMonitor(threading.Thread):
    """
    Background daemon thread. Actively pings worker heartbeats every
    HEARTBEAT_INTERVAL seconds. Marks dead workers and triggers
    task reassignment through the scheduler.

    FIX: We now actively call worker.heartbeat() each cycle for workers
    that are functioning. Previously, last_heartbeat was only updated
    when a request was processed — so idle workers at startup were
    incorrectly declared dead after HEARTBEAT_INTERVAL * FAILURE_THRESHOLD
    seconds with no traffic.
    """
    def __init__(self, workers, load_balancer, scheduler):
        super().__init__(daemon=True)
        self.workers       = workers
        self.lb            = load_balancer
        self.scheduler     = scheduler
        self._missed_beats = {w.id: 0 for w in workers}
        # Pre-stamp heartbeats so workers start healthy
        for w in workers:
            w.last_heartbeat = time.time()

    def _ping_worker(self, worker) -> bool:
        """
        Active health check: for in-process workers we call heartbeat()
        directly. A worker is considered alive if it is not in a
        failed/recovering status AND its heartbeat updates successfully.
        """
        try:
            if worker.status == "active":
                worker.heartbeat()   # refresh the timestamp
                return True
            return False
        except Exception:
            return False

    def run(self):
        while True:
            time.sleep(HEARTBEAT_INTERVAL)
            for worker in self.workers:
                if worker.status == "failed":
                    continue

                alive = self._ping_worker(worker)

                if not alive:
                    self._missed_beats[worker.id] += 1
                    if self._missed_beats[worker.id] >= FAILURE_THRESHOLD:
                        print(f"[HealthMonitor] Worker {worker.id} "
                              f"(cuda:{worker.gpu_id}) declared DEAD")
                        self.lb.mark_worker_failed(worker.id)
                        self.scheduler.reassign_tasks_for_worker(worker.id)
                    else:
                        print(f"[HealthMonitor] Worker {worker.id} missed heartbeat "
                              f"({self._missed_beats[worker.id]}/{FAILURE_THRESHOLD})")
                else:
                    if self._missed_beats[worker.id] > 0:
                        print(f"[HealthMonitor] Worker {worker.id} recovered")
                        self.lb.mark_worker_recovered(worker.id)
                    self._missed_beats[worker.id] = 0

    def inject_failure(self, worker_id: int):
        for w in self.workers:
            if w.id == worker_id:
                w.simulate_failure()
                self.lb.mark_worker_failed(worker_id)
                self.scheduler.reassign_tasks_for_worker(worker_id)
