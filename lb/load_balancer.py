import threading
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.gpu_worker import GPUWorker
from common.models import Request, Response

class LoadBalancer:
    def __init__(self, workers, strategy="round_robin"):
        self.workers  = workers
        self.strategy = strategy
        self._index   = 0
        self._lock    = threading.Lock()

    def get_active_workers(self):
        return [w for w in self.workers if w.status == "active"]

    def get_next_worker(self) -> GPUWorker:
        with self._lock:
            active = self.get_active_workers()
            if not active:
                raise RuntimeError("All GPU workers are offline.")

            if self.strategy == "round_robin":
                worker = active[self._index % len(active)]
                self._index = (self._index + 1) % len(active)

            elif self.strategy == "least_connections":
                worker = min(active, key=lambda w: w.active_requests)

            elif self.strategy == "load_aware":
                worker = min(active, key=lambda w: w.get_load_score())
            else:
                worker = active[self._index % len(active)]
                self._index = (self._index + 1) % len(active)

            return worker

    def dispatch(self, request: Request) -> Response:
        worker   = self.get_next_worker()
        response = worker.process(request)
        response.strategy_used = self.strategy
        return response

    def mark_worker_failed(self, worker_id: int):
        for w in self.workers:
            if w.id == worker_id:
                w.status = "failed"

    def mark_worker_recovered(self, worker_id: int):
        for w in self.workers:
            if w.id == worker_id:
                w.status = "active"

    def switch_strategy(self, strategy_name: str):
        assert strategy_name in ("round_robin","least_connections","load_aware")
        self.strategy = strategy_name
        print(f"[LoadBalancer] Strategy switched to '{strategy_name}'")
