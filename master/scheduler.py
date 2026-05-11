import threading
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.models import Request, Response
from config import MAX_RETRIES

class Scheduler:
    def __init__(self, load_balancer):
        self.lb            = load_balancer
        self.pending_tasks = {}         # {request_id: Request}
        self._lock         = threading.Lock()

    def handle_request(self, request: Request) -> Response:
        with self._lock:
            self.pending_tasks[request.id] = request

        for attempt in range(MAX_RETRIES):
            try:
                response = self.lb.dispatch(request)
                with self._lock:
                    self.pending_tasks.pop(request.id, None)
                return response
            except RuntimeError as e:
                if attempt == MAX_RETRIES - 1:
                    with self._lock:
                        self.pending_tasks.pop(request.id, None)
                    return Response(id=request.id, result="", latency=0,
                                    worker_id=-1, gpu_id=-1, strategy_used="",
                                    success=False, error=str(e))
                time.sleep(0.5)

        with self._lock:
            self.pending_tasks.pop(request.id, None)
        return Response(id=request.id, result="", latency=0,
                        worker_id=-1, gpu_id=-1, strategy_used="",
                        success=False, error="Max retries reached")

    def reassign_tasks_for_worker(self, worker_id: int):
        with self._lock:
            tasks = list(self.pending_tasks.values())
        print(f"[Scheduler] Reassigning {len(tasks)} tasks from failed worker {worker_id}")
        for task in tasks:
            task.retry_count += 1
            self.handle_request(task)

    def get_stats(self) -> dict:
        active = self.lb.get_active_workers()
        return {
            "active_workers":  len(active),
            "pending_tasks":   len(self.pending_tasks),
            "strategy":        self.lb.strategy,
            "worker_gpu_stats": [w.get_gpu_utilization() for w in active],
        }
