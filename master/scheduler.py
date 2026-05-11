import threading
import httpx
import time
import asyncio
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

    async def handle_request(self, request: Request) -> Response:
        with self._lock:
            self.pending_tasks[request.id] = request

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.lb.dispatch(request)
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
                await asyncio.sleep(0.5)

        with self._lock:
            self.pending_tasks.pop(request.id, None)
        return Response(id=request.id, result="", latency=0,
                        worker_id=-1, gpu_id=-1, strategy_used="",
                        success=False, error="Max retries reached")

    async def reassign_tasks_for_worker(self, worker_url: str):
        with self._lock:
            tasks = list(self.pending_tasks.values())
        print(f"[Scheduler] Reassigning {len(tasks)} tasks from failed worker {worker_url}")
        for task in tasks:
            task.retry_count += 1
            # In a true distributed system we'd enqueue these again
            # For this test, we fire and forget the re-handle as a background task
            asyncio.create_task(self.handle_request(task))

    async def get_stats(self) -> dict:
        active_urls = self.lb.get_active_workers()
        gpu_stats = []
        
        # Concurrently fetch stats from all active workers
        async def fetch_stat(url):
            try:
                # We can reuse the load balancer's client or use a temporary one
                # Since get_stats is infrequent, a temporary httpx.AsyncClient is fine,
                # but we'll use lb.client if available.
                resp = await self.lb.client.get(f"{url}/status", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "device_name": f"Worker at {url}",
                        "memory_pct": data.get("load_score", 0.0) * 100,
                        "status": data.get("status", "unknown")
                    }
            except:
                pass
            return None

        if active_urls:
            results = await asyncio.gather(*(fetch_stat(url) for url in active_urls))
            gpu_stats = [r for r in results if r is not None]

        return {
            "active_workers":  len(active_urls),
            "pending_tasks":   len(self.pending_tasks),
            "strategy":        self.lb.strategy,
            "worker_gpu_stats": gpu_stats,
        }
