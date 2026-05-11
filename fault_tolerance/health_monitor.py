import asyncio
import httpx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HEARTBEAT_INTERVAL, FAILURE_THRESHOLD

class HealthMonitor:
    """
    Background daemon task. Actively pings worker endpoints via HTTP GET /status
    every HEARTBEAT_INTERVAL seconds using asyncio and httpx.
    """
    def __init__(self, worker_urls, load_balancer, scheduler):
        self.worker_urls   = worker_urls
        self.lb            = load_balancer
        self.scheduler     = scheduler
        self._missed_beats = {url: 0 for url in worker_urls}
        self.task          = None
        self._running      = False
        
        # Use a connection pool for pings
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=2.0)

    def start(self):
        self._running = True
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self._running = False
        if self.task:
            self.task.cancel()
        await self.client.aclose()

    async def _ping_worker(self, url: str) -> bool:
        """
        Active health check via HTTP GET /status
        """
        try:
            resp = await self.client.get(f"{url}/status")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "active"
            return False
        except httpx.RequestError:
            return False

    async def run(self):
        try:
            while self._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                for url in self.worker_urls:
                    alive = await self._ping_worker(url)

                    if not alive:
                        self._missed_beats[url] += 1
                        if self._missed_beats[url] >= FAILURE_THRESHOLD:
                            if self.lb.worker_status.get(url) != "failed":
                                print(f"[HealthMonitor] Worker {url} declared DEAD")
                                self.lb.mark_worker_failed(url)
                                await self.scheduler.reassign_tasks_for_worker(url)
                        else:
                            if self.lb.worker_status.get(url) != "failed":
                                print(f"[HealthMonitor] Worker {url} missed heartbeat "
                                      f"({self._missed_beats[url]}/{FAILURE_THRESHOLD})")
                    else:
                        if self._missed_beats[url] > 0:
                            if self.lb.worker_status.get(url) == "failed":
                                print(f"[HealthMonitor] Worker {url} recovered")
                                self.lb.mark_worker_recovered(url)
                        self._missed_beats[url] = 0
        except asyncio.CancelledError:
            pass

    async def inject_failure(self, worker_url: str):
        try:
            await self.client.post(f"{worker_url}/simulate_failure")
        except:
            pass
        self.lb.mark_worker_failed(worker_url)
        await self.scheduler.reassign_tasks_for_worker(worker_url)
