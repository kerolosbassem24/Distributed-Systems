import threading
import httpx
import time
import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.models import Request, Response

class LoadBalancer:
    def __init__(self, worker_urls, strategy="round_robin"):
        self.worker_urls = worker_urls
        self.worker_status = {url: "active" for url in worker_urls}
        self.strategy = "round_robin" 
        self._index   = 0
        self._lock    = threading.Lock()
        
        # Connection pooling: Use a shared httpx.AsyncClient
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=1000)
        self.client = httpx.AsyncClient(limits=limits, timeout=30.0)

    def get_active_workers(self):
        return [url for url in self.worker_urls if self.worker_status[url] == "active"]

    def get_next_worker_url(self) -> str:
        with self._lock:
            active = self.get_active_workers()
            if not active:
                raise RuntimeError("All GPU workers are offline.")

            url = active[self._index % len(active)]
            self._index = (self._index + 1) % len(active)

            return url

    async def dispatch(self, request: Request) -> Response:
        url = self.get_next_worker_url()
        
        # Convert request to dict for JSON serialization
        req_data = {
            "id": request.id,
            "query": request.query,
            "timestamp": request.timestamp,
            "retry_count": request.retry_count
        }

        try:
            # Asynchronous HTTP POST request
            resp = await self.client.post(f"{url}/process", json=req_data)
            resp.raise_for_status()
            
            resp_data = resp.json()
            
            return Response(
                id=resp_data["id"],
                result=resp_data["result"],
                latency=resp_data["latency"],
                worker_id=resp_data["worker_id"],
                gpu_id=resp_data["gpu_id"],
                strategy_used=self.strategy,
                success=resp_data["success"],
                error=resp_data["error"]
            )
            
        except Exception as e:
            # In case of network failure or timeout
            return Response(
                id=request.id,
                result="",
                latency=0.0,
                worker_id=-1,
                gpu_id=-1,
                strategy_used=self.strategy,
                success=False,
                error=f"Network error: {str(e)}"
            )

    def mark_worker_failed(self, worker_url: str):
        if worker_url in self.worker_status:
            self.worker_status[worker_url] = "failed"
            print(f"[LoadBalancer] Worker {worker_url} marked as FAILED")

    def mark_worker_recovered(self, worker_url: str):
        if worker_url in self.worker_status:
            self.worker_status[worker_url] = "active"
            print(f"[LoadBalancer] Worker {worker_url} marked as ACTIVE")

    def switch_strategy(self, strategy_name: str):
        print(f"[LoadBalancer] Ignoring switch to '{strategy_name}': Forced 'round_robin' in network mode")
        self.strategy = "round_robin"
        
    async def close(self):
        await self.client.aclose()
