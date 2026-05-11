import threading
import time
import numpy as np
import httpx
from collections import defaultdict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import Response

class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self.latencies = []
        self.start_time = time.time()
        self.worker_stats = defaultdict(lambda: {"count": 0, "avg_latency_ms": 0.0, "gpu_id": -1, "latencies": []})
        self.strategy_distribution = defaultdict(int)
        self._lock = threading.Lock()
        self.worker_urls = []

    def register_workers(self, worker_urls: list):
        self.worker_urls = worker_urls

    def record(self, response: Response):
        with self._lock:
            self.total_requests += 1
            if response.success:
                self.successful += 1
                self.latencies.append(response.latency)
                w_stats = self.worker_stats[response.worker_id]
                w_stats["count"] += 1
                w_stats["gpu_id"] = response.gpu_id
                w_stats["latencies"].append(response.latency)
            else:
                self.failed += 1
            if response.strategy_used:
                self.strategy_distribution[response.strategy_used] += 1

    def get_summary(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            throughput_rps = self.total_requests / elapsed if elapsed > 0 else 0
            latencies_ms = [l * 1000 for l in self.latencies]
            
            avg_lat = np.mean(latencies_ms) if latencies_ms else 0.0
            p50_lat = np.percentile(latencies_ms, 50) if latencies_ms else 0.0
            p95_lat = np.percentile(latencies_ms, 95) if latencies_ms else 0.0
            p99_lat = np.percentile(latencies_ms, 99) if latencies_ms else 0.0

            per_worker_summary = {}
            for wid, stats in self.worker_stats.items():
                w_lats_ms = [l * 1000 for l in stats["latencies"]]
                w_avg = np.mean(w_lats_ms) if w_lats_ms else 0.0
                per_worker_summary[wid] = {
                    "count": stats["count"],
                    "avg_latency_ms": w_avg,
                    "gpu_id": stats["gpu_id"]
                }

            return {
                "total_requests": self.total_requests,
                "successful": self.successful,
                "failed": self.failed,
                "avg_latency_ms": avg_lat,
                "p50_latency_ms": p50_lat,
                "p95_latency_ms": p95_lat,
                "p99_latency_ms": p99_lat,
                "throughput_rps": throughput_rps,
                "per_worker_stats": per_worker_summary,
                "strategy_distribution": dict(self.strategy_distribution)
            }

    async def print_report(self):
        summary = self.get_summary()
        print("\n" + "=" * 40)
        print("|" + "DISTRIBUTED LLM SYSTEM REPORT".center(38) + "|")
        print("+" + "=" * 38 + "+")
        print(f"| Total Requests  : {summary['total_requests']:<20} |")
        print(f"| Successful      : {summary['successful']:<20} |")
        print(f"| Failed          : {summary['failed']:<20} |")
        print(f"| Avg Latency     : {summary['avg_latency_ms']:.0f}ms".ljust(39) + "|")
        print(f"| P95 Latency     : {summary['p95_latency_ms']:.0f}ms".ljust(39) + "|")
        print(f"| Throughput      : {summary['throughput_rps']:.1f} req/s".ljust(39) + "|")
        print("+" + "=" * 38 + "+")
        
        print("\nWorker Stats:")
        for wid, stats in summary['per_worker_stats'].items():
            print(f"  Worker {wid} (cuda:{stats['gpu_id']}) - "
                  f"{stats['count']} requests, Avg: {stats['avg_latency_ms']:.0f}ms")

        print("\n+-----------------------------------------+")
        print("| GPU UTILIZATION                         |")
        print("+-----------------------------------------+")
        if not self.worker_urls:
            print("| No workers registered.                  |")
        
        async with httpx.AsyncClient() as client:
            for url in self.worker_urls:
                try:
                    resp = await client.get(f"{url}/status", timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        print(f"| Worker at {url} (ID: {data.get('id')}, GPU: {data.get('gpu_id')})")
                        print(f"|   Status           : {data.get('status')}")
                        print(f"|   Load Score       : {data.get('load_score')}")
                        print(f"|   Active Requests  : {data.get('active_requests')}")
                        print(f"|   Total Processed  : {data.get('total_processed')}")
                        print(f"|   Total Failed     : {data.get('total_failed')}")
                        print("|")
                    else:
                        print(f"| Worker at {url}  Error fetching stats: {resp.status_code}")
                except Exception as e:
                    print(f"| Worker at {url}  Error fetching stats: {e}")
        print("+-----------------------------------------+")
