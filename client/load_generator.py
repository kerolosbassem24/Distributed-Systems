import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import THREAD_POOL_SIZE
from common.models import Request

SAMPLE_QUERIES = [
    "What is load balancing in distributed systems?",
    "Explain how RAG improves LLM response accuracy.",
    "What are the benefits of GPU parallelism for AI inference?",
    "How does fault tolerance work in distributed computing?",
    "What is the difference between horizontal and vertical scaling?",
    "Explain the CAP theorem and its implications.",
    "What is a vector database and how is it used in RAG?",
    "How does CUDA enable parallel computation on GPUs?",
    "What is model sharding in distributed LLM serving?",
    "How do transformer models handle attention at scale?",
    "What is the role of a master node in a GPU cluster?",
    "Explain round robin vs least connections load balancing.",
    "What metrics are used to evaluate distributed system performance?",
    "How do heartbeat mechanisms detect node failure?",
    "What is tensor parallelism in model serving?",
    "How does quantization reduce GPU memory usage for LLMs?",
    "Explain the difference between batch and streaming inference.",
    "What is KV cache and how does it speed up LLM inference?",
    "How does consistent hashing help in distributed caching?",
    "What are the tradeoffs between throughput and latency?",
]

async def simulate_user(scheduler, user_id, metrics_collector):
    request  = Request(id=user_id,
                       query=SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)])
    # We await the scheduler directly since this is an in-memory client
    response = await scheduler.handle_request(request)
    metrics_collector.record(response)
    return response

async def run_load_test(scheduler, metrics_collector, num_users=1000):
    print(f"\n[LoadTest] Launching {num_users} concurrent users...")
    tasks = [simulate_user(scheduler, i, metrics_collector) for i in range(num_users)]
    
    # Run all tasks concurrently using asyncio.gather
    # We use a semaphore to limit concurrency if needed, but the user requested truly concurrent.
    # We will wrap it in a semaphore just to avoid hitting OS socket limits, matching THREAD_POOL_SIZE 
    # but since it's async we can push it higher. Let's use the requested 1000 concurrently.
    
    completed = 0
    
    # We can use asyncio.as_completed to track progress
    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1
        if completed % 100 == 0:
            print(f"  Progress: {completed}/{num_users} requests completed")

async def run_scaled_load_tests(scheduler, metrics_class_ref, scales: list):
    print(f"\n[LoadTest] Starting Multi-Scale Load Test: {scales}")
    results = []
    for scale in scales:
        metrics = metrics_class_ref()
        await run_load_test(scheduler, metrics, num_users=scale)
        summary = metrics.get_summary()
        results.append({
            "users": scale,
            "avg_ms": summary["avg_latency_ms"],
            "p95_ms": summary["p95_latency_ms"],
            "rps": summary["throughput_rps"],
            "failed": summary["failed"]
        })
    
    print("\n  +-------------------------------------------------------+")
    print("  | SCALING TEST RESULTS                                  |")
    print("  +-------+----------+----------+----------+--------------+")
    print("  | Users | Avg (ms) | P95 (ms) | Req/s    | Failed       |")
    print("  +-------+----------+----------+----------+--------------+")
    for r in results:
        print(f"  | {r['users']:>5} | {r['avg_ms']:>8.0f} | {r['p95_ms']:>8.0f} | {r['rps']:>8.1f} | {r['failed']:>12} |")
    print("  +-------+----------+----------+----------+--------------+")
