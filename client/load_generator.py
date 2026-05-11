import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def simulate_user(args):
    scheduler, user_id, metrics_collector = args
    request  = Request(id=user_id,
                       query=SAMPLE_QUERIES[user_id % len(SAMPLE_QUERIES)])
    response = scheduler.handle_request(request)
    metrics_collector.record(response)
    return response

def run_load_test(scheduler, metrics_collector, num_users=1000):
    print(f"\n[LoadTest] Launching {num_users} concurrent users...")
    args = [(scheduler, i, metrics_collector) for i in range(num_users)]
    completed = 0
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        futures = {executor.submit(simulate_user, a): a for a in args}
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{num_users} requests completed")

def run_scaled_load_tests(scheduler, metrics_class_ref, scales: list):
    print(f"\n[LoadTest] Starting Multi-Scale Load Test: {scales}")
    results = []
    for scale in scales:
        metrics = metrics_class_ref()
        run_load_test(scheduler, metrics, num_users=scale)
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
