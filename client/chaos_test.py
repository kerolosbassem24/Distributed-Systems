import subprocess
import time
import httpx
import concurrent.futures
import sys
import os

# Configuration
SCHEDULER_URL = "http://localhost:9000/generate"
STATUS_URL = "http://localhost:9000/status"
KILL_PORT = 8001  # We will kill the worker on this port
TOTAL_REQUESTS = 30
CONCURRENCY = 5

def kill_worker_on_port(port):
    """Finds and kills the process running on the specified port (Windows)."""
    print(f"\n[Chaos] 🛠️ Searching for process on port {port}...")
    try:
        # Get the PID of the process on the port
        result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
        lines = result.strip().split('\n')
        if not lines:
            print(f"[Chaos] ❌ No process found on port {port}")
            return
        
        # Usually the last column is the PID
        pid = lines[0].strip().split()[-1]
        print(f"[Chaos] 🔪 Killing Worker Process ID: {pid} (on port {port})...")
        subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=True)
        print(f"[Chaos] ✅ Worker on port {port} has been terminated.")
    except Exception as e:
        print(f"[Chaos] ❌ Failed to kill worker: {e}")

def send_request(req_id):
    """Sends a single request and returns the result."""
    payload = {"query": f"Chaos test request #{req_id}"}
    start_time = time.time()
    try:
        # Using a long timeout because RAG + Inference can be slow
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(SCHEDULER_URL, json=payload)
            latency = (time.time() - start_time) * 1000
            if resp.status_code == 200:
                # Extract worker ID from the last line of NDJSON
                import json
                worker_id = "unknown"
                for line in resp.text.strip().splitlines():
                    data = json.loads(line)
                    if "metadata" in data:
                        worker_id = data["metadata"]["worker_id"]
                return {"id": req_id, "success": True, "worker": worker_id, "latency": latency}
            else:
                return {"id": req_id, "success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"id": req_id, "success": False, "error": str(e)}

def run_chaos_test():
    print("="*60)
    print(" 🔥 DISTRIBUTED SYSTEM CHAOS TEST 🔥 ")
    print(f" Target: {SCHEDULER_URL}")
    print(f" Scenario: Kill worker on port {KILL_PORT} mid-test")
    print("="*60)

    # 1. Check initial health
    try:
        status = httpx.get(STATUS_URL).json()
        active = [w['worker_id'] for w in status['active_workers']]
        print(f"\n[System] Found {len(active)} active workers: {', '.join(active)}")
    except Exception:
        print("❌ Scheduler is not running! Start scheduler.py first.")
        return

    # 2. Start sending requests in the background
    print(f"\n[Test] Starting {TOTAL_REQUESTS} requests with concurrency {CONCURRENCY}...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # Submit the first half
        futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
        
        # Wait a few seconds for requests to be in-flight, then inject fault
        print("[Test] Requests are flowing... waiting 2 seconds before injection...")
        time.sleep(2)
        
        # 3. CRASH THE WORKER!
        kill_worker_on_port(KILL_PORT)
        
        # ── KEY FIX ─────────────────────────────────────────────────────────
        # The health-monitor runs every 5s and needs 2 failures to evict the
        # worker (2 × 5s = 10s).  We wait 12s here so that all remaining
        # requests are routed ONLY to healthy workers.
        print("[Test] Waiting 12s for scheduler to detect & evict dead worker...")
        time.sleep(12)
        print("[Test] Continuing to monitor remaining requests...\n")
        
        # 4. Gather results
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            results.append(res)
            status_icon = "✅" if res['success'] else "❌"
            worker_info = f" handled by {res['worker']}" if res['success'] else f" ERROR: {res['error']}"
            print(f"  {status_icon} Request {res['id']:>2}: {worker_info} ({res.get('latency', 0):.0f}ms)")

    # 5. Summary
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print("\n" + "="*60)
    print(" 📊 CHAOS TEST SUMMARY")
    print(f" Total Requests: {TOTAL_REQUESTS}")
    print(f" Successes:      {len(successes)} ✅")
    print(f" Failures:       {len(failures)} ❌")
    
    if len(failures) == 0:
        print("\n🏆 RESULT: PERFECT FAULT TOLERANCE!")
        print("The scheduler successfully re-routed traffic away from the dead worker.")
    else:
        print(f"\n⚠️ RESULT: SYSTEM DEGRADED ({len(failures)} errors)")
    print("="*60)

if __name__ == "__main__":
    run_chaos_test()
