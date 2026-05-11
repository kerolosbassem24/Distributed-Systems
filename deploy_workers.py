"""
deploy_workers.py - Deploys worker.py to all Thunder Compute GPU instances
Run from your local Windows machine:  python deploy_workers.py
"""
import subprocess
import sys
import time

# ── SSH Host aliases from your ~/.ssh/config ────────────────────────────────
INSTANCES = [
    {"name": "tnr-0", "worker_id": "thunder-gpu-1"},
    {"name": "tnr-1", "worker_id": "thunder-gpu-2"},
    {"name": "tnr-2", "worker_id": "thunder-gpu-3"},
    {"name": "tnr-3", "worker_id": "thunder-gpu-4"},
]

WORKER_PY = r"worker_remote.py"   # local file to upload

REMOTE_SETUP = """
pip install -q fastapi uvicorn torch transformers accelerate 2>&1 | tail -5
pkill -f 'uvicorn worker' 2>/dev/null || true
nohup uvicorn worker:app --host 0.0.0.0 --port 8000 > worker.log 2>&1 &
sleep 3
if curl -s http://localhost:8000/health | grep -q ok; then
    echo "WORKER OK"
else
    echo "WORKER FAILED - check worker.log"
fi
"""

def run(cmd, check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    return result

def deploy(instance):
    name = instance["name"]
    print(f"\n{'='*50}")
    print(f"  Deploying to {name} ({instance['worker_id']})")
    print(f"{'='*50}")

    # 1. Copy worker.py to remote
    print("  --> Copying worker.py...")
    r = run(f'scp "{WORKER_PY}" {name}:~/worker.py')
    if r.returncode != 0:
        print(f"  [FAILED] SCP failed for {name}. Skipping.")
        return False

    # 2. Run setup + start worker in background (nohup so it survives terminal close)
    print("  --> Installing deps & starting worker (this takes ~30s)...")
    r = run(f'ssh {name} "{REMOTE_SETUP}"')
    if "WORKER OK" in r.stdout:
        print(f"  [OK] {name} is ONLINE and healthy!")
        return True
    else:
        print(f"  [FAILED] {name} health check failed. Output:\n{r.stdout}\n{r.stderr}")
        return False

if __name__ == "__main__":
    print("\n[*] Deploying GPU Workers to Thunder Compute cluster...")
    print(f"   Instances: {[i['name'] for i in INSTANCES]}\n")

    results = []
    for inst in INSTANCES:
        ok = deploy(inst)
        results.append((inst["name"], ok))

    print(f"\n{'='*50}")
    print("  DEPLOYMENT SUMMARY")
    print(f"{'='*50}")
    for name, ok in results:
        status = "[OK]" if ok else "[FAILED]"
        print(f"  {name}: {status}")

    online = sum(1 for _, ok in results if ok)
    print(f"\n  {online}/{len(INSTANCES)} workers deployed successfully.")
    if online > 0:
        print("  [OK] Now open VS Code -> Ports tab and forward each instance's")
        print("    port 8000 to localhost:8000, 8001, 8002, 8003")
