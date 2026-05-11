import asyncio
import logging
import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Move heavy RAG imports to top-level for performance
try:
    from rag.retriever import retrieve_context
except ImportError:
    def retrieve_context(query): return "RAG module not found."

# ---------------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("scheduler")

# ---------------------------------------------------------------------------
# 2. Load Balancing Strategy (switch here)
# ---------------------------------------------------------------------------
class RoutingStrategy(str, Enum):
    ROUND_ROBIN      = "round_robin"
    LEAST_CONNECTIONS = "least_connections"

ACTIVE_STRATEGY = RoutingStrategy.LEAST_CONNECTIONS   # ← change this to switch

# ---------------------------------------------------------------------------
# 3. Pydantic Models
# ---------------------------------------------------------------------------
class WorkerRegistration(BaseModel):
    worker_id: str
    host: Optional[str] = "localhost"
    port: Optional[int] = 8000
    base_url: Optional[str] = None
    gpu_name: Optional[str] = "Unknown GPU"

class GenerateRequest(BaseModel):
    query: str

class SchedulerResponse(BaseModel):
    response: str
    worker_id: str
    scheduler_timestamp: str
    latency_ms: float

class WorkerState(BaseModel):
    worker_id: str
    base_url: str
    gpu_name: str
    is_active: bool = True
    last_heartbeat: Optional[datetime] = None
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    active_requests: int = 0          # ← NEW: tracks in-flight requests

# ---------------------------------------------------------------------------
# 4. Worker Registry & Load Balancer
# ---------------------------------------------------------------------------
class WorkerRegistry:
    def __init__(self):
        self.workers: Dict[str, WorkerState] = {}
        self._rr_index = 0
        self._lock = asyncio.Lock()

    async def register(self, reg: WorkerRegistration) -> WorkerState:
        async with self._lock:
            base_url = reg.base_url if reg.base_url else f"http://{reg.host}:{reg.port}"
            state = WorkerState(
                worker_id=reg.worker_id,
                base_url=base_url,
                gpu_name=reg.gpu_name or "Unknown",
                last_heartbeat=datetime.now(timezone.utc)
            )
            self.workers[reg.worker_id] = state
            logger.info(f"Registered worker {reg.worker_id} at {base_url} ({state.gpu_name})")
            return state

    # ── Round-Robin ──────────────────────────────────────────────────────────
    async def _get_round_robin(self) -> Optional[WorkerState]:
        active = [w for w in self.workers.values() if w.is_active]
        if not active:
            return None
        self._rr_index = (self._rr_index + 1) % len(active)
        return active[self._rr_index]

    # ── Least-Connections ─────────────────────────────────────────────────────
    async def _get_least_connections(self) -> Optional[WorkerState]:
        active = [w for w in self.workers.values() if w.is_active]
        if not active:
            return None
        # Pick the worker with the fewest active in-flight requests
        chosen = min(active, key=lambda w: w.active_requests)
        logger.info(
            f"[Least-Connections] Routing to {chosen.worker_id} "
            f"(active_requests={chosen.active_requests}) | "
            + ", ".join(f"{w.worker_id}={w.active_requests}" for w in active)
        )
        return chosen

    async def get_next_worker(self) -> Optional[WorkerState]:
        async with self._lock:
            active = [w for w in self.workers.values() if w.is_active]
            if not active:
                # Log current state of all workers to help debug
                states = ", ".join([f"{w.worker_id}(active={w.is_active}, fails={w.consecutive_failures})" for w in self.workers.values()])
                logger.error(f"Routing failed: 0 active workers. Registry state: {states}")
                return None
            
            if ACTIVE_STRATEGY == RoutingStrategy.LEAST_CONNECTIONS:
                # Pick the worker with the fewest active in-flight requests
                chosen = min(active, key=lambda w: w.active_requests)
                return chosen
            
            # Round Robin
            self._rr_index = (self._rr_index + 1) % len(active)
            chosen = active[self._rr_index]
            return chosen

    async def increment_requests(self, worker_id: str):
        async with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id].active_requests += 1

    async def decrement_requests(self, worker_id: str):
        async with self._lock:
            if worker_id in self.workers:
                self.workers[worker_id].active_requests = max(
                    0, self.workers[worker_id].active_requests - 1
                )

    async def update_worker_health(self, worker_id: str, is_active: bool, latency: float = 0.0):
        async with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                was_active = worker.is_active
                
                if is_active:
                    worker.is_active = True
                    worker.last_heartbeat = datetime.now(timezone.utc)
                    worker.consecutive_failures = 0
                    worker.latency_ms = latency
                    if not was_active:
                        logger.info(f"Worker {worker_id} is BACK ONLINE.")
                else:
                    worker.consecutive_failures += 1
                    # Mark inactive after 2 consecutive failures for faster chaos-test recovery
                    if worker.consecutive_failures >= 2:
                        if worker.is_active:
                            worker.is_active = False
                            logger.warning(f"Worker {worker_id} marked as INACTIVE (failed {worker.consecutive_failures} times)")
                    else:
                        if worker.is_active:
                            logger.info(f"Worker {worker_id} missed health check ({worker.consecutive_failures}/2)")

    def get_all_workers(self) -> List[WorkerState]:
        return list(self.workers.values())

# ---------------------------------------------------------------------------
# 5. FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Distributed Scheduler", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = WorkerRegistry()

# ---------------------------------------------------------------------------
# 6. Background Health Monitor
# ---------------------------------------------------------------------------
async def health_monitor_task():
    logger.info("Starting background health monitoring task...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            for worker in registry.get_all_workers():
                health_url = f"{worker.base_url}/health"
                t0 = time.time()
                try:
                    resp = await client.get(health_url)
                    latency = (time.time() - t0) * 1000
                    if resp.status_code == 200:
                        await registry.update_worker_health(worker.worker_id, True, latency)
                    else:
                        await registry.update_worker_health(worker.worker_id, False)
                except Exception as e:
                    await registry.update_worker_health(worker.worker_id, False)
            await asyncio.sleep(5.0)

# ---------------------------------------------------------------------------
# 7. Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    config_path = "workers_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            static_workers = config.get("static_workers", [])
            for w in static_workers:
                reg = WorkerRegistration(
                    worker_id=w.get("worker_id", "unknown-worker"),
                    base_url=w.get("url"),
                    gpu_name=w.get("gpu_name", "Cloud GPU")
                )
                await registry.register(reg)
            logger.info(f"Loaded {len(static_workers)} static workers from {config_path}.")
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {e}")

    asyncio.create_task(health_monitor_task())
    logger.info(f"Scheduler ready. Strategy: {ACTIVE_STRATEGY.value}")

# ---------------------------------------------------------------------------
# 8. Endpoints
# ---------------------------------------------------------------------------
@app.post("/register")
async def register_worker(reg: WorkerRegistration):
    await registry.register(reg)
    return {"status": "registered", "worker_id": reg.worker_id}

@app.post("/strategy")
async def set_strategy(payload: dict):
    global ACTIVE_STRATEGY
    strategy = payload.get("strategy")
    if strategy == "round_robin":
        ACTIVE_STRATEGY = RoutingStrategy.ROUND_ROBIN
    elif strategy == "least_connections":
        ACTIVE_STRATEGY = RoutingStrategy.LEAST_CONNECTIONS
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy")
    logger.info(f"Strategy switched to: {ACTIVE_STRATEGY.value}")
    return {"status": "ok", "current_strategy": ACTIVE_STRATEGY.value}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    start_time = time.time()

    # ── RAG Retrieval ────────────────────────────────────────────────────────
    try:
        context = await asyncio.to_thread(retrieve_context, request.query)
        if context and context != "No relevant context found.":
            full_prompt = (
                f"<|system|>\nYou are a helpful assistant. Use context below.\n"
                f"CONTEXT:\n{context}</s>\n"
                f"<|user|>\n{request.query}</s>\n<|assistant|>\n"
            )
        else:
            full_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{request.query}</s>\n<|assistant|>\n"
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        full_prompt = f"<|user|>\n{request.query}</s>\n<|assistant|>\n"

    payload = {"query": full_prompt}

    # ── Fault-Tolerant Routing with Transparent Retries ──────────────────────
    max_retries = 5
    for attempt in range(max_retries):
        worker = await registry.get_next_worker()
        if not worker:
            raise HTTPException(status_code=503, detail="No healthy workers available")

        await registry.increment_requests(worker.worker_id)
        process_url = f"{worker.base_url}/process"
        
        client = httpx.AsyncClient(timeout=120.0)
        try:
            logger.info(f"Routing to {worker.worker_id} (Attempt {attempt+1})")
            
            # We use a context manager to ensure the response is handled correctly
            # but we return a StreamingResponse that will take over.
            req = client.build_request("POST", process_url, json=payload)
            resp = await client.send(req, stream=True)
            
            if resp.status_code != 200:
                await resp.aclose()
                raise httpx.HTTPStatusError(f"Status {resp.status_code}", request=req, response=resp)

            # If we reached here, the connection is established and headers are received.
            # We will now wrap the iterator to handle potential mid-stream drops.
            async def robust_stream_wrapper(r, c, w_id, t0):
                try:
                    async for chunk in r.aiter_text():
                        if chunk:
                            yield json.dumps({"chunk": chunk}) + "\n"
                except Exception as stream_err:
                    logger.error(f"Stream interrupted for {w_id}: {stream_err}")
                    yield json.dumps({"error": "Worker connection lost mid-stream", "worker_id": w_id}) + "\n"
                finally:
                    await r.aclose()
                    await c.aclose()
                    await registry.decrement_requests(w_id)
                
                latency_ms = (time.time() - t0) * 1000
                yield json.dumps({"metadata": {
                    "worker_id": w_id,
                    "latency_ms": round(latency_ms, 2),
                    "scheduler_timestamp": datetime.now(timezone.utc).isoformat()
                }}) + "\n"

            return StreamingResponse(
                robust_stream_wrapper(resp, client, worker.worker_id, start_time),
                media_type="application/x-ndjson"
            )

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Worker {worker.worker_id} failed on connect: {e}. Retrying with another worker...")
            await client.aclose()
            await registry.decrement_requests(worker.worker_id)
            await registry.update_worker_health(worker.worker_id, False)
            # Loop continues to next attempt...
            continue
        except Exception as e:
            logger.error(f"Unexpected error with worker {worker.worker_id}: {e}")
            await client.aclose()
            await registry.decrement_requests(worker.worker_id)
            await registry.update_worker_health(worker.worker_id, False)
            continue

    raise HTTPException(status_code=502, detail="All worker retries failed.")

@app.get("/status")
async def get_status():
    workers = registry.get_all_workers()
    return {
        "strategy": ACTIVE_STRATEGY.value,
        "active_workers": [
            {
                "worker_id": w.worker_id,
                "gpu_name": w.gpu_name,
                "base_url": w.base_url,
                "latency_ms": round(w.latency_ms, 2),
                "active_requests": w.active_requests,
            }
            for w in workers if w.is_active
        ],
        "inactive_workers": [w.worker_id for w in workers if not w.is_active],
        "total_workers": len(workers),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scheduler:app", host="0.0.0.0", port=9000, reload=True)
