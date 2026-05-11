from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import threading
import time
import os
import torch

from config import *
from workers.gpu_worker import GPUWorker
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from fault_tolerance.health_monitor import HealthMonitor
from monitoring.metrics import MetricsCollector
from rag.ingestor import ingest_documents, is_db_populated
from common.models import Request as LLMRequest

app_workers = []
lb = None
scheduler = None
metrics = None
monitor = None
req_counter = 0
req_lock = threading.Lock()

class QueryRequest(BaseModel):
    query: str
    request_id: int = None

class QueryResponse(BaseModel):
    request_id: int
    answer: str
    latency_ms: float
    worker_id: int
    gpu_id: int
    strategy_used: str
    success: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_workers, lb, scheduler, metrics, monitor
    
    if not is_db_populated():
        print("[App] Ingesting documents into vector database (this may take a minute)...")
        ingest_documents("data/documents/")
        print("[App] ✅ Document ingestion complete!")
    else:
        print("[App] ✅ Vector database already populated.")
        
    num_gpus = torch.cuda.device_count()
    mock_mode = False
    if num_gpus == 0:
        mock_mode = True
        # Default 4 simulated workers; override with env var MOCK_WORKERS=N
        num_workers = int(os.environ.get("MOCK_WORKERS", "4"))
        print(f"[App] No GPU detected — running {num_workers} simulated CPU workers.")
        print(f"[App] Set env var MOCK_WORKERS=N to change the count.")
    else:
        num_workers = num_gpus
        print(f"[App] Detected {num_gpus} GPU(s) — creating {num_workers} real GPU worker(s).")

    app_workers = [GPUWorker(gpu_id=i, mock_mode=mock_mode) for i in range(num_workers)]
    
    lb = LoadBalancer(app_workers, strategy=LOAD_BALANCE_STRATEGY)
    scheduler = Scheduler(lb)
    metrics = MetricsCollector()
    metrics.register_workers(app_workers)
    
    monitor = HealthMonitor(app_workers, lb, scheduler)
    monitor.start()
    
    print("[App] System ready.")
    yield
    print("[App] Shutting down.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global req_counter
    if req.request_id is None:
        with req_lock:
            req_counter += 1
            req_id = req_counter
    else:
        req_id = req.request_id
        
    llm_req = LLMRequest(id=req_id, query=req.query)
    response = scheduler.handle_request(llm_req)
    metrics.record(response)
    
    if not response.success:
        raise HTTPException(status_code=503, detail=response.error)
        
    return QueryResponse(
        request_id=response.id,
        answer=response.result,
        latency_ms=response.latency * 1000.0,
        worker_id=response.worker_id,
        gpu_id=response.gpu_id,
        strategy_used=response.strategy_used,
        success=response.success
    )

@app.get("/health")
def health():
    active = len([w for w in app_workers if w.status == "active"])
    failed = len(app_workers) - active
    
    if active == len(app_workers):
        status = "healthy"
    elif active > 0:
        status = "degraded"
    else:
        status = "critical"
        
    worker_details = []
    for w in app_workers:
        gpu_mem_pct = 0.0
        if not getattr(w, 'mock_mode', False) and w.status == "active":
            try:
                util = w.get_gpu_utilization()
                gpu_mem_pct = util.get("memory_pct", 0.0)
            except:
                pass
                
        worker_details.append({
            "worker_id": w.id,
            "status": w.status,
            "gpu_id": w.gpu_id,
            "active_requests": w.active_requests,
            "total_processed": w.total_processed,
            "gpu_memory_pct": gpu_mem_pct
        })
        
    return {
        "status": status,
        "total_workers": len(app_workers),
        "active_workers": active,
        "failed_workers": failed,
        "worker_details": worker_details
    }

@app.get("/metrics")
def get_metrics():
    return metrics.get_summary()

@app.get("/workers")
def get_workers():
    res = []
    for w in app_workers:
        data = {
            "worker_id": w.id,
            "status": w.status,
            "gpu_id": w.gpu_id,
            "active_requests": w.active_requests,
            "total_processed": w.total_processed,
            "load_score": w.get_load_score() if w.status == "active" else 1.0,
        }
        if not getattr(w, 'mock_mode', False) and w.status == "active":
            try:
                util = w.get_gpu_utilization()
                data["device_name"] = util.get("device_name", "")
                data["gpu_memory_used_gb"] = util.get("memory_used_gb", 0.0)
                data["gpu_memory_total_gb"] = util.get("memory_total_gb", 0.0)
            except:
                pass
        else:
            data["device_name"] = "Simulated CPU Node" if getattr(w, 'mock_mode', False) else "Offline"
            
        res.append(data)
    return res

@app.post("/workers/{worker_id}/fail")
def fail_worker(worker_id: int):
    active = len([w for w in app_workers if w.status == "active"])
    
    target = None
    for w in app_workers:
        if w.id == worker_id:
            target = w
            break
            
    if not target:
        raise HTTPException(status_code=400, detail="Invalid worker_id")
        
    if target.status == "failed":
        return {"message": f"Worker {worker_id} is already failed", "active_workers": active}
        
    if active <= 1:
        raise HTTPException(status_code=400, detail="Cannot fail the last active worker")
        
    monitor.inject_failure(worker_id)
    return {"message": f"Worker {worker_id} failure injected", "active_workers": active - 1}

@app.post("/workers/{worker_id}/recover")
def recover_worker(worker_id: int):
    target = None
    for w in app_workers:
        if w.id == worker_id:
            target = w
            break
            
    if not target:
        raise HTTPException(status_code=400, detail="Invalid worker_id")
        
    if target.status == "active":
        active = len([w for w in app_workers if w.status == "active"])
        return {"message": f"Worker {worker_id} is already active", "active_workers": active}
        
    target.recover()
    lb.mark_worker_recovered(worker_id)
    active = len([w for w in app_workers if w.status == "active"])
    return {"message": f"Worker {worker_id} recovered", "active_workers": active}

@app.post("/load-balancer/strategy")
def set_strategy(payload: dict):
    strategy = payload.get("strategy")
    if strategy not in ["round_robin", "least_connections", "load_aware"]:
        raise HTTPException(status_code=400, detail="Invalid strategy name")
    lb.switch_strategy(strategy)
    return {"message": f"Strategy switched to {strategy}"}

@app.get("/docs-ingested")
def docs_ingested():
    import chromadb
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_or_create_collection("knowledge_base")
        count = collection.count()
        return {"populated": count > 0, "chunk_count": count, "collection_name": "knowledge_base"}
    except Exception as e:
        return {"populated": False, "chunk_count": 0, "collection_name": "knowledge_base", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
