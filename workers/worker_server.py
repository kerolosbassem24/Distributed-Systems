import argparse
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent dir to sys.path to easily import from config, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.gpu_worker import GPUWorker
from common.models import Request as CommonRequest, Response as CommonResponse

app = FastAPI(title="GPU Worker Node")
worker: GPUWorker = None

# Pydantic models for request/response
class APIRequest(BaseModel):
    id: int
    query: str
    timestamp: float = 0.0
    retry_count: int = 0

class APIResponse(BaseModel):
    id: int
    result: str
    latency: float
    worker_id: int
    gpu_id: int
    strategy_used: str = ""
    success: bool = True
    error: str = ""

@app.post("/process", response_model=APIResponse)
async def process_request(req: APIRequest):
    if not worker or worker.status != "active":
        raise HTTPException(status_code=503, detail="Worker is not active")
    
    # Convert APIRequest to common Request
    common_req = CommonRequest(
        id=req.id,
        query=req.query,
        timestamp=req.timestamp,
        retry_count=req.retry_count
    )
    
    # Process asynchronously to avoid blocking the FastAPI event loop
    common_resp = await asyncio.to_thread(worker.process, common_req)
    
    # Convert common Response to APIResponse
    return APIResponse(
        id=common_resp.id,
        result=common_resp.result,
        latency=common_resp.latency,
        worker_id=common_resp.worker_id,
        gpu_id=common_resp.gpu_id,
        strategy_used=common_resp.strategy_used,
        success=common_resp.success,
        error=common_resp.error
    )

@app.get("/status")
async def get_status():
    if not worker:
        return {"status": "uninitialized"}
    
    return {
        "id": worker.id,
        "gpu_id": worker.gpu_id,
        "status": worker.status,
        "active_requests": worker.active_requests,
        "total_processed": worker.total_processed,
        "total_failed": worker.total_failed,
        "load_score": worker.get_load_score()
    }

@app.post("/simulate_failure")
async def simulate_failure():
    if worker:
        worker.simulate_failure()
    return {"message": "Failure simulated"}

@app.post("/recover")
async def recover():
    if worker:
        worker.recover()
    return {"message": "Recovered"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Worker Node")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--mock", action="store_true", help="Run in mock CPU mode")
    
    args = parser.parse_args()
    
    # Initialize the worker globally, using the port as the unique worker ID
    worker = GPUWorker(gpu_id=args.gpu_id, mock_mode=args.mock, worker_id=args.port)
    
    print(f"Starting worker on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
