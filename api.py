import asyncio
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import time

from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from fault_tolerance.health_monitor import HealthMonitor
from rag.retriever import retrieve_context
from common.models import Request as ModelRequest
import random

app = FastAPI()

# Fix for the CORS Trap mentioned earlier!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
worker_urls = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
    "http://localhost:8004"
]
lb = LoadBalancer(worker_urls, strategy="round_robin")
scheduler = Scheduler(lb)
monitor = HealthMonitor(worker_urls, lb, scheduler)

@app.on_event("startup")
async def startup_event():
    print("[API] Starting Health Monitor...")
    monitor.start()

@app.on_event("shutdown")
async def shutdown_event():
    print("[API] Shutting down...")
    await monitor.stop()
    await lb.close()

@app.get("/status")
async def get_status():
    """Endpoint for the UI's checkStatus() polling"""
    # Assuming your load balancer has a way to check active workers
    # Adjust these method calls based on your actual LoadBalancer implementation
    active = getattr(lb, 'active_workers', worker_urls) 
    
    return {
        "active_workers": [{"worker_id": w, "active_requests": 0, "latency_ms": 12} for w in active],
        "inactive_workers": [],
        "strategy": lb.strategy
    }

@app.post("/generate")
async def generate_response(request: FastAPIRequest):
    """Endpoint for the UI's prompt submission"""
    data = await request.json()
    user_query = data.get("query", "")

    async def event_stream():
        start_time = time.time()
        
        # 1. RAG Retrieval Step
        yield json.dumps({"chunk": "🔍 *Searching knowledge base...*\n\n"}) + "\n"
        await asyncio.sleep(0.5) # Fake slight delay for UI effect
        
        context = retrieve_context(user_query)
        prompt_with_context = f"Context: {context}\n\nQuestion: {user_query}"
        
        # 2. Scheduler Dispatch Step
        req_id = random.randint(1, 1000000)
        req = ModelRequest(id=req_id, query=prompt_with_context)
        
        try:
            response = await scheduler.handle_request(req)
            
            if not response.success:
                yield json.dumps({"chunk": f"❌ Error from cluster: {response.error}"}) + "\n"
                return

            worker_id = response.worker_id
            response_text = response.result
            
        except Exception as e:
            yield json.dumps({"chunk": f"\n\n❌ Scheduler Error: {str(e)}"}) + "\n"
            return
        
        # Stream the text back to the frontend simulating token-by-token
        words = response_text.split(" ")
        for word in words:
            yield json.dumps({"chunk": word + " "}) + "\n"
            await asyncio.sleep(0.05) # Simulate generation speed
            
        # 3. Send final metadata so the frontend renders the worker tags
        latency = (time.time() - start_time) * 1000
        yield json.dumps({
            "metadata": {
                "worker_id": worker_id,
                "latency_ms": latency
            }
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    print("🚀 Starting Master API Gateway on port 9000...")
    uvicorn.run(app, host="0.0.0.0", port=9000)