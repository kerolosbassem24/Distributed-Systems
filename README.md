# Distributed LLM System

A complete, production-quality Python project demonstrating efficient load balancing and GPU cluster task distribution for handling 1000+ concurrent LLM requests.

## Features
- Real GPU computing via CUDA
- Real LLM model instances pinned to each physical GPU
- RAG using ChromaDB and `sentence-transformers` embeddings
- Three load balancing strategies (round_robin, least_connections, load_aware)
- Node failure detection and fault tolerance recovery

## Requirements
You must have a system with real CUDA-enabled GPUs.

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
# source venv/bin/activate

# Install CUDA PyTorch (Change cu121 depending on your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the dependencies
pip install -r requirements.txt
```

## Running the system

```bash
python main.py
```

It will automatically:
1. Ingest dummy documents into ChromaDB.
2. Spin up one `GPUWorker` for each detected physical GPU.
3. Start the `HealthMonitor` for fault tolerance.
4. Launch the load test with 1000 concurrent simulated users.
5. Print the final metrics report.

## Third-Party Integration Guide

If you wish to integrate this Distributed LLM System with third-party applications (such as a web server, Slack bot, or an external API gateway), you can do so by directly interfacing with the `Scheduler` class.

### Integration Steps

1. **Initialize the Cluster (Once at startup)**
   You will need to initialize the workers, load balancer, and scheduler.
   ```python
   import torch
   from config import LOAD_BALANCE_STRATEGY
   from workers.gpu_worker import GPUWorker
   from lb.load_balancer import LoadBalancer
   from master.scheduler import Scheduler
   from fault_tolerance.health_monitor import HealthMonitor
   from rag.ingestor import is_db_populated, ingest_documents

   # Ensure DB is populated
   if not is_db_populated():
       ingest_documents("data/documents/")

   num_gpus = torch.cuda.device_count()
   workers = [GPUWorker(gpu_id=i) for i in range(num_gpus)]
   
   lb = LoadBalancer(workers, strategy=LOAD_BALANCE_STRATEGY)
   scheduler = Scheduler(lb)
   
   monitor = HealthMonitor(workers, lb, scheduler)
   monitor.start()
   ```

2. **Submit Requests**
   Your application can construct a `Request` object and submit it via the `Scheduler`.
   ```python
   from common.models import Request
   import uuid

   # When your app receives a user prompt:
   user_prompt = "Explain how RAG improves LLM response accuracy."
   
   # Create a unique ID for the request
   req_id = int(uuid.uuid4().int % (10**8))
   request = Request(id=req_id, query=user_prompt)

   # Dispatch the request to the GPU cluster
   response = scheduler.handle_request(request)

   if response.success:
       print("LLM Output:", response.result)
   else:
       print("Error:", response.error)
   ```

3. **Metrics Integration**
   You can query the current system load to feed into a third-party dashboard (like Grafana or Datadog) using the scheduler's stats:
   ```python
   stats = scheduler.get_stats()
   print(f"Active Workers: {stats['active_workers']}")
   print(f"Pending Tasks: {stats['pending_tasks']}")
   ```
