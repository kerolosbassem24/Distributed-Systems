# Distributed LLM System — Changes Report

## 1. Overview
In this session, we expanded the system's capabilities beyond simple local terminal testing by introducing robust testing suites and a web server. We added formal tests for scaling, failure recovery, and load balancing strategy comparisons. We also upgraded the metric reporting to include live GPU utilization data. Finally, we wrapped the entire backend in a FastAPI application, turning the system into a production-ready HTTP web API. A total of 4 existing files were modified, and 4 new files were created.

## 2. Changes Per File

### monitoring/metrics.py
- **Type**: Modified
- **What changed**: 
  - Added `register_workers` method to store references to GPU workers.
  - Updated `print_report` to include a new "GPU UTILIZATION" ASCII section.
- **Why**: Fulfills the requirement to display actual GPU hardware statistics in the final load test report (Task 1).
- **Modified functions**: 
  - `__init__`: Added `self.workers` array.
  - `print_report`: Appended GPU table generation.
- **New functions/classes added**:
  - `register_workers(self, workers: list)`: Associates worker objects with the metrics collector.

### client/load_generator.py
- **Type**: Modified
- **What changed**:
  - Appended `run_scaled_load_tests` function.
- **Why**: To prove the system can handle escalating load stages (100 -> 1000 users) as required by project docs (Task 2).
- **New functions/classes added**:
  - `run_scaled_load_tests(scheduler, metrics_class_ref, scales: list)`: Iterates over user scales, collects metrics, and prints an ASCII summary table.

### config.py
- **Type**: Modified
- **What changed**: Added 3 new boolean toggle constants.
- **Why**: Allows easy toggling of the new test modes without editing `main.py` directly.
- **New Config Keys Added**: see Section 4.

### main.py
- **Type**: Modified
- **What changed**:
  - Registered workers with `metrics.register_workers()`.
  - Replaced the hardcoded single load test with a dispatcher using the new config flags.
  - When `MULTI_SCALE_TEST=True`, removed the final `metrics.print_report()` call and replaced it with a closing message.
- **Why**: To act as a central hub for all the new testing suites (Tasks 1, 2, 3, 5) and clean up console output.
- **Modified functions**:
  - `main()`: Updated to conditionally call multi-scale, failure, strategy, or standard load tests.

### fault_tolerance/failure_test.py
- **Type**: Created
- **What changed**: Created from scratch to orchestrate a formal fault tolerance test. Calculates "Lost During Failure" dynamically by tracking completed request counts before and after the failure phase.
- **Why**: Provides a repeatable script to demonstrate node failure and automatic recovery for the project demo (Task 3).
- **New functions/classes added**:
  - `run_failure_simulation_test(scheduler, workers, lb, monitor, metrics_collector)`: Runs a 5-phase test (warmup, fail, stress, recover, burst) and prints a summary.

### client/strategy_comparison.py
- **Type**: Created
- **What changed**: Created from scratch to pit the 3 load balancing strategies against each other. Added `workers` as a parameter and calls `register_workers(workers)` on each fresh metrics instance.
- **Why**: Provides data for the project report by running identical loads across strategies and comparing latency and fairness (Task 5).
- **New functions/classes added**:
  - `run_strategy_comparison(scheduler, lb, workers, metrics_class, num_users=500)`: Iterates over strategies, cleans worker state, runs load, and outputs comparison table.

### app.py
- **Type**: Created
- **What changed**: 
  - Created a complete FastAPI application wrapping all system components.
  - Added a thread-safe auto-incrementing counter for `request_id` using `threading.Lock()`.
  - Added clear progress logging during document ingestion in lifespan.
  - Added `mock_mode` guard in the `/workers` endpoint to prevent crashes when `get_gpu_utilization()` is called on CPU workers.
- **Why**: Meets the requirement to allow users to interact with the system via HTTP, transforming it from a script to a service (Task 4).
- **New functions/classes added**:
  - `QueryRequest` / `QueryResponse`: Pydantic models for data validation.
  - `lifespan`: Startup context manager for DB ingestion and cluster spin-up.

### requirements.txt
- **Type**: Modified
- **What changed**: Appended `fastapi`, `uvicorn[standard]`, and `pydantic`.
- **Why**: Required dependencies for `app.py`.

## 3. New Endpoints (app.py)

| Method | Path | Description | Request Body | Response |
|---|---|---|---|---|
| POST | `/query` | Submits a query to the LLM via Load Balancer | `{"query": "...", "request_id": 1}` | JSON with answer, latency, worker info |
| GET | `/health` | Returns cluster status and active worker count | None | JSON system health payload |
| GET | `/metrics` | Returns full metrics aggregator summary | None | JSON metrics payload |
| GET | `/workers` | Detailed list of all workers + GPU hardware info | None | JSON array of workers |
| POST | `/workers/{worker_id}/fail` | Force injects a failure on a specific worker | None | JSON success message |
| POST | `/workers/{worker_id}/recover` | Recovers a previously failed worker | None | JSON success message |
| POST | `/load-balancer/strategy` | Swaps the active LoadBalancer routing method | `{"strategy": "load_aware"}` | JSON success message |
| GET | `/docs-ingested` | Verifies ChromaDB connection and chunk count | None | JSON document metrics |

## 4. New Config Keys

| Key | Default Value | Purpose |
|---|---|---|
| `MULTI_SCALE_TEST` | `False` | Toggles the 100->1000 progressive scaling load test on `main.py` execution. |
| `FAILURE_TEST` | `False` | Toggles the 5-phase formal fault tolerance simulation. |
| `STRATEGY_COMPARISON_TEST` | `False` | Toggles the benchmark comparing round-robin, least connections, and load aware. |

## 5. How to Run Each New Feature

**Multi-Scale Load Test**
1. Open `config.py` and set `MULTI_SCALE_TEST = True` (ensure others are False).
2. Run `python main.py` in the terminal.

**Failure Simulation Test**
1. Open `config.py` and set `FAILURE_TEST = True` (ensure others are False).
2. Run `python main.py` in the terminal.

**Strategy Comparison Test**
1. Open `config.py` and set `STRATEGY_COMPARISON_TEST = True` (ensure others are False).
2. Run `python main.py` in the terminal.

**FastAPI Web Server**
1. Run `python app.py` to start the Uvicorn web server.
2. The API will be available at `http://localhost:8000`.

**Testing the API Endpoints**
```bash
# Ask a question
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain distributed caching"}'

# Check cluster health
curl http://localhost:8000/health

# Change load balancer strategy
curl -X POST http://localhost:8000/load-balancer/strategy \
     -H "Content-Type: application/json" \
     -d '{"strategy": "round_robin"}'
```

## 6. Kaggle Compatibility Notes
- **Web Server (`app.py`)**: Kaggle Notebooks block inbound HTTP traffic, so `app.py` is primarily meant to be run locally or on a standard cloud VM (like AWS EC2). However, you can run it in the Kaggle background and use `requests` locally within the same notebook to ping `localhost:8000`.
- **Tests**: All terminal tests (`main.py`) remain 100% Kaggle compatible.
- **Mock vs Real Modes**: GPU utilization stats in the API and Metrics report automatically adapt. Locally, they show "Simulated CPU Node" and omit VRAM. On Kaggle, they will pull real NVIDIA T4 VRAM utilization.

## 7. What Was NOT Changed
- The underlying architecture of `GPUWorker`, `LoadBalancer`, `Scheduler`, `HealthMonitor`, and RAG integration (`rag/ingestor.py`, `rag/retriever.py`) were left completely untouched. 
- All existing function signatures and classes were preserved to guarantee stability.

## 8. Known Limitations
- FastAPI startup blocks until `ingest_documents` finishes if the ChromaDB vector database is empty.

## 9. YouTube Demo Flow
1. **Initial Setup**: Run `python app.py` and show the clear startup logs (DB ingestion, loading workers, system ready).
2. **API Interaction**: Send a complex RAG query via `curl` to the `/query` endpoint and show the detailed JSON response indicating which GPU processed it.
3. **Scaling Test**: Switch to `main.py` with `MULTI_SCALE_TEST = True` and show the ASCII summary table generating 100 to 1000 users.
4. **Strategy Comparison**: Switch to `main.py` with `STRATEGY_COMPARISON_TEST = True` and show the load-aware algorithm outperforming round-robin.
5. **Fault Tolerance**: Run `main.py` with `FAILURE_TEST = True` and show the 5-phase console output proving 0 dropped requests when Worker 1 is killed.
