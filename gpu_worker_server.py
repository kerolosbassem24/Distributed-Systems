"""
gpu_worker_server.py  —  Standalone GPU Worker Node
====================================================
Runs a FastAPI server that:
  1. Loads the LLM onto a real CUDA GPU (or falls back to CPU mock mode)
  2. Exposes  GET  /health   — health check for the scheduler
  3. Exposes  POST /process  — runs RAG + LLM inference, streams tokens back
  4. Auto-registers itself with the central scheduler (scheduler.py on port 9000)

Usage:
    python gpu_worker_server.py                        # GPU 0, port 8001
    python gpu_worker_server.py --gpu-id 1 --port 8002
    python gpu_worker_server.py --mock --port 8003     # CPU simulation (no GPU needed)
"""

import argparse
import asyncio
import logging
import os
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Make sure project root is on the path ─────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    LLM_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_DO_SAMPLE,
    LLM_DTYPE,
    GPU_MEMORY_FRACTION,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gpu_worker_server")

# ── CLI arguments (parsed early so globals are set before FastAPI startup) ─────
parser = argparse.ArgumentParser(description="GPU Worker Node")
parser.add_argument("--gpu-id",    type=int,  default=0,             help="CUDA device index (default: 0)")
parser.add_argument("--port",      type=int,  default=8001,          help="Port to listen on (default: 8001)")
parser.add_argument("--worker-id", type=str,  default=None,          help="Override worker ID (default: gpu-worker-<gpu-id>)")
parser.add_argument("--scheduler", type=str,  default="http://localhost:9000", help="Scheduler URL")
parser.add_argument("--mock",      action="store_true",              help="Run in CPU mock mode (no real GPU needed)")
args, _ = parser.parse_known_args()

WORKER_PORT   = args.port
GPU_ID        = args.gpu_id
MOCK_MODE     = args.mock
if not MOCK_MODE:
    try:
        import torch
        if not torch.cuda.is_available():
            MOCK_MODE = True
    except ImportError:
        MOCK_MODE = True
SCHEDULER_URL = args.scheduler
WORKER_ID     = args.worker_id or ("mock-cpu-node" if MOCK_MODE else f"gpu-worker-{GPU_ID}")

# ── Global model / tokenizer (loaded once at startup) ─────────────────────────
model     = None
tokenizer = None
device    = "cpu" if MOCK_MODE else f"cuda:{GPU_ID}"
gpu_name  = "Simulated CPU Node" if MOCK_MODE else "Loading..."

# ── Pydantic models ───────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    query: str

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title=f"GPU Worker — {WORKER_ID}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load model & register ───────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, gpu_name

    if MOCK_MODE:
        logger.info(f"[{WORKER_ID}] Starting in CPU MOCK mode — no real model loaded.")
        gpu_name = "Simulated CPU Node"
    else:
        logger.info(f"[{WORKER_ID}] Loading model '{LLM_MODEL_NAME}' onto {device}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer as HFTokenizer

            # Limit GPU memory usage
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=GPU_ID)

            tokenizer = HFTokenizer.from_pretrained(LLM_MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=LLM_DTYPE,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            model.eval()

            gpu_name = torch.cuda.get_device_name(GPU_ID)
            mem_gb   = torch.cuda.get_device_properties(GPU_ID).total_memory / 1e9
            logger.info(f"[{WORKER_ID}] Model ready on {gpu_name} ({mem_gb:.1f} GB)")

        except Exception as e:
            logger.error(f"[{WORKER_ID}] Failed to load model: {e}")
            logger.warning(f"[{WORKER_ID}] Falling back to MOCK mode.")
            gpu_name = "Fallback CPU Node"

    # Launch background task to register with scheduler
    asyncio.create_task(register_with_scheduler())


async def register_with_scheduler():
    """Retry loop — keeps trying until it successfully registers."""
    register_url = f"{SCHEDULER_URL}/register"
    payload = {
        "worker_id": WORKER_ID,
        "base_url":  f"http://localhost:{WORKER_PORT}",
        "gpu_name":  gpu_name,
    }
    while True:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(register_url, json=payload)
                if resp.status_code == 200:
                    logger.info(f"[{WORKER_ID}] Registered with scheduler at {SCHEDULER_URL}")
                    return
                else:
                    logger.warning(f"[{WORKER_ID}] Registration returned HTTP {resp.status_code}, retrying...")
        except Exception as e:
            logger.warning(f"[{WORKER_ID}] Scheduler unreachable ({e}), retrying in 3s...")
        await asyncio.sleep(3)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    gpu_info = {}
    if not MOCK_MODE:
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(GPU_ID) / 1e9
                total     = torch.cuda.get_device_properties(GPU_ID).total_memory / 1e9
                gpu_info  = {
                    "gpu_name":        gpu_name,
                    "memory_used_gb":  round(allocated, 2),
                    "memory_total_gb": round(total, 2),
                    "memory_pct":      round(allocated / total * 100, 1),
                }
        except Exception:
            pass

    return {
        "status":    "ok",
        "worker_id": WORKER_ID,
        "device":    device,
        "mock_mode": MOCK_MODE,
        **gpu_info,
    }


# ── /process ──────────────────────────────────────────────────────────────────
@app.post("/process")
async def process(req: ProcessRequest):
    """
    Run RAG retrieval + LLM inference.
    Streams the generated text back as plain text so the
    scheduler can forward it as NDJSON chunks.
    """

    async def generate_stream():
        start = time.time()

        # ── RAG retrieval ──────────────────────────────────────────────────
        try:
            from rag.retriever import retrieve_context
            context = await asyncio.to_thread(retrieve_context, req.query)
        except Exception as e:
            logger.error(f"RAG error: {e}")
            context = "No relevant context found."

        # ── Build prompt ───────────────────────────────────────────────────
        if context and context != "No relevant context found.":
            prompt = (
                "You are a helpful assistant. Use the context below to answer "
                "the question accurately and concisely.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {req.query}\n\nAnswer:"
            )
        else:
            prompt = (
                "<|system|>\nYou are a smart assistant.\n"
                f"<|user|>\n{req.query}\n<|assistant|>\n"
            )

        # ── Inference ──────────────────────────────────────────────────────
        if MOCK_MODE or model is None:
            await asyncio.sleep(0.4)          # simulate compute delay
            answer = (
                f"[{WORKER_ID} | Mock CPU] "
                f"Simulated answer for: \"{req.query[:80]}\". "
                f"Context snippet: {context[:80]}..."
            )
            yield answer
        else:
            try:
                import torch
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        do_sample=LLM_DO_SAMPLE,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                answer     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                latency_ms = (time.time() - start) * 1000
                logger.info(f"[{WORKER_ID}] Inference done in {latency_ms:.1f}ms")
                yield answer

            except Exception as e:
                logger.error(f"[{WORKER_ID}] Inference error: {e}")
                yield f"[ERROR from {WORKER_ID}]: {str(e)}"

    return StreamingResponse(generate_stream(), media_type="text/plain")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting GPU Worker '{WORKER_ID}' on port {WORKER_PORT} | device={device}")
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting GPU Worker '{WORKER_ID}' on port {WORKER_PORT} | device={device}")
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
