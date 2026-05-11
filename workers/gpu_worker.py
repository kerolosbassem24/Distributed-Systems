import torch
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Add parent dir to sys.path to easily import from config, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
                    LLM_DO_SAMPLE, LLM_DTYPE, GPU_MEMORY_FRACTION)
from rag.retriever import retrieve_context
from common.models import Request, Response

class GPUWorker:
    """
    Represents a single GPU node in the cluster.
    Each worker owns one physical CUDA device and hosts its own
    loaded LLM model instance. All inference is real GPU computation.
    """

    def __init__(self, gpu_id: int, mock_mode: bool = False):
        self.id          = gpu_id
        self.gpu_id      = gpu_id
        self.mock_mode   = mock_mode
        self.device      = f"cuda:{gpu_id}" if not mock_mode else "cpu"
        self.status      = "active"          # "active" | "failed" | "recovering"
        self.active_requests  = 0
        self.total_processed  = 0
        self.total_failed     = 0
        self.latency_history  = []           # last N latencies for avg calc
        self.last_heartbeat   = time.time()
        self._lock            = threading.Lock()

        if not self.mock_mode:
            print(f"[Worker {self.id}] Loading model '{LLM_MODEL_NAME}' onto {self.device}...")

            # Set per-GPU memory limit
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=gpu_id)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model directly onto this GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=LLM_DTYPE,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            gpu_name = torch.cuda.get_device_name(gpu_id)
            mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            print(f"[Worker {self.id}] Ready on {gpu_name} ({mem_total:.1f} GB)")
        else:
            print(f"[Worker {self.id}] Ready in Simulated CPU Mode")

    def process(self, request: Request) -> Response:
        """
        Full pipeline: RAG retrieval → prompt construction → real GPU inference.
        """
        start = time.time()

        with self._lock:
            self.active_requests += 1
        self.last_heartbeat = time.time()

        try:
            # ── Step 1: RAG retrieval ────────────────────────────────
            context = retrieve_context(request.query)

            # ── Step 2: Build prompt ─────────────────────────────────
            prompt = (
                "You are a helpful assistant. Use the context below to answer "
                "the question accurately and concisely.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {request.query}\n\n"
                "Answer:"
            )

            # ── Step 3: Tokenize or Simulate ─────────────────────────
            if self.mock_mode:
                time.sleep(0.5) # Simulate processing delay
                answer = f"[Simulated Output from CPU Worker {self.id}] Based on the context: {context[:50]}..."
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True,
                ).to(self.device)

                # ── Step 4: Real GPU inference ───────────────────────────
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        do_sample=LLM_DO_SAMPLE,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # ── Step 5: Decode — strip the prompt tokens ─────────────
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            latency = time.time() - start

            with self._lock:
                self.active_requests -= 1
                self.total_processed += 1
                self.latency_history.append(latency)
                if len(self.latency_history) > 200:
                    self.latency_history.pop(0)

            return Response(
                id=request.id,
                result=answer,
                latency=latency,
                worker_id=self.id,
                gpu_id=self.gpu_id,
                strategy_used="",        # filled in by load balancer
                success=True,
            )

        except Exception as e:
            latency = time.time() - start
            with self._lock:
                self.active_requests -= 1
                self.total_failed += 1
            return Response(
                id=request.id, result="", latency=latency,
                worker_id=self.id, gpu_id=self.gpu_id,
                strategy_used="", success=False, error=str(e),
            )

    def get_gpu_utilization(self) -> dict:
        """Return real-time GPU memory and utilization stats."""
        if self.mock_mode:
            import psutil
            import random
            mem = psutil.virtual_memory()
            return {
                "gpu_id":        self.gpu_id,
                "device_name":   "Simulated CPU Node",
                "memory_used_gb":    round((mem.used / 1e9) + random.uniform(0, 0.5), 2),
                "memory_reserved_gb": round(mem.total / 1e9, 2),
                "memory_total_gb":    round(mem.total / 1e9, 2),
                "memory_pct":    round(mem.percent + random.uniform(-2, 2), 1),
            }

        props  = torch.cuda.get_device_properties(self.gpu_id)
        allocated = torch.cuda.memory_allocated(self.gpu_id) / 1e9
        reserved  = torch.cuda.memory_reserved(self.gpu_id)  / 1e9
        total     = props.total_memory / 1e9
        return {
            "gpu_id":        self.gpu_id,
            "device_name":   props.name,
            "memory_used_gb":    round(allocated, 2),
            "memory_reserved_gb": round(reserved, 2),
            "memory_total_gb":    round(total, 2),
            "memory_pct":    round(allocated / total * 100, 1),
        }

    def get_load_score(self) -> float:
        """
        Composite load score (0.0 = idle, 1.0 = fully loaded).
        Combines: active request ratio + GPU memory pressure + avg latency norm.
        """
        MAX_CONCURRENT = 8
        req_score = min(self.active_requests / MAX_CONCURRENT, 1.0)
        gpu_stats = self.get_gpu_utilization()
        mem_score = gpu_stats["memory_pct"] / 100.0
        avg_lat   = (sum(self.latency_history[-20:]) / len(self.latency_history[-20:])
                     if self.latency_history else 0.0)
        lat_score = min(avg_lat / 10.0, 1.0)       # normalize against 10s ceiling
        return round(0.4 * req_score + 0.4 * mem_score + 0.2 * lat_score, 4)

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def simulate_failure(self):
        """Force this worker offline — used by fault tolerance tests."""
        self.status = "failed"
        device_str = "CPU" if self.mock_mode else f"cuda:{self.gpu_id}"
        print(f"[Worker {self.id}] [!] FAILURE INJECTED on {device_str}")

    def recover(self):
        self.status = "active"
        self.last_heartbeat = time.time()
        device_str = "CPU" if self.mock_mode else f"cuda:{self.gpu_id}"
        print(f"[Worker {self.id}] [OK] Recovered on {device_str}")
