"""
NOTE: Primary LLM inference lives inside GPUWorker.process() to keep
the model pinned to its GPU. This module provides a standalone
inference function for cases where a caller needs LLM access
without going through the full worker pipeline (e.g., testing).
It will use cuda:0 only — for multi-GPU use GPUWorker directly.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_DTYPE

_model     = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _model     = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, torch_dtype=LLM_DTYPE, device_map=device
        )
        _model.eval()

def run_llm(query: str, context: str) -> str:
    _load_model()
    prompt = (f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=LLM_MAX_NEW_TOKENS,
                              temperature=LLM_TEMPERATURE, do_sample=True)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
